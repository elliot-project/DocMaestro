"""Multi-GPU sharded extraction launcher.

Splits a directory of PDFs into N equal shards and launches one independent
`main.py` subprocess per GPU, each pinned to its GPU via CUDA_VISIBLE_DEVICES.
Each shard runs the full pipeline (extraction → XML → figure-clf) in parallel.

Why subprocesses instead of multiprocessing?
  Each subprocess gets a clean CUDA context before any model loads, avoiding
  device-assignment races. Model init cost is paid once per process.
  WandB is handled independently per shard — all runs share a group ID so
  they appear together in the WandB UI.

Backends:
  - pipeline: each subprocess runs independently on its GPU.
  - hybrid-auto-engine: each subprocess loads the VLM model on its GPU.
  - hybrid-http-client: point --config at a server_url; no GPU needed locally,
    but you still benefit from N concurrent request streams.

Usage:
    uv run scripts/run_sharded.py --gpus 0,1 --pdf data/pdfs/ --no-wandb
    uv run scripts/run_sharded.py --gpus 0,1,2,3 --pdf data/pdfs/ \\
        --config configs/extractor/pipeline.yaml --batch-size 4
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import chunked, collect_pdfs  # noqa: E402

_MAIN = Path(__file__).resolve().parents[1] / "main.py"
_DEFAULT_CONFIG = Path("configs/extractor/default.yaml")


# ── Output filtering ─────────────────────────────────────────────────────────

# Patterns for noisy MinerU/vLLM internal progress bars we want to hide
_NOISE_RE = re.compile(
    r"(Layout Predict|MFR Predict|OCR-det|OCR-rec|Processing pages"
    r"|Processed prompts|Adding requests"
    r"|it/s\]|toks/s\]"
    r"|\d+%\|[█▏▎▍▌▋▊▉ ]+\|"
    r"|wandb:|Synced \d|Find logs at)"
)

# A completed document line: "✓ docname: N pages in Xs"
_DONE_RE = re.compile(r"✓.*pages in")

# A failed document line
_FAIL_RE = re.compile(r"✗")

# A skipped (already processed) document
_SKIP_RE = re.compile(r"Skipping|already processed", re.IGNORECASE)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _stage_shards(shards: list[list[Path]], base_dir: Path) -> list[Path]:
    """Create temporary directories with symlinks for each shard's PDFs.

    Args:
        shards: List of per-shard PDF path lists.
        base_dir: Original base directory to calculate relative paths from.

    Returns:
        List of temporary directory paths (one per shard).
    """
    shard_dirs = []
    base_res = base_dir.resolve()
    for i, shard in enumerate(shards):
        shard_dir = Path(tempfile.mkdtemp(prefix=f"mineru_shard_{i}_"))
        for pdf in shard:
            try:
                rel = pdf.resolve().relative_to(base_res)
            except ValueError:
                rel = Path(pdf.name)
            link = shard_dir / rel
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                link.symlink_to(pdf.resolve())
        shard_dirs.append(shard_dir)
    return shard_dirs


def _cleanup_shards(shard_dirs: list[Path]) -> None:
    """Remove temporary shard staging directories.

    Args:
        shard_dirs: List of temporary directory paths to remove.
    """
    for d in shard_dirs:
        shutil.rmtree(d, ignore_errors=True)


def _stream_proc(
    proc: subprocess.Popen,
    log_file: Path,
    gpu_id: str,
    progress: Progress,
    gpu_task_id: int,
    overall_task_id: int,
    console: Console,
) -> None:
    """Stream subprocess output, update progress bars, and log to file.

    All output is always written to the log file for debugging. Only
    non-noisy lines are printed above the progress bar. When a document
    completes (✓) or fails (✗), the per-GPU and overall progress bars
    are advanced.

    Args:
        proc: Running subprocess whose stdout will be streamed.
        log_file: Destination log file path.
        gpu_id: GPU index string used as line prefix.
        progress: Shared Rich Progress instance.
        gpu_task_id: Rich task ID for this GPU's progress bar.
        overall_task_id: Rich task ID for the overall progress bar.
        console: Rich Console for printing log lines.
    """
    assert proc.stdout is not None
    with open(log_file, "w") as fh:
        for line in proc.stdout:
            fh.write(line)
            stripped = line.strip()
            if not stripped:
                continue

            # Advance progress on completed/failed/skipped documents
            if _DONE_RE.search(stripped) or _FAIL_RE.search(stripped):
                progress.advance(gpu_task_id)
                progress.advance(overall_task_id)
                # Print the result line above the progress bar
                progress.console.print(f"  [GPU {gpu_id}] {stripped}")
            elif _SKIP_RE.search(stripped):
                progress.advance(gpu_task_id)
                progress.advance(overall_task_id)
            # Suppress noisy lines entirely
            elif _NOISE_RE.search(stripped):
                continue


def _launch_shards(
    gpu_indices: list[str],
    shard_dirs: list[Path],
    shard_sizes: list[int],
    args: argparse.Namespace,
    console: Console,
) -> None:
    """Launch one main.py subprocess per GPU shard and wait for all to finish.

    Each subprocess runs with CUDA_VISIBLE_DEVICES set to its GPU index.
    A Rich progress display tracks per-GPU and overall progress.

    Args:
        gpu_indices: Physical GPU indices (as strings).
        shard_dirs: Staged shard PDF directories, aligned with gpu_indices.
        shard_sizes: Number of PDFs in each shard (aligned with gpu_indices).
        args: Parsed arguments from parse_args().
        console: Rich Console instance.
    """
    group_id = f"sharded-{int(time.time())}"
    log_dir = Path("output/shard_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    total_pdfs = sum(shard_sizes)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:

        # Overall progress bar
        overall_task = progress.add_task(
            "[bold cyan]Overall", total=total_pdfs
        )

        # Per-GPU progress bars
        gpu_tasks: dict[str, int] = {}
        for gpu_id, size in zip(gpu_indices, shard_sizes):
            gpu_tasks[gpu_id] = progress.add_task(
                f"  [dim]GPU {gpu_id}[/dim]", total=size
            )

        procs: list[tuple[subprocess.Popen, Path, str]] = []

        for gpu_id, shard_dir in zip(gpu_indices, shard_dirs):
            cmd = [sys.executable, str(_MAIN), "--pdf", str(shard_dir), "--config", str(args.config)]
            if args.batch_size is not None:
                cmd += ["--batch-size", str(args.batch_size)]
            if args.window_size is not None:
                cmd += ["--window-size", str(args.window_size)]
            if args.no_wandb:
                cmd += ["--no-wandb"]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["WANDB_RUN_GROUP"] = group_id
            env["WANDB_JOB_TYPE"] = f"shard-gpu{gpu_id}"

            log_file = log_dir / f"shard_gpu{gpu_id}_{int(time.time())}.log"
            progress.console.print(
                f"  [bold]GPU {gpu_id}[/bold]: {shard_sizes[gpu_indices.index(gpu_id)]} PDFs → {log_file}"
            )
            proc = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            procs.append((proc, log_file, gpu_id))

        threads = [
            threading.Thread(
                target=_stream_proc,
                args=(proc, log_file, gpu_id, progress, gpu_tasks[gpu_id], overall_task, console),
                daemon=True,
            )
            for proc, log_file, gpu_id in procs
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        exit_codes = {gpu_id: proc.wait() for proc, _, gpu_id in procs}

    # ── Results summary ──────────────────────────────────────────────────
    console.print()
    console.rule("[bold blue]Shard Results[/bold blue]")
    console.print(f"  WandB group: {group_id}")
    all_ok = True
    for gpu_id, code in exit_codes.items():
        if code == 0:
            console.print(f"  GPU {gpu_id}: [green]OK[/green]")
        else:
            console.print(f"  GPU {gpu_id}: [red]FAILED (exit {code})[/red]")
            all_ok = False
    console.print(f"  Full logs: {log_dir}/")

    if not all_ok:
        sys.exit(1)


# ── Entry point ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run main.py across multiple GPUs by sharding the PDF list."
    )
    parser.add_argument("--gpus", required=True, metavar="IDX",
                        help="Comma-separated physical GPU indices, e.g. '0,1,2,3'.")
    parser.add_argument("--pdf", type=Path, required=True,
                        help="PDF file or directory of PDFs to process.")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG,
                        help=f"YAML config file passed to each shard. Default: {_DEFAULT_CONFIG}")
    parser.add_argument("--batch-size", type=int, default=None, metavar="N",
                        help="--batch-size forwarded to each main.py shard (pipeline only).")
    parser.add_argument("--window-size", type=int, default=None, metavar="N",
                        help="--window-size forwarded to each main.py shard.")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logging for all shards.")
    return parser.parse_args()


def main() -> None:
    """Shard PDFs across GPUs and launch parallel main.py subprocesses."""
    args = parse_args()
    console = Console()

    gpu_indices = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_indices:
        console.print("[red]ERROR: --gpus must contain at least one GPU index.[/red]")
        sys.exit(1)

    all_pdfs = collect_pdfs(args.pdf)
    shards = chunked(all_pdfs, max(1, len(all_pdfs) // len(gpu_indices)))
    shard_sizes = [len(s) for s in shards]

    console.rule("[bold blue]OmniDocParser — Multi-GPU[/bold blue]")
    console.print(f"  Total PDFs:  [bold]{len(all_pdfs)}[/bold]")
    console.print(f"  GPUs:        [bold]{', '.join(gpu_indices)}[/bold]")
    for i, (gpu_id, size) in enumerate(zip(gpu_indices, shard_sizes)):
        console.print(f"    GPU {gpu_id}: {size} PDFs")
    console.print()

    shard_dirs = _stage_shards(shards, args.pdf)
    try:
        _launch_shards(gpu_indices, shard_dirs, shard_sizes, args, console)
    finally:
        _cleanup_shards(shard_dirs)


if __name__ == "__main__":
    main()
