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
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import chunked, collect_pdfs  # noqa: E402

_MAIN = Path(__file__).resolve().parents[1] / "main.py"
_DEFAULT_CONFIG = Path("configs/extractor/default.yaml")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _stage_shards(shards: list[list[Path]]) -> list[Path]:
    """Create temporary directories with symlinks for each shard's PDFs.

    Args:
        shards: List of per-shard PDF path lists.

    Returns:
        List of temporary directory paths (one per shard).
    """
    shard_dirs = []
    for i, shard in enumerate(shards):
        shard_dir = Path(tempfile.mkdtemp(prefix=f"mineru_shard_{i}_"))
        for pdf in shard:
            link = shard_dir / pdf.name
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


def _stream_proc(proc: subprocess.Popen, log_file: Path, gpu_id: str) -> None:
    """Write proc stdout to log_file and echo each line prefixed with GPU id.

    Args:
        proc: Running subprocess whose stdout will be streamed.
        log_file: Destination log file path.
        gpu_id: GPU index string used as line prefix.
    """
    assert proc.stdout is not None
    with open(log_file, "w") as fh:
        for line in proc.stdout:
            fh.write(line)
            print(f"[GPU {gpu_id}] {line}", end="")


def _launch_shards(
    gpu_indices: list[str],
    shard_dirs: list[Path],
    args: argparse.Namespace,
) -> None:
    """Launch one main.py subprocess per GPU shard and wait for all to finish.

    Each subprocess runs with CUDA_VISIBLE_DEVICES set to its GPU index.
    stdout/stderr stream to per-shard log files and the terminal in parallel.

    Args:
        gpu_indices: Physical GPU indices (as strings).
        shard_dirs: Staged shard PDF directories, aligned with gpu_indices.
        args: Parsed arguments from parse_args().
    """
    group_id = f"sharded-{int(time.time())}"
    log_dir = Path("output/shard_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

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
        print(
            f"  [GPU {gpu_id}] Starting shard — "
            f"{len(list(shard_dir.glob('*.pdf')))} PDFs → log: {log_file}"
        )
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs.append((proc, log_file, gpu_id))

    threads = [
        threading.Thread(target=_stream_proc, args=(proc, log_file, gpu_id), daemon=True)
        for proc, log_file, gpu_id in procs
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    exit_codes = {gpu_id: proc.wait() for proc, _, gpu_id in procs}

    print("\n── Shard Results ──────────────────────────────────────────────")
    print(f"  WandB group: {group_id}")
    all_ok = True
    for gpu_id, code in exit_codes.items():
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(f"  GPU {gpu_id}: {status}")
        if code != 0:
            all_ok = False
    print(f"  Logs: {log_dir}/")

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

    gpu_indices = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_indices:
        print("ERROR: --gpus must contain at least one GPU index.", file=sys.stderr)
        sys.exit(1)

    all_pdfs = collect_pdfs(args.pdf)
    shards = chunked(all_pdfs, max(1, len(all_pdfs) // len(gpu_indices)))

    shard_dirs = _stage_shards(shards)
    try:
        _launch_shards(gpu_indices, shard_dirs, args)
    finally:
        _cleanup_shards(shard_dirs)


if __name__ == "__main__":
    main()
