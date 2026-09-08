"""Multi-GPU work-queue dispatcher.

Launches one long-lived `main.py --worker` subprocess per GPU and feeds PDFs
from a single shared pending queue: whichever worker finishes first gets the
next document. This replaces static up-front sharding (which silently dropped
PDFs whenever the count wasn't evenly divisible by the GPU count) with a
self-balancing queue where every task is dispatched exactly once.

Why subprocesses instead of multiprocessing?
  Each subprocess gets a clean CUDA context before any model loads, avoiding
  device-assignment races. Model init cost is paid once per process and the
  worker then serves PDFs for the lifetime of the run.
  WandB is handled independently per worker — all runs share a group ID so
  they appear together in the WandB UI.

Backends:
  - pipeline: each subprocess runs independently on its GPU.
  - hybrid-auto-engine: each subprocess loads the VLM model on its GPU.
  - hybrid-http-client: point --config at a server_url; no GPU needed locally,
    but you still benefit from N concurrent request streams.

Protocol:
  Workers receive absolute PDF paths on stdin (one per line) and emit exactly
  one ``@@OMNIDOC@@ {json}`` line per document on stdout. All human-readable
  output (MinerU/vLLM logs, the worker's own progress prints) goes to stderr,
  which is teed to a per-GPU log file under output/shard_logs/.

Usage:
    uv run scripts/run_sharded.py --gpus 0,1 --pdf data/pdfs/ --no-wandb
    uv run scripts/run_sharded.py --gpus 0,1,2,3 --pdf data/pdfs/ \\
        --config configs/extractor/pipeline.yaml --batch-size 4
    uv run scripts/run_sharded.py --gpus 0,1 --pdf data/pdfs/ --retry-failed
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
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

from src.pipeline.dispatcher import run_dispatch  # noqa: E402
from src.pipeline.protocol import decode_status  # noqa: E402
from src.utils.config import collect_pdfs, load_config  # noqa: E402
from src.utils.manifest import Manifest, compute_run_id, manifest_path  # noqa: E402
from src.utils.paths import doc_rel_dir  # noqa: E402

_MAIN = Path(__file__).resolve().parents[1] / "main.py"
_DEFAULT_CONFIG = Path("configs/extractor/default.yaml")


# ── Worker subprocess handle ─────────────────────────────────────────────────


class SubprocessWorker:
    """Dispatcher-facing handle around a `main.py --worker` subprocess.

    Implements the `src.pipeline.dispatcher.Worker` protocol: `send` writes
    one PDF path line to the subprocess's stdin, `close` closes stdin so the
    worker's `worker_loop` sees EOF and exits cleanly.

    Also records which GPU each in-flight task was sent to in a shared dict,
    so `on_result` (which only receives the result payload, not the worker
    id) can still attribute completions to the right per-GPU progress bar.
    """

    def __init__(self, gpu_id: str, proc: subprocess.Popen, task_owner: dict[str, str]) -> None:
        """Wrap a running worker subprocess.

        Args:
            gpu_id: Physical GPU index this worker is pinned to.
            proc: The running `main.py --worker` subprocess.
            task_owner: Shared ``{task: gpu_id}`` map updated on every send,
                so callbacks can look up which GPU handled a given task.
        """
        self.id = gpu_id
        self.proc = proc
        self._task_owner = task_owner

    def send(self, task: str) -> None:
        """Write one PDF path to the worker's stdin and flush immediately."""
        assert self.proc.stdin is not None
        self._task_owner[task] = self.id
        self.proc.stdin.write(task + "\n")
        self.proc.stdin.flush()

    def close(self) -> None:
        """Close stdin so the worker's loop sees EOF and exits."""
        if self.proc.stdin is not None:
            self.proc.stdin.close()


# ── Reader threads ────────────────────────────────────────────────────────────


def _read_stdout(
    worker_id: str, proc: subprocess.Popen, results: "queue.Queue[tuple[str, dict]]"
) -> None:
    """Decode protocol lines from a worker's stdout and push them to results.

    Pushes ``{"event": "died"}`` once stdout closes (EOF), so the dispatcher
    can fail any in-flight task and keep going with the remaining workers.

    Args:
        worker_id: GPU index identifying this worker.
        proc: The worker subprocess.
        results: Shared queue the dispatcher drains.
    """
    assert proc.stdout is not None
    for line in proc.stdout:
        payload = decode_status(line)
        if payload is not None:
            results.put((worker_id, payload))
    results.put((worker_id, {"event": "died"}))


def _read_stderr(proc: subprocess.Popen, log_file: Path) -> None:
    """Tee a worker's stderr (human logs, MinerU/vLLM noise) to a log file.

    Args:
        proc: The worker subprocess.
        log_file: Destination log file path.
    """
    assert proc.stderr is not None
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as fh:
        for line in proc.stderr:
            fh.write(line)
            fh.flush()


# ── Entry point ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Dispatch PDFs across multiple GPU workers from a shared queue."
    )
    parser.add_argument("--gpus", required=True, metavar="IDX",
                        help="Comma-separated physical GPU indices, e.g. '0,1,2,3'.")
    parser.add_argument("--pdf", type=Path, required=True,
                        help="PDF file or directory of PDFs to process.")
    parser.add_argument("--exclude", action="append", default=None, metavar="PATTERN",
                        help="Skip PDFs whose path under --pdf matches PATTERN (repeatable). "
                             "Overrides paths.exclude from the config when given.")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG,
                        help=f"YAML config file passed to each worker. Default: {_DEFAULT_CONFIG}")
    parser.add_argument("--batch-size", type=int, default=None, metavar="N",
                        help="--batch-size forwarded to each worker (pipeline only).")
    parser.add_argument("--window-size", type=int, default=None, metavar="N",
                        help="--window-size forwarded to each worker.")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logging for all workers.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-queue documents previously marked 'failed' in the manifest.")
    return parser.parse_args()


def main() -> None:
    """Collect pending PDFs, spawn one worker per GPU, and dispatch the queue."""
    args = parse_args()
    console = Console()

    gpu_indices = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_indices:
        console.print("[red]ERROR: --gpus must contain at least one GPU index.[/red]")
        sys.exit(1)

    config = load_config(args.config)
    ext = config.extraction

    exclude = args.exclude if args.exclude is not None else config.paths.exclude
    all_pdfs = collect_pdfs(args.pdf, exclude=exclude)

    def doc_id(pdf: Path) -> str:
        return doc_rel_dir(pdf, args.pdf).as_posix()

    run_id = compute_run_id(args.pdf, ext.backend, config.paths.output_dir)
    manifest = Manifest(manifest_path(config.paths.state_dir, run_id))

    all_ids = [doc_id(p) for p in all_pdfs]
    pending_ids = set(manifest.pending(all_ids, retry_failed=args.retry_failed))
    pending_pdfs = [p for p, did in zip(all_pdfs, all_ids) if did in pending_ids]
    tasks = [str(p.resolve()) for p in pending_pdfs]

    console.rule("[bold blue]DocMaestro — Multi-GPU Dispatch[/bold blue]")
    console.print(f"  Total PDFs found:   [bold]{len(all_pdfs)}[/bold]")
    console.print(f"  Already processed:  [yellow]{len(all_pdfs) - len(tasks)}[/yellow]")
    console.print(f"  Remaining to do:    [bold green]{len(tasks)}[/bold green]")
    console.print(f"  GPUs:                [bold]{', '.join(gpu_indices)}[/bold]")
    console.print()

    if not tasks:
        console.print("[bold green]✓ All documents have been processed. Nothing to do.[/bold green]")
        return

    group_id = f"sharded-{int(time.time())}"
    log_dir = Path("output/shard_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    procs: list[subprocess.Popen] = []
    workers: list[SubprocessWorker] = []
    reader_threads: list[threading.Thread] = []
    results: "queue.Queue[tuple[str, dict]]" = queue.Queue()
    task_owner: dict[str, str] = {}

    for gpu_id in gpu_indices:
        cmd = [
            sys.executable, str(_MAIN),
            "--worker",
            "--pdf", str(args.pdf),
            "--input-root", str(args.pdf),
            "--config", str(args.config),
        ]
        if args.batch_size is not None:
            cmd += ["--batch-size", str(args.batch_size)]
        if args.window_size is not None:
            cmd += ["--window-size", str(args.window_size)]
        if args.no_wandb:
            cmd += ["--no-wandb"]
        if args.retry_failed:
            cmd += ["--retry-failed"]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["WANDB_RUN_GROUP"] = group_id
        env["WANDB_JOB_TYPE"] = f"shard-gpu{gpu_id}"
        # vLLM reserves gpu_memory_utilization upfront in its own process; the
        # torch-side layout/OCR work runs outside that budget. Reduce allocator
        # fragmentation so large PDFs don't OOM against the remaining headroom.
        # Respect an explicit user override if one is already set.
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        proc = subprocess.Popen(
            cmd, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        procs.append(proc)
        workers.append(SubprocessWorker(gpu_id, proc, task_owner))

        log_file = log_dir / f"gpu{gpu_id}_{int(time.time())}.log"
        console.print(f"  [bold]GPU {gpu_id}[/bold]: worker started → {log_file}")

        reader_threads.append(
            threading.Thread(target=_read_stdout, args=(gpu_id, proc, results), daemon=True)
        )
        reader_threads.append(
            threading.Thread(target=_read_stderr, args=(proc, log_file), daemon=True)
        )

    for t in reader_threads:
        t.start()

    counts = {"done": 0, "failed": 0}

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
        overall_task = progress.add_task("[bold cyan]Overall", total=len(tasks))
        # Per-GPU share is decided dynamically by the queue, not known up front,
        # so these bars track a live count (total=None) rather than a fixed goal.
        gpu_tasks = {
            gpu_id: progress.add_task(f"  [dim]GPU {gpu_id}[/dim]", total=None)
            for gpu_id in gpu_indices
        }

        def on_result(payload: dict) -> None:
            doc_path = payload.get("doc")
            did = doc_id(Path(doc_path)) if doc_path else None
            owner_gpu = task_owner.pop(doc_path, None) if doc_path else None

            if payload.get("event") == "done":
                counts["done"] += 1
                if did is not None:
                    manifest.mark_done(did, pages=payload.get("pages"), seconds=payload.get("seconds"))
            else:
                counts["failed"] += 1
                error = payload.get("error", "unknown error")
                if did is not None:
                    manifest.mark_failed(did, error)
                progress.console.print(f"  [red]✗[/red] {doc_path}: {error}")

            progress.advance(overall_task)
            if owner_gpu is not None:
                progress.advance(gpu_tasks[owner_gpu])

        try:
            run_dispatch(tasks, workers, results, on_result)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted — terminating workers...[/yellow]")
            for proc in procs:
                proc.terminate()
            for proc in procs:
                proc.wait()
            for t in reader_threads:
                t.join(timeout=5)
            sys.exit(130)

    exit_codes = [proc.wait() for proc in procs]
    for t in reader_threads:
        t.join(timeout=5)

    # ── Summary ───────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold blue]Summary[/bold blue]")
    console.print(f"  WandB group: {group_id}")
    console.print(f"  Done:        [green]{counts['done']}[/green]")
    console.print(f"  Failed:      [red]{counts['failed']}[/red]")
    console.print(f"  Logs:        {log_dir}/")

    if any(code != 0 for code in exit_codes):
        for gpu_id, code in zip(gpu_indices, exit_codes):
            if code != 0:
                console.print(f"  [red]GPU {gpu_id} worker exited with code {code}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
