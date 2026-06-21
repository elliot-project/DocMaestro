"""Throughput benchmark for the MinerU hybrid extraction pipeline.

Measures steady-state per-page processing time (model init excluded via a
warm-up pass) and monitors live GPU utilisation during inference.  Generates:

  1. Per-document latency bar chart (s/page, colour-coded by doc).
  2. GPU utilisation timeline during the benchmark run.
  3. Scale projection — total wall-clock at 1K–1B pages for 1/4/8/16 GPUs.

Multi-GPU strategies
--------------------
The MinerU router launches one worker subprocess per GPU and load-balances
documents across them (data parallel). Each worker owns one GPU. This is the
recommended approach for throughput benchmarking.

  Strategy A — data parallel via router (recommended for throughput):
    # Terminal 1: start the router with 2 local workers
    mineru-router --host 0.0.0.0 --port 8002 --local-gpus 4,5

    # Terminal 2: run the benchmark against the router
    uv run scripts/benchmark_throughput.py \\
        --pdf-dir data/pdfs/ \\
        --backend hybrid-http-client \\
        --server-url http://localhost:8002 \\
        --gpu-memory-utilization 0.6 \\
        --gpu-index 4,5

  Strategy B — tensor parallel (single model split across N GPUs):
    CUDA_VISIBLE_DEVICES=4,5 uv run scripts/benchmark_throughput.py \\
        --pdf-dir data/pdfs/ \\
        --backend hybrid-auto-engine \\
        --tensor-parallel-size 2 \\
        --gpu-memory-utilization 0.6 \\
        --gpu-index 4,5

  Strategy A doubles document throughput.
  Strategy B helps only if a single doc is bottlenecked by single-GPU memory.
"""

import argparse
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

import wandb

matplotlib.use("Agg")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extractor import SUPPORTED_BACKENDS, batch_extract, extract  # noqa: E402
from src.utils.benchmark import init_run  # noqa: E402
from src.utils.config import chunked  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output/plots")

_PAGE_SCALES = [
    1_000, 5_000, 10_000, 50_000,
    100_000, 500_000, 1_000_000, 5_000_000,
    10_000_000, 100_000_000, 1_000_000_000,
]
_GPU_COUNTS  = [1, 4, 8, 16]
_GPU_COLORS  = {1: "#4fc3f7", 4: "#81c784", 8: "#ffb74d", 16: "#f06292"}
_TIME_REFS   = [
    ("1 min",    60),
    ("1 hour",   3_600),
    ("1 day",    86_400),
    ("1 week",   604_800),
    ("1 month",  2_592_000),
    ("1 year",   31_536_000),
]

# Palette for per-document bars
_DOC_PALETTE = [
    "#4fc3f7", "#81c784", "#ffb74d", "#f06292", "#ce93d8",
    "#80deea", "#a5d6a7", "#fff176", "#ffcc80", "#ef9a9a",
    "#b39ddb", "#80cbc4", "#f48fb1", "#bcaaa4", "#90caf9",
]


# ── GPU utilisation monitor ───────────────────────────────────────────────────

class _GpuMonitor:
    """Background thread that polls nvidia-smi every second."""

    def __init__(self, gpu_indices: list[int], interval: float = 1.0) -> None:
        self.gpu_indices = gpu_indices
        self.interval    = interval
        self.utilizations: list[float] = []
        self.timestamps:   list[float] = []
        self._stop = threading.Event()
        self._t0   = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        id_str = ",".join(str(g) for g in self.gpu_indices)
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={id_str}",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout=3,
                )
                utils = [float(x) for x in out.decode().strip().splitlines() if x.strip()]
                if utils:
                    self.utilizations.append(float(np.mean(utils)))
                    self.timestamps.append(time.perf_counter() - self._t0)
            except Exception:
                pass
            self._stop.wait(self.interval)

    @property
    def mean(self) -> float:
        return float(np.mean(self.utilizations)) if self.utilizations else 0.0

    @property
    def peak(self) -> float:
        return float(np.max(self.utilizations)) if self.utilizations else 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _human_time(seconds: float) -> str:
    if seconds < 60:      return f"{seconds:.1f}s"
    if seconds < 3_600:   return f"{seconds/60:.1f} min"
    if seconds < 86_400:  return f"{seconds/3_600:.1f} h"
    if seconds < 604_800: return f"{seconds/86_400:.1f} days"
    if seconds < 2_592_000: return f"{seconds/604_800:.1f} wk"
    if seconds < 31_536_000: return f"{seconds/2_592_000:.1f} mo"
    return f"{seconds/31_536_000:.2f} yr"


def _human_pages(n: int) -> str:
    if n >= 1_000_000_000: return f"{n//1_000_000_000}B"
    if n >= 1_000_000:     return f"{n//1_000_000}M"
    if n >= 1_000:         return f"{n//1_000}K"
    return str(n)


# ── Plot 1: per-document latency ──────────────────────────────────────────────

def _plot_latency(
    records: list[dict],   # [{doc, pages, secs_per_page, pages_per_sec, total_s}]
    backend: str,
    gpu_mem_util: float,
) -> Path:
    fig, ax = plt.subplots(figsize=(max(8, len(records) * 0.9 + 2), 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    xs     = np.arange(len(records))
    pps    = [r["pages_per_sec"] for r in records]
    labels = [r["doc"] for r in records]
    colors = [_DOC_PALETTE[i % len(_DOC_PALETTE)] for i in range(len(records))]

    bars = ax.bar(xs, pps, color=colors, alpha=0.88, width=0.65)

    total_pages = sum(r["pages"] for r in records)
    total_time = sum(r["total_s"] for r in records)
    mean_pps = total_pages / total_time if total_time > 0 else 0
    std_pps  = float(np.std(pps))

    ax.axhline(mean_pps, color="#ffb74d", linewidth=2,
               label=f"mean {mean_pps:.2f} pages/sec")
    ax.axhspan(mean_pps - std_pps, mean_pps + std_pps,
               color="#ffb74d", alpha=0.12, label="±1 σ")

    for bar, v, rec in zip(bars, pps, records):
        ax.text(bar.get_x() + bar.get_width() / 2, v + (max(pps)*0.01),
                f"{v:.2f}\n({rec['pages']}p)",
                ha="center", va="bottom", fontsize=7.5,
                color="#ddd", fontfamily="monospace")

    ppm = mean_pps * 60.0
    ax.set_title(
        f"MinerU per-doc throughput — {backend}  |  "
        f"gpu_mem_util={gpu_mem_util}\n"
        f"mean {mean_pps:.2f} pages/sec  ·  {ppm:.1f} pages/min  ·  "
        f"{ppm*60:.0f} pages/hour",
        color="white", fontsize=10, pad=10,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8, color="#aaa")
    ax.set_ylabel("Pages / second", color="#aaa")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(facecolor="#2a2a4e", edgecolor="#555", labelcolor="white", fontsize=9)

    out = OUTPUT_DIR / f"benchmark_{backend}_throughput.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# ── Plot 2: GPU utilisation timeline ─────────────────────────────────────────

def _plot_gpu_util(
    monitor: _GpuMonitor,
    doc_boundaries: list[tuple[float, str]],  # [(wall_time, doc_name)]
    backend: str,
    gpu_indices: list[int] | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    ts   = np.array(monitor.timestamps)
    util = np.array(monitor.utilizations)
    ax.fill_between(ts, util, alpha=0.3, color="#4fc3f7")
    ax.plot(ts, util, color="#4fc3f7", linewidth=1.4, label="GPU util %")
    ax.axhline(monitor.mean, color="#ffb74d", linewidth=1.5, linestyle="--",
               label=f"mean {monitor.mean:.1f}%")

    # Vertical lines between documents
    for t, name in doc_boundaries:
        ax.axvline(t, color="#555", linewidth=0.8, linestyle=":")
        ax.text(t + 1, 95, name[:14], color="#888", fontsize=7,
                fontfamily="monospace", va="top")

    ax.set_ylim(0, 102)
    ax.set_xlabel("Wall-clock time (s)", color="#aaa")
    ax.set_ylabel("GPU utilisation (%)", color="#aaa")
    gpu_label = f"GPU(s) {gpu_indices}" if gpu_indices else "GPU"
    ax.set_title(
        f"GPU utilisation during benchmark — {backend}  |  {gpu_label}"
        f"  |  mean {monitor.mean:.1f}%  peak {monitor.peak:.1f}%",
        color="white", fontsize=10, pad=10,
    )
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(facecolor="#2a2a4e", edgecolor="#555", labelcolor="white", fontsize=9)

    out = OUTPUT_DIR / f"benchmark_{backend}_gpu_util.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# ── Plot 3: scale projection ──────────────────────────────────────────────────

def _plot_projection(
    mean_spp: float,
    std_spp: float,
    backend: str,
    gpu_mem_util: float,
) -> Path:
    pages = np.array(_PAGE_SCALES, dtype=float)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for ngpu in _GPU_COUNTS:
        spp    = mean_spp / ngpu
        total  = pages * spp
        std_t  = pages * (std_spp / ngpu)
        color  = _GPU_COLORS[ngpu]
        ax.fill_between(pages, total - std_t, total + std_t, alpha=0.10, color=color)
        ax.plot(pages, total, color=color, linewidth=2.2,
                label=f"{ngpu} GPU{'s' if ngpu > 1 else ''}")
        ax.annotate(
            f"{ngpu}G: {_human_time(total[-1])}",
            xy=(pages[-1], total[-1]),
            xytext=(8, 0), textcoords="offset points",
            fontsize=8.5, color=color, va="center", fontfamily="monospace",
        )

    for label, ref_s in _TIME_REFS:
        lo = pages[0] * mean_spp
        hi = pages[-1] * mean_spp
        if lo <= ref_s <= hi:
            ax.axhline(ref_s, color="#555", linewidth=0.8, linestyle="--", zorder=0)
            ax.text(pages[0] * 1.1, ref_s * 1.15, label,
                    color="#777", fontsize=8, fontfamily="monospace")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(pages[0] * 0.7, pages[-1] * 3)
    ax.set_xticks(pages)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: _human_pages(int(v))))
    y_ticks  = [60, 3_600, 86_400, 604_800, 2_592_000, 31_536_000]
    y_labels = ["1 min", "1 h", "1 day", "1 wk", "1 mo", "1 yr"]
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: y_labels[y_ticks.index(v)] if v in y_ticks else ""))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="both", which="both", colors="#aaa", labelsize=9)
    ax.tick_params(axis="x", which="major", rotation=35)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(which="major", color="#333", linewidth=0.6)
    ax.grid(which="minor", visible=False)

    ppm = 60.0 / mean_spp
    pps = 1.0 / mean_spp if mean_spp > 0 else 0
    ax.set_title(
        f"MinerU scale projection — {backend}  |  gpu_mem_util={gpu_mem_util}\n"
        f"baseline 1 GPU: {pps:.2f} pages/sec  "
        f"({ppm:.1f} p/min  ·  {ppm*60:.0f} p/hr)",
        color="white", fontsize=11, pad=12,
    )
    ax.set_xlabel("Document scale (pages)", color="#aaa", fontsize=10)
    ax.set_ylabel("Total wall-clock time", color="#aaa", fontsize=10)
    ax.legend(facecolor="#2a2a4e", edgecolor="#555", labelcolor="white",
              fontsize=10, loc="upper left")

    out = OUTPUT_DIR / f"benchmark_{backend}_scale_projection.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MinerU throughput benchmark with GPU utilisation monitoring."
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--pdf",     type=Path, help="Single PDF to benchmark.")
    grp.add_argument("--pdf-dir", type=Path, help="Directory of PDFs (all *.pdf).")
    parser.add_argument("--backend", default="hybrid-auto-engine",
                        choices=sorted(SUPPORTED_BACKENDS))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        metavar="F",
                        help="vLLM KV-cache fraction (default 0.85). "
                             "MinerU default is 0.5.")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Timed passes per document (default 1).")
    parser.add_argument("--gpu-index", type=str, default=None,
                        metavar="IDX",
                        help="Physical GPU index or comma-separated list for multi-GPU "
                             "(e.g. '4' or '4,5'). Sets CUDA_VISIBLE_DEVICES and "
                             "configures nvidia-smi monitoring (utilisation is averaged "
                             "across all listed GPUs). Defaults to CUDA_VISIBLE_DEVICES "
                             "or 0 if unset.")
    parser.add_argument("--server-url", type=str, default=None,
                        help="VLM server or mineru-router base URL for http-client "
                             "backends (e.g. 'http://localhost:8002'). Required when "
                             "--backend is hybrid-http-client.")
    parser.add_argument("--tensor-parallel-size", type=int, default=None, metavar="N",
                        help="Number of GPUs to use for vLLM tensor parallelism. "
                             "Requires CUDA_VISIBLE_DEVICES (or --gpu-index) to expose N "
                             "GPUs. No-op for pipeline and http-client backends. "
                             "See Strategy B in the module docstring.")
    parser.add_argument("--batch-size", type=int, default=None, metavar="N",
                        help="Number of documents to batch together (pipeline backend only).")
    parser.add_argument("--window-size", type=int, default=None, metavar="N",
                        help="MinerU processing window size.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    # Collect PDFs
    if args.pdf:
        pdfs = [args.pdf]
    else:
        pdfs = sorted(args.pdf_dir.glob("*.pdf"))
        if not pdfs:
            parser.error(f"No PDFs found in {args.pdf_dir}")

    # Exclude very large files (>30 MB) to keep the benchmark manageable
    pdfs = [p for p in pdfs if p.stat().st_size < 30 * 1024 * 1024]
    if not pdfs:
        parser.error("No eligible PDFs (all exceeded size limit).")
    print(f"Benchmarking {len(pdfs)} PDF(s) with {args.backend}, "
          f"gpu_mem_util={args.gpu_memory_utilization}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gmu = args.gpu_memory_utilization

    run = init_run(
        pdf_name="multi-doc" if len(pdfs) > 1 else pdfs[0].name,
        backend=args.backend,
        extra_config={
            "benchmark_docs": len(pdfs),
            "benchmark_repeats": args.repeats,
            "gpu_memory_utilization": gmu,
            "title_aided": False,
        },
    )

    # ── Warm-up ── loads model into GPU; excluded from timing ─────────────────
    warmup_pdf = pdfs[0]
    print(f"\n[warmup] {warmup_pdf.name} (pages 0-2) …")
    t_wu0 = time.perf_counter()
    wu = extract(
        pdf_path=warmup_pdf,
        output_dir=args.data_dir / warmup_pdf.stem / args.backend / "warmup",
        backend=args.backend,
        start_page=0, end_page=2,
        server_url=args.server_url,
        gpu_memory_utilization=gmu,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    init_overhead = time.perf_counter() - t_wu0 - wu.duration_seconds
    print(f"[warmup] done — reported {wu.duration_seconds:.1f}s, "
          f"init overhead ≈ {init_overhead:.1f}s")

    # Resolve physical GPU indices for nvidia-smi and CUDA_VISIBLE_DEVICES.
    import os
    if args.gpu_index is not None:
        gpu_indices = [int(x.strip()) for x in args.gpu_index.split(",") if x.strip()]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_indices)
    else:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        gpu_indices = [int(x.strip()) for x in cvd.split(",") if x.strip()] if cvd else [0]
    print(f"[monitor] polling GPU(s) {gpu_indices} via nvidia-smi (util averaged across all)")

    # ── Timed benchmark runs ──────────────────────────────────────────────────
    monitor = _GpuMonitor(gpu_indices=gpu_indices, interval=0.5)
    monitor.start()

    records: list[dict] = []
    doc_boundaries: list[tuple[float, str]] = []
    bench_t0 = time.perf_counter()

    use_batching = args.backend == "pipeline" and args.batch_size is not None and args.batch_size > 1
    if args.window_size is not None:
        os.environ["MINERU_PROCESSING_WINDOW_SIZE"] = str(args.window_size)

    for rep in range(args.repeats):
        out_root = args.data_dir / "bench_sharded" / args.backend / f"bench_r{rep}"

        if use_batching:
            for chunk in chunked(pdfs, args.batch_size):
                print(f"  [{rep+1}/{args.repeats}] Batch processing {[p.name for p in chunk]} …", flush=True)
                chunk_results = batch_extract(
                    pdf_paths=chunk,
                    output_dir=out_root,
                    backend=args.backend,
                    server_url=args.server_url,
                    gpu_memory_utilization=gmu,
                    tensor_parallel_size=args.tensor_parallel_size,
                )
                for pdf, result in zip(chunk, chunk_results):
                    doc_name = pdf.stem
                    t_start  = time.perf_counter() - bench_t0
                    doc_boundaries.append((t_start, doc_name))
                    if isinstance(result, Exception):
                        print(f"  [error] {doc_name}: {result}")
                        continue

                    spp = result.duration_seconds / max(result.page_count, 1)
                    pps = max(result.page_count, 1) / result.duration_seconds
                    records.append({
                        "doc":           doc_name,
                        "pages":         result.page_count,
                        "secs_per_page": spp,
                        "pages_per_sec": pps,
                        "total_s":       result.duration_seconds,
                        "rep":           rep,
                    })
                    print(f"    {doc_name}: {result.duration_seconds:.1f}s / {result.page_count}p = "
                          f"{pps:.2f} pages/sec  ({pps*60:.1f} p/min)")

                    wandb.log({
                        "bench/doc":           doc_name,
                        "bench/rep":           rep,
                        "bench/pages":         result.page_count,
                        "bench/total_s":       result.duration_seconds,
                        "bench/pages_per_sec": pps,
                        "bench/pages_per_min": pps * 60.0,
                    })
        else:
            for pdf in pdfs:
                doc_name = pdf.stem
                run_dir  = args.data_dir / doc_name / args.backend / f"bench_r{rep}"
                t_start  = time.perf_counter() - bench_t0
                doc_boundaries.append((t_start, doc_name))

                print(f"  [{rep+1}/{args.repeats}] {doc_name} …", end=" ", flush=True)
                result = extract(
                    pdf_path=pdf,
                    output_dir=run_dir,
                    backend=args.backend,
                    server_url=args.server_url,
                    gpu_memory_utilization=gmu,
                    tensor_parallel_size=args.tensor_parallel_size,
                )
                spp = result.duration_seconds / max(result.page_count, 1)
                pps = max(result.page_count, 1) / result.duration_seconds
                records.append({
                    "doc":           doc_name,
                    "pages":         result.page_count,
                    "secs_per_page": spp,
                    "pages_per_sec": pps,
                    "total_s":       result.duration_seconds,
                    "rep":           rep,
                })
                print(f"{result.duration_seconds:.1f}s / {result.page_count}p = "
                      f"{pps:.2f} pages/sec  ({pps*60:.1f} p/min)")

                wandb.log({
                    "bench/doc":           doc_name,
                    "bench/rep":           rep,
                    "bench/pages":         result.page_count,
                    "bench/total_s":       result.duration_seconds,
                    "bench/pages_per_sec": pps,
                    "bench/pages_per_min": pps * 60.0,
                })

    monitor.stop()
    total_wall = time.perf_counter() - bench_t0
    total_pages = sum(r["pages"] for r in records)

    # ── Aggregate stats ───────────────────────────────────────────────────────
    all_pps  = [r["pages_per_sec"] for r in records]
    all_spp  = [r["secs_per_page"] for r in records]
    mean_spp = float(np.mean(all_spp))
    std_spp  = float(np.std(all_spp))
    mean_pps = total_pages / total_wall if total_wall > 0 else 0
    std_pps  = float(np.std(all_pps))
    ppm      = mean_pps * 60.0

    print("\n── Aggregate Results ──────────────────────────────────────────")
    print(f"  Documents        : {len(pdfs)}  ×  {args.repeats} repeat(s)")
    print(f"  Total pages      : {total_pages}")
    print(f"  Wall-clock total : {total_wall:.1f}s")
    print(f"  pages/sec mean   : {mean_pps:.4f}  (std {std_pps:.4f})")
    print(f"  Pages / min      : {ppm:.2f}")
    print(f"  Pages / hour     : {ppm*60:.0f}")
    print(f"  GPU util mean    : {monitor.mean:.1f}%  peak {monitor.peak:.1f}%")
    print(f"  Model-init       : {init_overhead:.1f}s (excluded)")

    wandb.log({
        "bench/mean_pages_per_sec":    mean_pps,
        "bench/std_pages_per_sec":     std_pps,
        "bench/pages_per_min":         ppm,
        "bench/pages_per_hour":        ppm * 60,
        "bench/total_pages":           total_pages,
        "bench/gpu_util_mean_pct":     monitor.mean,
        "bench/gpu_util_peak_pct":     monitor.peak,
        "bench/model_init_s":          init_overhead,
        "bench/gpu_memory_utilization": gmu,
    })

    # ── Plots ─────────────────────────────────────────────────────────────────
    lat_path  = _plot_latency(records, args.backend, gmu)
    util_path = _plot_gpu_util(monitor, doc_boundaries, args.backend, gpu_indices)
    proj_path = _plot_projection(mean_spp, std_spp, args.backend, gmu)

    print(f"\n  Latency plot    : {lat_path}")
    print(f"  GPU util plot   : {util_path}")
    print(f"  Projection plot : {proj_path}")

    wandb.log({
        "bench/latency_chart":    wandb.Image(str(lat_path)),
        "bench/gpu_util_chart":   wandb.Image(str(util_path)),
        "bench/projection_chart": wandb.Image(str(proj_path)),
    })
    wandb.finish()


if __name__ == "__main__":
    main()
