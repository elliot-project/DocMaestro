# Benchmarking

## Overview

The pipeline integrates two complementary benchmarking layers:

1. **WandB experiment tracking** (`src/doc_maestro/utils/benchmark.py`) — logs metrics for every extraction run
   to the `mineru-xml-pipeline` project. Enables comparing backends, GPUs, and configurations
   across runs in the WandB dashboard.

2. **Throughput benchmark** (`scripts/benchmark_throughput.py`) — measures steady-state
   per-page processing time, monitors live GPU utilization, and generates scale-projection
   plots for capacity planning.

---

## WandB Experiment Tracking

### Project

All runs log to: **`mineru-xml-pipeline`** (WandB project name, hardcoded).

### Initialization

```python
from doc_maestro.utils.benchmark import init_run

run = init_run(
    pdf_name="macdonals.pdf",
    backend="hybrid-auto-engine",
    extra_config={
        "language": "en",
        "start_page": 0,
        "end_page": None,
        "server_url": None,
        "title_aided": False,
    },
)
```

Uses `wandb.init(project="mineru-xml-pipeline", config=config, reinit=True)`.

### Logged Metrics

#### Extraction performance (`log_extraction_metrics`)

| Metric key | Value |
|------------|-------|
| `extraction/<backend>/total_duration_s` | Total wall-clock time (seconds) |
| `extraction/<backend>/latency_per_page_s` | `duration / max(page_count, 1)` |
| `extraction/<backend>/pages` | Page count from `middle.json` |
| `extraction/<backend>/content_items` | Number of content blocks in `content_list.json` |
| `extraction/<backend>/pdf_size_mb` | PDF file size in MB |
| `extraction/backend` | Backend string (e.g. `"hybrid-auto-engine"`) |

#### GPU memory (`log_gpu_memory`)

Called before and after extraction. Iterates over all visible CUDA devices:

| Metric key | Value |
|------------|-------|
| `gpu/<i>/vram_used_gb` | Allocated VRAM in GB (from `torch.cuda.memory_allocated`) |
| `gpu/<i>/vram_total_gb` | Total VRAM in GB |
| `gpu/<i>/vram_utilization` | `used / total` ratio |

#### XML output stats (`log_xml_stats`)

| Metric key | Value |
|------------|-------|
| `xml/size_kb` | Serialized XML size in kilobytes |
| `xml/output_path` | Absolute path to the saved XML file |

### WandB Artifacts

Each processed PDF is logged as a WandB artifact:

```python
artifact = wandb.Artifact(
    name=f"{doc_name}-{backend}",
    type="document-xml",
    description=f"Synthesized XML for {doc_name} using {backend}",
)
artifact.add_file(str(xml_output_path))
wandb_run.log_artifact(artifact)
```

This enables version-controlled access to XML outputs through the WandB UI.

---

## Throughput Benchmark

`scripts/benchmark_throughput.py` measures MinerU's steady-state throughput by:
1. Running a warm-up pass (pages 0–2 of the first document) to load model weights and
   compile CUDA kernels — not counted in timing.
2. Extracting all target documents sequentially, recording per-page latency for each.
3. Monitoring GPU utilization in a background thread throughout.
4. Generating three plots.

### Warm-Up Design

```
[first PDF, pages 0-2]  ← warm-up (discarded)
[all PDFs × N repeats]  ← timed benchmark
```

The warm-up is essential because vLLM's first forward pass triggers:
- CUDA kernel compilation (~20–60 s)
- Model weight loading to VRAM
- KV-cache allocation

Including this in per-page timing would understate steady-state throughput by 10–100×
depending on document size.

### GPU Utilization Monitor

The `_GpuMonitor` class runs a background thread that polls `nvidia-smi` every 0.5 seconds:

```python
monitor = _GpuMonitor(gpu_phys=6)   # physical GPU index
monitor.start()
# ... run benchmark ...
monitor.stop()
readings = monitor.readings          # list of (elapsed_s, util_pct) tuples
```

Physical GPU index is auto-detected from `CUDA_VISIBLE_DEVICES`:
```python
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
gpu_phys = int(cvd.split(",")[0]) if cvd else 0
```

### Document Filtering

The benchmark automatically excludes documents larger than 30 MB to avoid extremely long runs:

```python
pdfs = [p for p in pdfs if p.stat().st_size < 30 * 1024 * 1024]
```

Adjust this threshold with a code edit if needed.

### Generated Plots

Three plots are saved to `output/plots/`:

#### 1. Per-document latency bar chart (`*_latency.png`)
- Horizontal bar chart, one bar per (document, repeat).
- Y-axis: seconds per page.
- Each document is color-coded; bar labels show the exact s/page value.

#### 2. GPU utilization timeline (`*_gpu_util.png`)
- Line chart of GPU utilization (%) vs. elapsed time (seconds).
- Vertical dashed lines mark document boundaries.
- Annotated with document names at the top.
- Shows whether the GPU is saturated during inference or stalling between pages.

#### 3. Scale projection (`*_scale_projection.png`)
- Log-log chart projecting total wall-clock time at 1K–1B pages.
- Four curves for 1, 4, 8, and 16 GPUs.
- Horizontal reference lines for 1 minute, 1 hour, 1 day, 1 week, 1 month, 1 year.
- Based on the median s/page from the benchmark run, divided by GPU count (assumes perfect
  parallelism — the actual limit is I/O and model memory bandwidth).

### Reading Benchmark Results

After the run, the console prints an aggregate summary:

```
══════════════════════════════════════════
 Aggregate Results
══════════════════════════════════════════
  Documents        : 12  ×  1 repeat(s)
  Median s/page    : 1.41 s
  Mean s/page      : 1.53 s
  Pages/min        : 39.2
  Pages/hour       : 2350

  GPU util mean    : 72.4%
  GPU util peak    : 98.0%
══════════════════════════════════════════
```

### Baseline Performance Reference

| Configuration | Median s/page | Pages/min |
|--------------|--------------|-----------|
| `hybrid-auto-engine`, `gpu_memory_utilization=0.5` (default) | ~1.41 s | ~42 |
| `hybrid-auto-engine`, `gpu_memory_utilization=0.75` | measured during benchmark | — |

Note: throughput is primarily bottlenecked by the VLM forward pass time per page, which depends
on page complexity (number of layout regions) and GPU generation. Increasing
`gpu_memory_utilization` allows larger KV-cache batches, which can improve throughput on pages
with many regions.

---

## Running the Throughput Benchmark

```bash
# Single GPU (GPU 6), all docs under 30 MB, 1 repeat
CUDA_VISIBLE_DEVICES=6 uv run scripts/benchmark_throughput.py \
  --pdf-dir data/pdfs/ \
  --backend hybrid-auto-engine \
  --gpu-memory-utilization 0.75 \
  --repeats 1

# Single PDF for quick testing
CUDA_VISIBLE_DEVICES=6 uv run scripts/benchmark_throughput.py \
  --pdf data/pdfs/macdonals.pdf \
  --backend hybrid-auto-engine \
  --repeats 3
```

See [cli-reference.md](cli-reference.md) for all flags.

---

## WandB Authentication

WandB requires authentication. Set your API key before running:

```bash
wandb login          # interactive, stores key in ~/.netrc
# or
export WANDB_API_KEY=<your_key>
```

To disable WandB (offline mode):

```bash
export WANDB_MODE=offline
# or
wandb offline
```
