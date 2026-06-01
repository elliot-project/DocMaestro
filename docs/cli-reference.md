# CLI Reference

All scripts are run via `uv run`. The primary entry point is `main.py`; the `scripts/`
directory holds specialized runners for benchmarking, dataset sweeps, multi-GPU sharding,
figure enrichment, and standalone visualization.

---

## `main.py`

**Purpose:** Primary orchestrator — PDF or directory → MinerU extraction → XML synthesis →
WandB logging.

```
usage: main.py [-h] [--pdf PATH_OR_DIR] [--config YAML]
               [--batch-size N] [--window-size N] [--no-wandb]
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--pdf` | `data/pdfs/` | Path to a single PDF file or a directory of PDFs. When omitted, defaults to `data/pdfs/`. When a directory is given, all `*.pdf` files are processed in sorted order. |
| `--config` | `configs/extractor/default.yaml` | YAML config file. Controls backend, `server_url`, `gpu_memory_utilization`, title-aided settings, and other extraction parameters. |
| `--batch-size` | `1` | Documents per `do_parse` call. Set `>1` with the `pipeline` backend to enable GPU batching across multiple documents. |
| `--window-size` | `64` | MinerU page processing window. Sets `MINERU_PROCESSING_WINDOW_SIZE`. Lower values reduce peak memory; higher values improve GPU utilization on long documents. |
| `--no-wandb` | `False` | Disable WandB logging for this run. |

### Examples

```bash
# Single PDF with default config (hybrid-http-client backend)
uv run main.py --pdf data/pdfs/macdonals.pdf

# Specify a YAML config
uv run main.py --pdf data/pdfs/macdonals.pdf --config configs/extractor/default.yaml

# All PDFs in a directory, pipeline backend, batch size 4
uv run main.py --pdf data/pdfs/ --batch-size 4

# Reduce page window to lower memory pressure
uv run main.py --pdf data/pdfs/macdonals.pdf --window-size 32

# Disable WandB
uv run main.py --pdf data/pdfs/macdonals.pdf --no-wandb
```

### Config File

`configs/extractor/default.yaml` is the standard config. Fields map directly to `extract()`
parameters. Example structure:

```yaml
backend: hybrid-http-client
server_url: http://localhost:8000
gpu_memory_utilization: 0.75
language: en
title_aided: null
```

### Outputs

For each processed PDF:
- `data/<doc_name>/<backend>/<doc_name>_middle.json`
- `data/<doc_name>/<backend>/<doc_name>_content_list.json`
- `output/<doc_name>/<doc_name>_<backend>.xml`
- WandB artifact: `<doc_name>-<backend>` (type: `document-xml`)

---

## `scripts/run_docvqa2026.py`

**Purpose:** DocVQA-2026 dataset sweep — downloads the VLR-CVC/DocVQA-2026 dataset (val +
test splits), converts image-format documents to PDF, runs batched extraction on every
document, and saves XML + `qa.json` per document.

```
usage: run_docvqa2026.py [-h]
                         [--backend BACKEND]
                         [--batch-size N]
                         [--window-size N]
                         [--gpu-memory-utilization F]
                         [--server-url URL]
                         [--limit N]
                         [--no-resume]
                         [--no-wandb]
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `hybrid-http-client` | MinerU backend for extraction. |
| `--batch-size` | `1` | Documents per `do_parse` call. |
| `--window-size` | `64` | MinerU page processing window (`MINERU_PROCESSING_WINDOW_SIZE`). |
| `--gpu-memory-utilization` | `0.75` | vLLM KV-cache VRAM fraction. |
| `--server-url` | `None` | VLM server URL. Required when `--backend hybrid-http-client`. |
| `--limit` | `None` | Process only the first N documents (useful for debugging). |
| `--no-resume` | `False` | Disable resume mode. By default, documents whose XML already exists are skipped. |
| `--no-wandb` | `False` | Disable WandB logging. |

### Resume Mode

By default the script skips any document whose output XML already exists under
`data/docvqa2026/<doc_id>/`. Pass `--no-resume` to force re-extraction of all documents.

### Outputs

For each document:
- `data/docvqa2026/<doc_id>/<doc_id>_<backend>.xml`
- `data/docvqa2026/<doc_id>/qa.json` — question/answer pairs from the dataset

### Examples

```bash
# Full sweep with HTTP client backend
uv run scripts/run_docvqa2026.py \
  --backend hybrid-http-client \
  --server-url http://localhost:8000

# Debug: first 10 docs only, no WandB
uv run scripts/run_docvqa2026.py --limit 10 --no-wandb
```

---

## `scripts/run_sharded.py`

**Purpose:** Multi-GPU launcher — splits the PDF list into N equal shards and launches one
`main.py` subprocess per GPU with `CUDA_VISIBLE_DEVICES` set to the corresponding device.
All runs share a WandB group for aggregated tracking.

```
usage: run_sharded.py [-h] --gpus N
                      [--pdf PATH]
                      [--config YAML]
                      [--batch-size N]
                      [--no-wandb]
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--gpus` | required | Number of GPUs. Each GPU receives one shard; GPUs are numbered `0` to `N-1`. |
| `--pdf` | `data/pdfs/` | Path to a directory of PDFs to shard across GPUs. |
| `--config` | `configs/extractor/default.yaml` | YAML config passed to each `main.py` subprocess. |
| `--batch-size` | `1` | Batch size forwarded to each `main.py` shard. |
| `--no-wandb` | `False` | Disable WandB logging for all shards. |

### Examples

```bash
# Split data/pdfs/ across 4 GPUs
uv run scripts/run_sharded.py --gpus 4

# Custom PDF directory, 2 GPUs, batch size 2
uv run scripts/run_sharded.py --gpus 2 --pdf data/pdfs/ --batch-size 2
```

---

## `scripts/benchmark_throughput.py`

**Purpose:** Measure steady-state per-page extraction throughput with GPU monitoring and
scale-projection plots. Performs a 3-page warm-up pass (not recorded), then runs background
GPU monitoring via `nvidia-smi` at 0.5 s poll intervals during the benchmark.

```
usage: benchmark_throughput.py [-h]
                                [--backend BACKEND]
                                [--pdf PATH]
                                [--no-wandb]
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `hybrid-auto-engine` | MinerU backend to benchmark. |
| `--pdf` | `data/pdfs/` | Single PDF or directory of PDFs to benchmark. |
| `--no-wandb` | `False` | Disable WandB logging. |

### Warm-Up Behavior

The first document in the list is used as the warm-up target (pages 0–2 only). Warm-up
timing is **not** recorded. After warm-up completes, the full benchmark begins.

### GPU Monitoring

Set `CUDA_VISIBLE_DEVICES` to the GPU you want to benchmark:

```bash
CUDA_VISIBLE_DEVICES=6 uv run scripts/benchmark_throughput.py --backend hybrid-auto-engine
```

### Outputs

Three plots saved to `output/plots/`:
- `<tag>_latency.png` — per-document latency bar chart
- `<tag>_gpu_util.png` — GPU utilization timeline
- `<tag>_scale_projection.png` — log-log scale projection (1K–1B pages, 1/4/8/16 GPUs)

WandB run logged to `mineru-xml-pipeline` with benchmark configuration and aggregate metrics.

### Examples

```bash
# Multi-document benchmark on GPU 6
CUDA_VISIBLE_DEVICES=6 uv run scripts/benchmark_throughput.py \
  --backend hybrid-auto-engine

# Pipeline backend (CPU-only, no CUDA needed)
uv run scripts/benchmark_throughput.py --backend pipeline --no-wandb
```

---

## `scripts/classify_figures.py`

**Purpose:** Figure enrichment — loads an existing XML file, runs SigLIP2 zero-shot
classification on all `<figure>` elements, and writes `category` and `category_confidence`
attributes back into the XML in place.

```
usage: classify_figures.py [-h] --xml PATH
                            [--img-root DIR]
                            [--batch-size N]
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--xml` | required | Path to the XML file to enrich. Modified in place. |
| `--img-root` | inferred from XML path | Directory that contains the figure images referenced by `<img src="..."/>` elements. Defaults to the parent directory of the XML file. |
| `--batch-size` | `16` | Number of images to classify per SigLIP2 forward pass. |

### Outputs

The XML file at `--xml` is updated in place. Each `<figure>` element gains two new
attributes:

```xml
<figure bbox="[62, 200, 940, 500]" category="bar_chart" category_confidence="0.923">
  ...
</figure>
```

Categories come from `LABEL_REGISTRY` in `src/utils/labels.py`.

### Examples

```bash
# Enrich figures in an extracted XML
uv run scripts/classify_figures.py \
  --xml output/macdonals/macdonals_hybrid-http-client.xml

# Custom image root and batch size
uv run scripts/classify_figures.py \
  --xml output/macdonals/macdonals_hybrid-http-client.xml \
  --img-root data/macdonals/hybrid-http-client/ \
  --batch-size 32
```

---

## `scripts/visualize.py`

**Purpose:** Generate visualizations for an already-extracted document. Does not re-run
MinerU — reads from the JSON files written by `main.py`.

```
usage: visualize.py [-h] --doc DOC_ID
                    [--backend BACKEND]
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--doc` | required | Document ID (stem of the PDF filename, e.g. `macdonals`). The script reads JSON from `data/<doc_id>/<backend>/`. |
| `--backend` | auto-detect | Backend subdirectory to load JSON from. If omitted, searches for any `<doc_id>_content_list.json` under `data/<doc_id>/`. |

### Data Discovery

Without `--backend`, the script checks the canonical path
(`data/<doc_id>/hybrid-http-client/<doc_id>_content_list.json`) first, then falls back to a
recursive search under `data/<doc_id>/`.

### Outputs

- `output/<doc_id>/plots/<doc>_page<NNN>_layout_bboxes.png` — per page
- `output/<doc_id>/plots/<doc>_page<NNN>_reading_order.png` — per page
- `output/<doc_id>/plots/<doc>_document_tree.png` — full document
- `output/<doc_id>/plots/<doc>_dom_full.html` — interactive D3 DOM (full doc)
- `output/<doc_id>/plots/<doc>_page<NNN>_dom.html` — interactive D3 DOM (per page)

### Examples

```bash
# Visualize macdonals (auto-detect backend)
uv run scripts/visualize.py --doc macdonals

# Specify backend
uv run scripts/visualize.py --doc macdonals --backend pipeline
```
