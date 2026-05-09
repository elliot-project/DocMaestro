# CLI Reference

All scripts are run via `uv run scripts/<script>.py`. Three scripts cover the full pipeline.

---

## `scripts/run_extraction.py`

**Purpose:** Full pipeline — PDF → MinerU extraction → XML synthesis → WandB logging.

```
usage: run_extraction.py [-h] --pdf PDF
                         [--backend {hybrid-auto-engine,hybrid-http-client,pipeline,vlm-auto-engine}]
                         [--server-url SERVER_URL]
                         [--lang LANG]
                         [--start-page START_PAGE] [--end-page END_PAGE]
                         [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]
                         [--title-aided-api-key KEY]
                         [--title-aided-base-url URL]
                         [--title-aided-model MODEL]
                         [--title-aided-enable-thinking]
```

### Required Arguments

| Flag | Type | Description |
|------|------|-------------|
| `--pdf` | path | Path to a single PDF file, or a directory of PDFs. When a directory is given, all `*.pdf` files are processed in sorted order. |

### Core Extraction Options

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `hybrid-http-client` | MinerU inference backend. |
| `--server-url` | `None` | VLM server base URL, e.g. `http://localhost:8000`. **Required** when `--backend hybrid-http-client`. |
| `--lang` | `en` | OCR language hint (`en`, `ch`, `ja`, etc.). |
| `--start-page` | `0` | Zero-based first page index. |
| `--end-page` | `None` (all) | Zero-based last page index (inclusive). |

### Directory Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data/` | Root directory for raw MinerU JSON outputs. Outputs go to `<data-dir>/<doc_name>/<backend>/`. |
| `--output-dir` | `output/` | Root directory for synthesized XML. XML files go to `<output-dir>/<doc_name>/`. |

### Title-Aided Options

All three `--title-aided-*` flags must be provided together, or none.

| Flag | Default | Description |
|------|---------|-------------|
| `--title-aided-api-key` | `None` | API key for the LLM endpoint. Use `"dummy"` for local servers. |
| `--title-aided-base-url` | `None` | OpenAI-compatible LLM base URL, e.g. `http://localhost:8000/v1`. |
| `--title-aided-model` | `None` | Model name, e.g. `Qwen/Qwen3-VL-8B-Instruct`. |
| `--title-aided-enable-thinking` | `False` | Pass `enable_thinking=True` to the LLM endpoint. Required for Qwen3-Thinking models. |

### Examples

```bash
# Basic extraction with HTTP client backend
uv run scripts/run_extraction.py \
  --pdf data/pdfs/macdonals.pdf \
  --backend hybrid-http-client \
  --server-url http://localhost:8000

# In-process VLM, pages 0-9 only
CUDA_VISIBLE_DEVICES=6 uv run scripts/run_extraction.py \
  --pdf data/pdfs/macdonals.pdf \
  --backend hybrid-auto-engine \
  --start-page 0 --end-page 9

# Batch: all PDFs in data/pdfs/
CUDA_VISIBLE_DEVICES=6 uv run scripts/run_extraction.py \
  --pdf data/pdfs/ \
  --backend hybrid-auto-engine

# With title-aided H1–H4 detection
CUDA_VISIBLE_DEVICES=6 uv run scripts/run_extraction.py \
  --pdf data/pdfs/macdonals.pdf \
  --backend hybrid-auto-engine \
  --title-aided-api-key dummy \
  --title-aided-base-url http://localhost:8000/v1 \
  --title-aided-model Qwen/Qwen3-VL-8B-Instruct
```

### Outputs

For each processed PDF:
- `data/<doc_name>/<backend>/<doc_name>_middle.json` — raw MinerU topology JSON
- `data/<doc_name>/<backend>/<doc_name>_content_list.json` — raw semantic list JSON
- `output/<doc_name>/<doc_name>_<backend>.xml` — synthesized XML
- WandB artifact: `<doc_name>-<backend>` (type: `document-xml`)

---

## `scripts/visualize.py`

**Purpose:** Generate visualizations for an already-extracted document. Does not re-run
MinerU — reads from the JSON files written by `run_extraction.py`.

```
usage: visualize.py [-h] --pdf PDF
                    [--backend BACKEND]
                    [--pages [PAGES ...]]
                    [--data-dir DATA_DIR]
                    [--output-dir OUTPUT_DIR]
                    [--dom]
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--pdf` | required | Path to the source PDF (must exist on disk for page rendering). |
| `--backend` | auto-detect | Backend subdirectory to load JSON from. If omitted, searches for any `<doc_name>_content_list.json` under `<data-dir>/<doc_name>/`. |
| `--pages` | first 3 with content | Space-separated list of page indices to visualize. |
| `--data-dir` | `data/` | Root directory for extracted JSON. |
| `--output-dir` | `output/` | Root directory for visualization outputs. |
| `--dom` | `False` | Also generate interactive DOM graph HTML files. |

### Data Discovery

Without `--backend`, the script searches `<data-dir>/<doc_name>/` recursively for
`<doc_name>_content_list.json`. It first checks the canonical path
(`<data-dir>/<doc_name>/hybrid-auto-engine/<doc_name>_content_list.json`) before falling back
to a recursive search.

### Examples

```bash
# Visualize first 3 pages (default), auto-detect backend
uv run scripts/visualize.py --pdf data/pdfs/macdonals.pdf

# Specific pages + DOM HTML
uv run scripts/visualize.py \
  --pdf data/pdfs/macdonals.pdf \
  --pages 0 1 5 10 \
  --dom

# From a specific backend's output
uv run scripts/visualize.py \
  --pdf data/pdfs/macdonals.pdf \
  --backend pipeline \
  --pages 0 1 2
```

### Outputs

For each requested page: `output/<doc_name>/plots/<doc>_page<NNN>_layout_bboxes.png`
and `output/<doc_name>/plots/<doc>_page<NNN>_reading_order.png`.

Always: `output/<doc_name>/plots/<doc>_document_tree.png`.

With `--dom`: `output/<doc_name>/plots/<doc>_dom_full.html` and
`output/<doc_name>/plots/<doc>_page<NNN>_dom.html` for each requested page.

---

## `scripts/benchmark_throughput.py`

**Purpose:** Measure steady-state per-page extraction throughput with GPU monitoring and
scale-projection plots.

```
usage: benchmark_throughput.py [-h]
                               (--pdf PDF | --pdf-dir PDF_DIR)
                               [--backend BACKEND]
                               [--gpu-memory-utilization FLOAT]
                               [--repeats N]
                               [--data-dir DATA_DIR]
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--pdf` | — | Single PDF to benchmark (mutually exclusive with `--pdf-dir`). |
| `--pdf-dir` | — | Directory of PDFs (mutually exclusive with `--pdf`). All `*.pdf` files ≤ 30 MB are included. |
| `--backend` | `hybrid-auto-engine` | MinerU backend to benchmark. |
| `--gpu-memory-utilization` | `0.85` | vLLM KV-cache VRAM fraction. Recommended: `0.75` on 44 GB A100. |
| `--repeats` | `1` | Number of full passes over the document set (for statistical averaging). |
| `--data-dir` | `data/` | Root directory for MinerU JSON outputs. |

### Warm-Up Behavior

The first document in the list is used as the warm-up target (pages 0–2 only). Warm-up
timing is **not** recorded. After warm-up completes, the full benchmark begins.

### GPU Monitoring

`CUDA_VISIBLE_DEVICES` must be set to the GPU you want to benchmark. The monitor reads the
physical GPU index from this environment variable:

```bash
# Benchmark on GPU 6
CUDA_VISIBLE_DEVICES=6 uv run scripts/benchmark_throughput.py --pdf-dir data/pdfs/
```

### Examples

```bash
# Multi-document benchmark, GPU 6, 0.75 memory utilization
CUDA_VISIBLE_DEVICES=6 uv run scripts/benchmark_throughput.py \
  --pdf-dir data/pdfs/ \
  --backend hybrid-auto-engine \
  --gpu-memory-utilization 0.75 \
  --repeats 1

# Single PDF, 3 repeats for statistical significance
CUDA_VISIBLE_DEVICES=6 uv run scripts/benchmark_throughput.py \
  --pdf data/pdfs/macdonals.pdf \
  --backend hybrid-auto-engine \
  --repeats 3

# Pipeline backend (CPU-only, no CUDA needed)
uv run scripts/benchmark_throughput.py \
  --pdf-dir data/pdfs/ \
  --backend pipeline
```

### Outputs

Plots saved to `output/plots/`:
- `<tag>_latency.png` — per-document latency bar chart
- `<tag>_gpu_util.png` — GPU utilization timeline
- `<tag>_scale_projection.png` — log-log scale projection (1K–1B pages, 1/4/8/16 GPUs)

Console output: per-document timing and aggregate results (median/mean s/page, pages/min,
GPU utilization stats).

WandB run: logged to `mineru-xml-pipeline` with benchmark configuration and aggregate metrics.
