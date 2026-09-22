# Extraction

## Overview

`src/doc_maestro/extractor.py` is the single interface between DocMaestro and MinerU. It exposes two
public functions — `extract()` and `batch_extract()` — that wrap MinerU's `do_parse()` Python
API, read the resulting JSON files from disk, and return structured `ExtractionResult` objects.

---

## MinerU Backends

Four backends are supported. The right choice depends on your infrastructure:

| Backend | How it works | When to use |
|---------|-------------|-------------|
| `hybrid-http-client` | **Default.** Connects to a **running VLM server** (e.g., started with `mineru-serve`). Sends page images over HTTP to `--server-url`. | Production. Keeps model warm between runs. Lowest per-call overhead. Requires `--server-url`. |
| `hybrid-auto-engine` | Loads the **VLM in-process** via vLLM on first call. Model stays in VRAM for the process lifetime. | Single-machine research. No server setup needed. |
| `vlm-auto-engine` | Pure VLM mode; loads VLM in-process. | Experimentation. Lower accuracy than hybrid on complex layouts. |
| `pipeline` | Rule-based, **no VLM**. Uses classical CV for layout detection and Tesseract-style OCR. | CPU-only fallback. Fast but lower quality. |

All four backends produce the same output files (`_middle.json`, `_content_list.json`) with the
same schema. XML synthesis is backend-agnostic.

---

## The `extract()` Function

```python
from doc_maestro.extractor import extract
from pathlib import Path

result = extract(
    pdf_path=Path("data/pdfs/macdonals.pdf"),
    output_dir=Path("data/macdonals/hybrid-auto-engine"),
    backend="hybrid-auto-engine",
    language="en",
    start_page=0,
    end_page=None,          # None = all pages
    server_url=None,        # required for hybrid-http-client
    title_aided_config=None,
    gpu_memory_utilization=0.75,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdf_path` | `Path` | required | Path to the source PDF file |
| `output_dir` | `Path` | required | Directory for MinerU's raw JSON output |
| `backend` | `str` | `"hybrid-http-client"` | One of the four backend identifiers |
| `language` | `str` | `"en"` | OCR language hint (`"en"`, `"ch"`, etc.) |
| `start_page` | `int` | `0` | Zero-based first page to parse |
| `end_page` | `int \| None` | `None` | Zero-based last page (inclusive); `None` = all |
| `server_url` | `str \| None` | `None` | VLM server base URL, e.g. `http://localhost:8000` |
| `title_aided_config` | `dict \| None` | `None` | LLM config for title hierarchy; see below |
| `gpu_memory_utilization` | `float \| None` | `None` | vLLM KV-cache VRAM fraction; `None` = MinerU default (0.5) |

### Return Value

Returns an `ExtractionResult` dataclass with fields: `content_list`, `middle_json`, `backend`,
`pdf_path`, `duration_seconds`, `page_count`, `output_dir`.

### Errors

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `backend` not in `SUPPORTED_BACKENDS`; malformed `title_aided_config` |
| `FileNotFoundError` | `pdf_path` does not exist |
| `RuntimeError` | MinerU produced no output files (extraction failed silently) |

---

## The `batch_extract()` Function

`batch_extract()` accepts a list of PDF paths and issues a **single `do_parse` call** for
pipeline-level batching. This is more efficient than calling `extract()` in a loop when using
the `pipeline` backend, because MinerU can schedule pages from multiple documents in one pass.

```python
from doc_maestro.extractor import batch_extract
from pathlib import Path

results = batch_extract(
    pdf_paths=[Path("data/pdfs/a.pdf"), Path("data/pdfs/b.pdf")],
    output_dir=Path("data/batch_out"),
    backend="pipeline",
    batch_size=4,
    window_size=64,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdf_paths` | `list[Path]` | required | List of PDF paths to process in a single batch |
| `output_dir` | `Path` | required | Directory for MinerU's raw JSON output |
| `backend` | `str` | `"hybrid-http-client"` | One of the four backend identifiers |
| `batch_size` | `int` | `1` | Documents per `do_parse` call. Set `>1` with the `pipeline` backend for GPU batching |
| `window_size` | `int` | `64` | MinerU page processing window. Sets the `MINERU_PROCESSING_WINDOW_SIZE` environment variable |
| `**kwargs` | — | — | Forwarded to `extract()` for each document (e.g. `language`, `server_url`) |

### Return Value

Returns a `list[ExtractionResult]`, one per input PDF, in the same order as `pdf_paths`.

### Environment Variable

`MINERU_PROCESSING_WINDOW_SIZE` (default `64`) controls how many pages MinerU processes in
each internal window. Lower values reduce peak memory; higher values improve GPU utilization
on long documents. Controlled via `--window-size` in `main.py`.

---

## Output Files

After a successful `extract()` call, MinerU writes files under `output_dir`. The pipeline
reads the **most recently modified** match for each suffix (to handle MinerU's nested
subdirectory structure where `do_parse` may also leave older copies from prior runs):

| Suffix | Contents |
|--------|---------|
| `_middle.json` | Full topology: page geometry, layout blocks, raw VLM output |
| `_content_list.json` | Flat reading-order semantic list used for XML synthesis |

The pipeline then copies both to `data/<doc_name>/<backend>/` at deterministic paths for
reproducibility:

```
data/<doc_name>/<backend>/<doc_name>_middle.json
data/<doc_name>/<backend>/<doc_name>_content_list.json
```

---

## Page Slicing

Use `start_page` / `end_page` to extract a subset of pages. This is useful for:
- Quick testing (pages 0–2 for warm-up)
- Large documents where only a section is needed
- Debugging a specific page with a known layout issue

Both indices are **zero-based and inclusive**. A 100-page document's last page is index `99`.

---

## Title-Aided Hierarchy (H1–H4)

By default, MinerU assigns `text_level=0` to all body text and `text_level=1` to detected
titles, without distinguishing heading levels. The `title_aided` feature calls an LLM to
classify titles into H1–H4, enabling richer document-tree structure.

### Activation

Pass a `title_aided_config` dict to `extract()`:

```python
result = extract(
    pdf_path=...,
    backend="hybrid-auto-engine",   # only works on hybrid backends
    title_aided_config={
        "api_key": "dummy",          # use "dummy" for local servers
        "base_url": "http://localhost:8000/v1",
        "model": "Qwen/Qwen3-VL-8B-Instruct",
    },
)
```

Required keys: `api_key`, `base_url`, `model` (all strings).
Optional key: `enable_thinking` (bool) — set `True` for Qwen3-Thinking models.

**Note:** Title-aided is only active for `hybrid-auto-engine` and `hybrid-http-client`
backends. It has no effect on `pipeline` or `vlm-auto-engine`.

### How it works (internal)

`extract()` uses a context manager (`_title_aided_patch`) that temporarily monkey-patches two
MinerU modules before calling `do_parse()` and restores them afterward:

1. **`mineru.backend.hybrid.hybrid_model_output_to_middle_json`** — sets `title_aided_enable=True`
   and injects the LLM config dict.
2. **`mineru.utils.llm_aided._request_title_levels`** — wraps the function with a robust JSON
   parser that handles three response formats:
   - Standard JSON dict `{"0": 1, "1": 2, ...}` (via `json_repair`)
   - Python syntax with unquoted integer keys `{0:1,1:2,...}` (via `ast.literal_eval`)
   - Prose-wrapped dicts where the model includes reasoning text before the dict (regex
     extraction of the last `{...}` block)

This three-attempt fallback is necessary because different LLM models and configurations
produce different output formats for the title-level mapping dict.

### Model recommendation

Use `Qwen/Qwen3-VL-8B-Instruct` (not Thinking variant). The Thinking model exhausts its token
budget generating chain-of-thought reasoning instead of the required dict, causing parsing
failures.

---

## GPU Memory Configuration

The `gpu_memory_utilization` parameter controls how much of total GPU VRAM vLLM reserves for
its KV-cache. The remainder is available for other processes.

```python
# Safe setting leaving ~9 GB headroom on a 44 GB A100
result = extract(..., gpu_memory_utilization=0.75)
```

**Headroom math for 44 GB A100:**
- At `gpu_memory_utilization=0.75`: vLLM uses ~33 GB → 11 GB free
- MinerU's layout/OCR models need ~3.6 GB alongside vLLM
- Net free: ~7.4 GB (safe)

At `0.85` (37.3 GB for vLLM + 3.6 GB = 40.9 GB): reported OOM in testing on 44 GB A100.

**Default behavior:** If `gpu_memory_utilization=None`, MinerU passes its default (`0.5`) to
vLLM. This leaves half of VRAM free but constrains batch size and throughput.

---

## CUDA Initialization Order

**Critical constraint:** CUDA must not be initialized in the main Python process before vLLM
spawns its `EngineCore` subprocess. If it is, vLLM falls back to `spawn` multiprocessing mode
instead of `fork`, which can trigger OOM or initialization failures.

The pipeline enforces this by:
1. Never calling `torch.cuda.manual_seed_all()` at module import time.
2. Importing `mineru` and `vllm` lazily inside `extract()`, not at module level.
3. Setting only CPU seeds at the top of scripts (`random.seed`, `np.random.seed`,
   `torch.manual_seed`).

If you encounter `RuntimeError: Engine core initialization failed`, check whether any code
in the call stack initializes CUDA before `extract()` is called.

---

## Zombie EngineCore Processes

A failed vLLM run may leave an orphan `VLLM::EngineCore` subprocess holding GPU memory. Before
retrying, check for and kill these:

```bash
# Find orphan vLLM processes
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# Kill by PID
kill -9 <pid>
```

These processes do not exit automatically when the parent Python process crashes.
