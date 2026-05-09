# Architecture

## System Overview

```
                         ┌─────────────────────────────────────────────────────┐
                         │                OmniDocParser Pipeline                │
                         └─────────────────────────────────────────────────────┘

  Input                         Core                              Outputs
  ──────                        ────                              ───────

  data/pdfs/                ┌──────────────┐               output/<doc>/
  *.pdf  ──────────────────▶│  extractor   │──────────────▶  <doc>_<backend>.xml
                            │  (src/)      │
                            └──────┬───────┘               output/<doc>/plots/
                                   │                          *_layout_bboxes.png
                                   │ middle.json              *_reading_order.png
                                   │ content_list.json        *_document_tree.png
                                   ▼                          *_dom_full.html
                            ┌──────────────┐                  *_page<N>_dom.html
                            │ xml_formatter│
                            │  (src/)      │               output/plots/
                            └──────┬───────┘                 *_latency.png
                                   │                          *_scale_projection.png
                                   │ XML string               *_gpu_util.png
                                   ▼
                            ┌──────────────┐
                            │  visualizer  │               WandB Project:
                            │  (src/)      │          ──▶  mineru-xml-pipeline
                            └──────┬───────┘               - extraction metrics
                                   │                        - GPU memory
                                   │ visualizations         - XML stats
                                   ▼
                            ┌──────────────┐
                            │  benchmark   │
                            │  (src/)      │
                            └──────────────┘
```

---

## Component Map

| Module | Location | Responsibility |
|--------|----------|----------------|
| `extractor` | `src/extractor.py` | Wraps MinerU's `do_parse()` API; handles all four backends; manages title-aided LLM patching; returns `ExtractionResult` |
| `xml_formatter` | `src/xml_formatter.py` | Converts `content_list` + `middle_json` → pretty-printed XML string |
| `xml_schema` | `src/xml_schema.py` | Maps MinerU content types to XML tags; formats bounding boxes |
| `visualizer` | `src/visualizer.py` | Renders layout-bbox PNGs, reading-order PNGs, document tree PNGs, and interactive D3 DOM HTML files |
| `benchmark` | `src/benchmark.py` | WandB helpers: `init_run`, `log_extraction_metrics`, `log_gpu_memory`, `log_xml_stats` |
| `run_extraction` | `scripts/run_extraction.py` | CLI orchestrator: PDF → extraction → XML → WandB logging |
| `visualize` | `scripts/visualize.py` | CLI for standalone visualization of already-extracted documents |
| `benchmark_throughput` | `scripts/benchmark_throughput.py` | Throughput benchmark with GPU monitoring and scale-projection plots |

---

## Data Flow

### Stage 1 — PDF → Raw JSON (extraction)

```
pdf_path  →  do_parse()  →  <doc_name>_middle.json
                         →  <doc_name>_content_list.json
```

`do_parse()` is MinerU's core entry point. It:
1. Runs layout detection (bounding box classification per page block).
2. Runs the selected VLM backend for OCR and content understanding.
3. Optionally calls an LLM endpoint to assign title hierarchy levels (H1–H4).
4. Writes two JSON files under the configured output directory.

The pipeline copies both files to `data/<doc_name>/<backend>/` for reproducibility.

### Stage 2 — Raw JSON → XML (formatting)

```
content_list.json  ─┐
                    ├──▶  build_document_xml()  ──▶  XML string  ──▶  output/<doc>/<doc>_<backend>.xml
middle.json        ─┘
```

`build_document_xml()` groups items by page, resolves XML tags via `xml_schema`, attaches
bbox attributes, and serializes to pretty-printed XML.

### Stage 3 — XML + JSON → Visualizations (visualizer)

```
content_list.json  ─┐
                    ├──▶  generate_all_visualizations()  ──▶  output/<doc>/plots/
pdf_path           ─┘
```

Three independent render functions are called per document:
- `draw_layout_viz()` — per page
- `draw_reading_order_viz()` — per page
- `draw_doc_tree_viz()` — whole document (static PNG)
- `draw_dom_graph_viz()` — whole document + per page (interactive HTML, optional)

### Stage 4 — Metrics → WandB (benchmark)

```
ExtractionResult  ──▶  log_extraction_metrics()  ──▶  WandB run
torch.cuda        ──▶  log_gpu_memory()          ──▶  WandB run
xml_str           ──▶  log_xml_stats()           ──▶  WandB run
```

---

## Key Data Structures

### ExtractionResult

Returned by `extract()`. All downstream code works from this dataclass.

```python
@dataclass
class ExtractionResult:
    content_list: list[dict]   # flat reading-order list from content_list.json
    middle_json: dict          # topology dict from middle.json
    backend: str               # e.g. 'hybrid-auto-engine'
    pdf_path: Path             # absolute path to source PDF
    duration_seconds: float    # wall-clock time for extraction
    page_count: int            # number of pages (from middle.json pdf_info)
    output_dir: Path           # directory where MinerU wrote raw outputs
```

### content_list.json item schema

Each item in `content_list` is a dict with these keys:

| Key | Type | Description |
|-----|------|-------------|
| `type` | str | Content type: `text`, `image`, `table`, `chart`, `equation`, `code`, `list`, `header`, `footer`, `page_number`, `aside_text`, `page_footnote`, `seal`, `index` |
| `text` | str | Extracted text (for text-bearing types) |
| `text_level` | int | `0` = body text; `1–4` = heading level (only for `type=text`) |
| `page_idx` | int | Zero-based page index |
| `bbox` | list[float] | `[x0, y0, x1, y1]` in 0–1000 normalized coordinates |
| `img_path` | str | Relative path to extracted image (for `image`, `table`, `chart`, `equation`) |
| `image_caption` | list[str] | Caption strings (for `image`) |
| `image_footnote` | list[str] | Footnote strings (for `image`) |
| `table_caption` | list[str] | Caption strings (for `table`) |
| `table_footnote` | list[str] | Footnote strings (for `table`) |
| `table_body` | str | Raw HTML table string from MinerU (for `table`) |
| `chart_caption` | list[str] | Caption strings (for `chart`) |
| `sub_type` | str | Sub-classification for `code` blocks (e.g. `algorithm`) |

### middle.json structure

The key top-level fields used by this pipeline:

| Key | Type | Description |
|-----|------|-------------|
| `pdf_info` | list[dict] | One entry per page; each has `page_idx`, `page_size: [w, h]`, and layout block lists |
| `_backend` | str | Backend identifier written by MinerU |
| `_version_name` | str | MinerU version string |

---

## Coordinate System

MinerU normalizes all bounding boxes in `content_list.json` to a **0–1000 scale** on both
axes, where `(0, 0)` is the top-left corner. This normalization is independent of the actual
page dimensions in pixels or points.

The pipeline preserves this normalization in the XML output. To convert to absolute pixels for
a rendered image at scale `s`:

```
x_px = (x_normalized / 1000) * image_width_at_scale_s
y_px = (y_normalized / 1000) * image_height_at_scale_s
```

The `visualizer` uses this conversion internally when drawing bounding boxes on rendered PDF
pages (see `_bbox_to_pixels()` in `src/visualizer.py:94`).

Page dimensions in `middle.json` are in PDF points (72 pts/inch). The visualizer renders at
`scale=2.0` (≈144 DPI) using `pypdfium2`.

---

## Dependency Graph

```
scripts/run_extraction.py
  └── src/extractor.py           ← mineru, vllm (heavy deps, lazy imported)
  └── src/xml_formatter.py
      └── src/xml_schema.py
  └── src/benchmark.py           ← wandb

scripts/visualize.py
  └── src/visualizer.py          ← matplotlib, networkx, pypdfium2, d3 (CDN)

scripts/benchmark_throughput.py
  └── src/extractor.py
  └── src/benchmark.py
  └── matplotlib
```

MinerU and vLLM are **lazily imported** inside `extractor.py`. This means importing
`src.extractor` at the top of a script does not trigger CUDA initialization — only calling
`extract()` does. This is critical: CUDA must not be initialized before vLLM spawns its engine
subprocess (doing so forces vLLM into slower spawn mode and can cause OOM failures).
