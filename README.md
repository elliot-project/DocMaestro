# OmniDocParser — MinerU2.5-Pro Document Extraction & XML Synthesis Pipeline

A high-performance document ingestion pipeline built on **MinerU2.5-Pro** (1.2B VLM). Given a PDF, it runs one of three inference backends to produce a structured XML file that embeds every content block—text, titles, figures, tables, equations, code—with precise bounding boxes and reading order. All runs are benchmarked and tracked in **Weights & Biases**.

---

## How It Works

The pipeline has four stages:

```
PDF
 │
 ▼
[1] MinerU Extraction  ──►  middle.json  (layout topology + page geometry)
                       ──►  content_list.json  (flat, reading-order content blocks)
 │
 ▼
[2] XML Synthesis  ──►  <document> tree with bbox attributes on every element
 │
 ▼
[3] Artifact Storage  ──►  data/<docname>/<backend>/  (raw JSONs)
                      ──►  output/<docname>_<backend>.xml  (synthesized XML)
 │
 ▼
[4] WandB Logging  ──►  engine latency, per-page timing, GPU VRAM, XML stats
```

### Stage 1 — MinerU Extraction

`src/extractor.py` calls the `mineru` CLI as a subprocess and collects its two primary output files:

- **`middle.json`** — full layout topology: per-page block hierarchy (`para_blocks`, `images`, `tables`, `interline_equations`), page dimensions, reading-order coordinates. Every block has `bbox: [x0, y0, x1, y1]` in page-pixel space.
- **`content_list.json`** — a flat list of all readable content blocks in human reading order. Each entry has `type`, `text` or visual data, `bbox` (normalized 0–1000), and `page_idx`.

Three backends are supported, selectable via `--backend`:

| Backend | Accuracy | Hardware | Notes |
|---|---|---|---|
| `pipeline` | 85+ | CPU or GPU (4 GB VRAM min) | Fastest, most portable |
| `vlm-auto-engine` | 95+ | GPU (8 GB VRAM min) | Local VLM inference via vLLM |
| `hybrid-auto-engine` | 95+ | GPU (8 GB VRAM min) | **Default.** VLM + native text extraction; lowest hallucination rate |

### Stage 2 — XML Synthesis

`src/xml_formatter.py` combines both JSON files into a hierarchical XML document:

```xml
<document name="macdonals" page_count="12" backend="hybrid-auto-engine">
  <page idx="0" width="595.0" height="842.0">
    <title level="1" bbox="[62, 120, 530, 150]">McDonald's Annual Report 2023</title>
    <text-block bbox="[62, 180, 530, 220]">Global comparable sales increased 8.7%...</text-block>
    <figure bbox="[62, 250, 530, 480]">
      <img src="images/chart_revenue.jpg"/>
      <caption>Fig. 1. Revenue breakdown by segment.</caption>
    </figure>
    <table bbox="[62, 500, 530, 700]">
      <caption>Table 1. Key financial metrics</caption>
      <table-body><html>...</html></table-body>
    </table>
    <math-block bbox="[100, 720, 400, 750]">$$\text{ROIC} = \frac{\text{NOPAT}}{\text{Capital}}$$</math-block>
  </page>
  ...
</document>
```

The tag mapping (`src/xml_schema.py`) is:

| `content_list.json` type | XML tag | Notes |
|---|---|---|
| `text` (level 0) | `<text-block>` | Body paragraph |
| `text` (level 1–N) | `<title level="N">` | Section heading |
| `image` | `<figure>` | Contains `<img>`, `<caption>`, `<footnote>` |
| `table` | `<table>` | Contains `<img>`, `<caption>`, `<table-body>`, `<footnote>` |
| `chart` | `<chart>` | Contains `<img>`, `<caption>` |
| `equation` | `<math-block>` | LaTeX in text content |
| `code` | `<code-block>` | With optional `sub-type="algorithm"` |
| `list` | `<list>` | |
| `header` / `footer` | `<page-header>` / `<page-footer>` | |
| `page_number` | `<page-number>` | |
| `aside_text` | `<aside>` | |
| `page_footnote` | `<page-footnote>` | |
| `seal` | `<seal>` | |
| anything else | `<block>` | Fallback for future types |

Bounding boxes are always formatted as `bbox="[x1, y1, x2, y2]"` integers. Elements with no bbox emit no attribute.

### Stage 3 — Artifact Storage

Raw MinerU outputs are preserved under `data/` for reproducibility and downstream use:

```
data/
  macdonals/
    hybrid-auto-engine/
      macdonals_middle.json
      macdonals_content_list.json
      images/
        ...
```

Synthesized XML is written to `output/`:

```
output/
  macdonals_hybrid-auto-engine.xml
  plots/          ← benchmark visualizations (future)
```

### Stage 4 — WandB Benchmarking

`src/benchmark.py` logs the following metrics to the `mineru-xml-pipeline` WandB project on every run:

| Metric | Description |
|---|---|
| `extraction/<backend>/total_duration_s` | Full subprocess wall-clock time |
| `extraction/<backend>/latency_per_page_s` | Duration ÷ page count |
| `extraction/<backend>/pages` | Page count |
| `extraction/<backend>/content_items` | Number of content blocks extracted |
| `extraction/<backend>/pdf_size_mb` | Source PDF file size |
| `gpu/<N>/vram_used_gb` | VRAM allocated per GPU (pre and post extraction) |
| `gpu/<N>/vram_utilization` | Fractional VRAM usage |
| `xml/size_kb` | Synthesized XML file size |

Each run also uploads the synthesized XML as a WandB Artifact (`type="document-xml"`), linked to the run for full lineage tracking.

---

## Project Structure

```
OmniDocParser/
├── src/
│   ├── xml_schema.py       # Content-type → XML tag mapping + bbox formatting
│   ├── xml_formatter.py    # document XML synthesis from content_list + middle JSON
│   ├── extractor.py        # MinerU CLI subprocess wrapper, returns ExtractionResult
│   └── benchmark.py        # WandB init, metric logging, GPU VRAM tracking
│
├── scripts/
│   └── run_extraction.py   # Main CLI: PDF → extract → XML → WandB
│
├── tests/
│   ├── test_xml_schema.py      # Tag resolution + bbox formatting (20 tests)
│   └── test_xml_formatter.py   # Full XML synthesis (27 tests)
│
├── data/                   # Raw MinerU JSON outputs (gitignored)
├── docs/                   # Source PDFs
├── output/
│   └── plots/              # Benchmark visualizations
└── pyproject.toml
```

---

## Setup

**Requirements:** Python 3.10–3.13, CUDA GPU with ≥8 GB VRAM (for `hybrid-auto-engine` or `vlm-auto-engine`). The `pipeline` backend runs on CPU.

```bash
# Install all dependencies (including mineru, vLLM, wandb)
uv add "mineru[all]"

# Log in to WandB
wandb login
```

MinerU downloads model weights on first run (~2–4 GB). They are cached in `~/.cache/huggingface/`.

---

## Usage

### Run a single PDF

```bash
uv run scripts/run_extraction.py --pdf docs/macdonals.pdf
```

### Choose a backend

```bash
# Hybrid (default, highest accuracy, GPU required)
uv run scripts/run_extraction.py --pdf docs/macdonals.pdf --backend hybrid-auto-engine

# Pipeline (CPU-friendly, still high quality)
uv run scripts/run_extraction.py --pdf docs/macdonals.pdf --backend pipeline

# VLM only
uv run scripts/run_extraction.py --pdf docs/macdonals.pdf --backend vlm-auto-engine
```

### Process all PDFs in a directory

```bash
uv run scripts/run_extraction.py --pdf docs/ --backend hybrid-auto-engine
```

### Parse a page range

```bash
# Pages 0–4 (first 5 pages)
uv run scripts/run_extraction.py --pdf docs/Microsoft_2024_annual_report.pdf --start-page 0 --end-page 4
```

### Full options

```
--pdf PATH          PDF file or directory of PDFs (required)
--backend           pipeline | vlm-auto-engine | hybrid-auto-engine (default: hybrid-auto-engine)
--lang              OCR language hint: en, ch, korean, japan, ... (default: en)
--start-page INT    Zero-based first page (default: 0)
--end-page INT      Zero-based last page inclusive (default: all)
--data-dir PATH     Raw JSON output root (default: data/)
--output-dir PATH   XML output root (default: output/)
```

---

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Lint + format
uv run ruff check --fix .
uv run ruff format .
```

Tests cover all XML tag mappings, bbox edge cases, figure/table/equation/code element builders, multi-page routing, and missing-bbox handling. No external services or GPU required to run tests.

---

## Tech Stack

| Component | Package |
|---|---|
| Document parsing | `mineru[all]` 3.1.x (OpenDataLab) |
| VLM inference | `vLLM` 0.11.x |
| Experiment tracking | `wandb` 0.26.x |
| Plotting | `matplotlib` 3.9.x |
| Linting / formatting | `ruff` |
| Testing | `pytest` |
| Package manager | `uv` |
