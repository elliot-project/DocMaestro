# Overview

## What Is OmniDocParser?

OmniDocParser is a research-grade document intelligence pipeline built on top of
**MinerU2.5-Pro** (a 1.2B-parameter Vision-Language Model). It ingests PDF documents and
produces three primary artifacts:

1. **Structured XML** — a hierarchical representation of the document's content (text, tables,
   figures, equations, code, headings) with bounding-box coordinates on every element.
2. **Visualizations** — layout overlays, reading-order maps, document-tree diagrams, and
   interactive DOM HTML files for inspecting the parsed structure visually.
3. **Benchmark metrics** — wall-clock timing, per-page latency, GPU utilization, and WandB
   experiment logs for comparing backends and configurations.

The pipeline is designed to run on GPU servers (CUDA required for VLM inference) and outputs
everything to disk for downstream consumption by LLMs, RAG systems, or evaluation harnesses.

---

## Core Capabilities

| Capability | Details |
|------------|---------|
| **Multi-backend extraction** | Pipeline (rule-based), VLM-Auto-Engine (in-process vLLM), Hybrid-Auto-Engine (in-process, best accuracy), Hybrid-HTTP-Client (default; connects to a running VLM server via HTTP `--server-url`) |
| **Rich content types** | Text paragraphs, titles (H1–H4), figures, tables (with HTML table body), charts, equations (LaTeX), code blocks, headers/footers, footnotes, page numbers, aside text |
| **Bounding boxes everywhere** | Every content element in the XML carries a `bbox="[x1, y1, x2, y2]"` attribute in 0–1000 normalized page coordinates |
| **Title hierarchy (H1–H4)** | Optional LLM-aided title-level labeling via any OpenAI-compatible endpoint (e.g., Qwen3-VL-8B-Instruct) |
| **GPU-optimized inference** | Configurable `gpu_memory_utilization` for vLLM KV-cache allocation; warm-up pass separates model init from steady-state timing |
| **WandB experiment tracking** | Every extraction run logs duration, per-page latency, GPU memory, XML size, and PDF metadata to the `mineru-xml-pipeline` WandB project |
| **Interactive DOM viewer** | D3 v7 collapsible tree rendered as self-contained HTML; pan/zoom/expand; full-doc collapsed view and per-page expanded view |

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.11 |
| CUDA | 11.8 | 12.1+ |
| GPU VRAM | 24 GB | 44 GB (A100/H100) |
| RAM | 32 GB | 64 GB |
| Disk | 20 GB (model weights) | 50 GB |

**Package manager:** [`uv`](https://github.com/astral-sh/uv) — all commands use `uv run`.
Do not use `pip` directly.

**GPU isolation:** Use `CUDA_VISIBLE_DEVICES=<N>` to pin execution to a specific GPU. On
multi-GPU nodes, always set this to avoid vLLM contending with other jobs:

```bash
CUDA_VISIBLE_DEVICES=6 uv run main.py --pdf data/pdfs/macdonals.pdf
```

---

## Installation

```bash
git clone <repo>
cd OmniDocParser
uv sync          # installs all dependencies from pyproject.toml + uv.lock
```

MinerU downloads model weights on first run (~8 GB for the 1.2B VLM). Weights are cached in
`~/.cache/huggingface/`.

---

## Key Design Decisions

### Why XML instead of Markdown?
MinerU's default output format is Markdown. Markdown discards structural metadata (bounding
boxes, element types, page indices). The custom XML format preserves all of this, making the
output suitable for spatial reasoning, layout-aware RAG, and grounded document QA.

### Why separate content_list.json from middle.json?
MinerU writes two JSON intermediates. `content_list.json` is the flat, reading-order semantic
list (what we parse into XML). `middle.json` contains the full topological graph with page
geometry (page dimensions, raw block coordinates). The pipeline reads both and cross-references
them so every XML element carries accurate geometry.

### Why not wrap MinerU in a subprocess?
The pipeline calls MinerU's Python API (`do_parse`) directly in-process. This avoids the
overhead of a subprocess, enables passing Python objects (byte arrays, config dicts) directly,
and makes it possible to monkey-patch MinerU's LLM parsing code for the title-aided feature
without modifying MinerU's source.

### Why a warm-up pass in the benchmark?
vLLM loads model weights and compiles CUDA kernels on the first forward pass. Including this
initialization time in throughput numbers would be misleading for production estimates, where
a server runs continuously. The benchmark always runs a 3-page warm-up before recording timing.
