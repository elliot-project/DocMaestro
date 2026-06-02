# DocMaestro

A high-performance document ingestion pipeline built on **MinerU2.5-Pro** (1.2B VLM). Converts PDFs into structured XML with bounding boxes and reading order, supports four extraction backends, and tracks every run — latency, GPU VRAM, XML stats — in **Weights & Biases**.

**Requirements:** Python 3.10–3.13 · CUDA 11.8+ · 24 GB VRAM minimum

---

## Quick Start

```bash
uv sync
wandb login
uv run main.py --pdf path/to/doc.pdf
```

---

## Backends

| Backend | Description |
|---|---|
| `pipeline` | Rule-based, no VLM |
| `vlm-auto-engine` | In-process vLLM |
| `hybrid-auto-engine` | In-process vLLM + native text extraction |
| `hybrid-http-client` | External VLM server via HTTP (default) |

---

## Scripts

- [`main.py`](main.py) — Primary pipeline orchestrator, single doc or batch, YAML-config driven
- [`scripts/run_docvqa2026.py`](scripts/run_docvqa2026.py) — DocVQA-2026 HuggingFace dataset sweep
- [`scripts/run_sharded.py`](scripts/run_sharded.py) — Multi-GPU sharded launcher
- [`scripts/benchmark_throughput.py`](scripts/benchmark_throughput.py) — GPU throughput benchmark
- [`scripts/classify_figures.py`](scripts/classify_figures.py) — Enrich XML `<figure>` elements with SigLIP2 categories
- [`scripts/visualize.py`](scripts/visualize.py) — Standalone visualizations from existing JSON

---

## Documentation

- [Overview & capabilities](docs/overview.md)
- [Architecture & data flow](docs/architecture.md)
- [Extraction backends & API](docs/extraction.md)
- [XML schema reference](docs/xml-schema.md)
- [Visualizations](docs/visualizations.md)
- [Benchmarking & WandB metrics](docs/benchmarking.md)
- [CLI reference](docs/cli-reference.md)
- [Repository structure](docs/repository-structure.md)
- [Task Composer — SSL task generation](docs/task-composer.md)

---

## Development

```bash
# Run tests
uv run pytest tests/

# Lint + format
uv run ruff check --fix . && uv run ruff format .
```

YAML configs live in `configs/extractor/` (`default.yaml`, `fast_img_cls.yaml`).
