# DocMaestro

A data generation engine that turns raw documents into self-supervised datasets for document understanding via rich extraction and modular task composition. Built on top of MinerU2.5-Pro, DocMaestro provides an end-to-end modular framework that converts documents into structured annotations and exports training-ready WebDataset shards for distributed training frameworks such as Megatron.

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
- [`scripts/run_doctasks.py`](scripts/run_doctasks.py) — Run every registered SSL task composer over `output/`
- [`viz/task_visualizer/server.py`](viz/task_visualizer/server.py) — Interactive SSL task inspector (`uv run viz/task_visualizer/server.py`, then open http://localhost:8767)

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


---

## Acknowledgements

This project builds directly on [MinerU](https://github.com/opendatalab/mineru). We thank the OpenDataLab team for developing and open-sourcing their document extraction toolkit and models.
