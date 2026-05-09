# OmniDocParser Documentation

**MinerU2.5-Pro document extraction, XML synthesis, and benchmarking pipeline.**

---

## Contents

| File | Description |
|------|-------------|
| [overview.md](overview.md) | Project purpose, capabilities, system requirements, quick-start |
| [architecture.md](architecture.md) | System design, component map, data flow, key data structures |
| [extraction.md](extraction.md) | MinerU backends, `extract()` API, title-aided LLM labeling, CUDA notes |
| [xml-schema.md](xml-schema.md) | Custom XML format specification, tag reference, bbox conventions |
| [visualizations.md](visualizations.md) | Layout, reading-order, document tree PNG, interactive D3 DOM HTML |
| [benchmarking.md](benchmarking.md) | WandB integration, logged metrics, throughput benchmark, GPU monitoring |
| [cli-reference.md](cli-reference.md) | All CLI scripts — flags, defaults, usage examples |
| [repository-structure.md](repository-structure.md) | Directory layout, file conventions, coding standards |

---

## 30-Second Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Run extraction on a PDF (requires a running VLM server)
uv run scripts/run_extraction.py \
  --pdf data/pdfs/macdonals.pdf \
  --backend hybrid-http-client \
  --server-url http://localhost:8000

# 3. Visualize the extracted document
uv run scripts/visualize.py \
  --pdf data/pdfs/macdonals.pdf \
  --dom

# 4. Benchmark throughput
CUDA_VISIBLE_DEVICES=6 uv run scripts/benchmark_throughput.py \
  --pdf-dir data/pdfs/ \
  --backend hybrid-auto-engine \
  --gpu-memory-utilization 0.75
```

Outputs land in `output/<doc_name>/` (XML + per-doc plots) and `output/plots/` (global benchmark charts).
