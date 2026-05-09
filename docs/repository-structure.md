# Repository Structure

## Directory Layout

```
OmniDocParser/
│
├── CLAUDE.md                       ← Project instructions for AI assistant
├── pyproject.toml                  ← Dependencies, Python version, Ruff config
├── uv.lock                         ← Locked dependency versions
│
├── src/                            ← Core library modules
│   ├── __init__.py
│   ├── extractor.py                ← MinerU extraction wrapper
│   ├── xml_formatter.py            ← content_list + middle_json → XML
│   ├── xml_schema.py               ← Content-type → XML tag mapping
│   ├── visualizer.py               ← All visualization renderers
│   └── benchmark.py                ← WandB helpers
│
├── scripts/                        ← Executable CLI scripts
│   ├── run_extraction.py           ← Full pipeline: extract → XML → WandB
│   ├── visualize.py                ← Standalone visualization
│   └── benchmark_throughput.py    ← Throughput benchmark with GPU monitoring
│
├── tests/                          ← pytest unit tests
│   ├── __init__.py
│   ├── test_xml_schema.py          ← Tag resolution + bbox formatting tests
│   ├── test_xml_formatter.py       ← XML synthesis tests (all element types)
│   └── test_visualizer.py          ← Tree building + DOM graph tests
│
├── docs/                           ← Documentation (this folder)
│   ├── README.md                   ← Index and quick start
│   ├── overview.md                 ← Project purpose, capabilities, requirements
│   ├── architecture.md             ← System design, data flow, data structures
│   ├── extraction.md               ← Backends, extract() API, title-aided, CUDA
│   ├── xml-schema.md               ← XML format specification
│   ├── visualizations.md           ← All visualization types
│   ├── benchmarking.md             ← WandB + throughput benchmark
│   ├── cli-reference.md            ← All CLI flags and examples
│   ├── repository-structure.md     ← This file
│   └── superpowers/                ← AI assistant skill files (not user docs)
│
├── data/                           ← All data inputs and raw outputs
│   ├── pdfs/                       ← Source PDF documents
│   │   ├── macdonals.pdf
│   │   ├── ASML.pdf
│   │   └── ...
│   ├── macdonals/                  ← Raw MinerU outputs for macdonals
│   │   └── hybrid-auto-engine/
│   │       ├── macdonals_middle.json
│   │       └── macdonals_content_list.json
│   ├── ASML/
│   │   └── hybrid-auto-engine/
│   │       └── ...
│   └── ...
│
├── output/                         ← Pipeline outputs, organized per document
│   ├── macdonals/
│   │   ├── macdonals_hybrid-auto-engine.xml
│   │   ├── macdonals_hybrid-http-client.xml
│   │   └── plots/
│   │       ├── macdonals_document_tree.png
│   │       ├── macdonals_dom_full.html
│   │       ├── macdonals_page000_layout_bboxes.png
│   │       ├── macdonals_page000_reading_order.png
│   │       ├── macdonals_page000_dom.html
│   │       └── ...
│   ├── MinerU2_5Pro/
│   │   ├── MinerU2_5Pro_hybrid-auto-engine.xml
│   │   └── plots/
│   │       └── ...
│   └── plots/                      ← Global benchmark plots
│       ├── *_latency.png
│       ├── *_gpu_util.png
│       └── *_scale_projection.png
│
└── lib/                            ← Vendored JS libraries (legacy, from PyVis era)
    ├── bindings/
    └── tom-select/
```

---

## Module Reference

### `src/extractor.py`

| Symbol | Type | Purpose |
|--------|------|---------|
| `SUPPORTED_BACKENDS` | `frozenset[str]` | The four valid backend identifiers |
| `ExtractionResult` | dataclass | Parsed outputs from a single extraction run |
| `extract()` | function | Main entry point: PDF → `ExtractionResult` |
| `_find_json_file()` | function | Recursively find newest JSON matching a suffix |
| `_title_aided_patch()` | context manager | Temporarily patches MinerU for title hierarchy |
| `_make_patched_request_title_levels()` | function | Robust JSON parser for LLM dict output |
| `_validate_title_aided_config()` | function | Input validation for title_aided_config |

### `src/xml_formatter.py`

| Symbol | Type | Purpose |
|--------|------|---------|
| `build_document_xml()` | function | Main entry point: content_list + middle_json → XML string |
| `save_document_xml()` | function | Write XML string to disk |
| `_build_figure_element()` | function | `type=image` → `<figure>` |
| `_build_chart_element()` | function | `type=chart` → `<chart>` |
| `_build_table_element()` | function | `type=table` → `<table>` |
| `_build_equation_element()` | function | `type=equation` → `<math-block>` |
| `_build_code_element()` | function | `type=code` → `<code-block>` |
| `_build_generic_text_element()` | function | All other text-bearing types |
| `_clean()` | function | Strip XML-illegal control characters from text |

### `src/xml_schema.py`

| Symbol | Type | Purpose |
|--------|------|---------|
| `CONTENT_TYPE_TO_TAG` | `dict[str, str]` | MinerU type → XML tag name mapping |
| `XMLTagConfig` | frozen dataclass | Resolved tag + extra_attrs for one element |
| `resolve_tag()` | function | Resolve XML tag for any (type, text_level) pair |
| `format_bbox()` | function | Format bbox list as `"[x1, y1, x2, y2]"` string |

### `src/visualizer.py`

| Symbol | Type | Purpose |
|--------|------|---------|
| `CONTENT_TYPE_COLORS` | `dict[str, str]` | Hex color per content type |
| `CONTENT_TYPE_ICONS` | `dict[str, str]` | Short string icon per content type |
| `render_pdf_page()` | function | Render one PDF page to PIL Image via pypdfium2 |
| `draw_layout_viz()` | function | Layout bboxes overlay PNG |
| `draw_reading_order_viz()` | function | Numbered reading-order PNG |
| `build_doc_tree()` | function | Flat content_list → nested title/leaf tree |
| `draw_doc_tree_viz()` | function | Static indented tree PNG |
| `draw_dom_graph_viz()` | function | Interactive D3 DOM HTML |
| `generate_all_visualizations()` | function | Convenience: runs all four renderers |
| `_build_networkx_graph()` | function | Tree → NetworkX DiGraph with node attrs |
| `_nx_to_d3_json()` | function | NetworkX DiGraph → nested D3 hierarchy dict |
| `_write_d3_tree_html()` | function | Inject D3 data into HTML template |

### `src/benchmark.py`

| Symbol | Type | Purpose |
|--------|------|---------|
| `init_run()` | function | Initialize WandB run for one extraction job |
| `log_extraction_metrics()` | function | Log timing + document stats to WandB |
| `log_gpu_memory()` | function | Log VRAM usage for all CUDA devices |
| `log_xml_stats()` | function | Log XML size and output path |

---

## Tests

Tests mirror `src/` structure and live in `tests/`. Run with:

```bash
uv run pytest tests/
uv run pytest tests/ -v        # verbose output
uv run pytest tests/ -k "bbox" # filter by name
```

### Test Files

| File | Coverage |
|------|---------|
| `tests/test_xml_schema.py` | `resolve_tag()` for all 14 content types; `format_bbox()` edge cases (None, empty, wrong length, float truncation) |
| `tests/test_xml_formatter.py` | `build_document_xml()` for all element types; multi-page routing; bbox attribute format; XML-illegal character handling edge cases |
| `tests/test_visualizer.py` | `build_doc_tree()` tree structure; skipped heading levels; `_count_leaves()`; `_flatten_tree_for_render()` depth and truncation; `_build_networkx_graph()` node structure, cluster/typegroup hierarchy; `draw_dom_graph_viz()` HTML output |

All tests use pure Python fixtures with no PDF files — unit tests only, no integration tests
that call MinerU.

---

## Coding Standards

### Python version
Python 3.10–3.13. The codebase uses `|` union types (`Path | None`) which require ≥ 3.10.

### Package management
Always use `uv`:
```bash
uv add <package>          # add dependency
uv remove <package>       # remove dependency
uv sync                   # install from lock file
```

Never use `pip` directly.

### Linting and formatting
```bash
uv run ruff check --fix .     # lint + auto-fix
uv run ruff format .          # format
```

Line length: 100. Target: Python 3.10. Rules: E, F, W, I (ignores E501 for long strings).

### Docstrings
Google-style docstrings on all public functions. Args and Returns sections required.

### Comments
Only for non-obvious WHY, not what. No multi-line comment blocks.

### CUDA seeds
Set CPU seeds at script top (`random.seed`, `np.random.seed`, `torch.manual_seed`).
Never call `torch.cuda.manual_seed_all()` at module level — it initializes CUDA and
breaks vLLM's subprocess spawning.

### Headless rendering
Always `matplotlib.use("Agg")` before any matplotlib import. Never call `plt.show()`.

### Paths
Always use `Path` objects. Use `.mkdir(parents=True, exist_ok=True)` before writing.
Use absolute paths (`Path(__file__).resolve().parents[N]`) for script-relative imports.

---

## Adding a New Document

```bash
# 1. Place the PDF
cp my_doc.pdf data/pdfs/

# 2. Run extraction
CUDA_VISIBLE_DEVICES=6 uv run scripts/run_extraction.py \
  --pdf data/pdfs/my_doc.pdf \
  --backend hybrid-auto-engine

# 3. Visualize
uv run scripts/visualize.py \
  --pdf data/pdfs/my_doc.pdf \
  --dom
```

This creates:
```
data/my_doc/hybrid-auto-engine/
  my_doc_middle.json
  my_doc_content_list.json

output/my_doc/
  my_doc_hybrid-auto-engine.xml
  plots/
    my_doc_document_tree.png
    my_doc_dom_full.html
    my_doc_page000_layout_bboxes.png
    my_doc_page000_reading_order.png
    ...
```
