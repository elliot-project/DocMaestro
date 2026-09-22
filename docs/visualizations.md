# Visualizations

## Overview

`src/doc_maestro/visualizer.py` produces four types of visual output from a document's `content_list.json`
and its source PDF. All outputs are saved under `output/<doc_name>/plots/` (per-document) or
`output/plots/` (global benchmark plots).

The visualizer is **headless** — it uses `matplotlib.use("Agg")` and never calls `plt.show()`.
All outputs are PNG or HTML files written to disk.

---

## Output Types

| Function | Output | Per-page / Full-doc |
|----------|--------|---------------------|
| `draw_layout_viz()` | `*_page<NNN>_layout_bboxes.png` | Per page |
| `draw_reading_order_viz()` | `*_page<NNN>_reading_order.png` | Per page |
| `draw_doc_tree_viz()` | `*_document_tree.png` | Full document |
| `draw_dom_graph_viz()` | `*_dom_full.html`, `*_page<NNN>_dom.html` | Both modes |

---

## 1. Layout & Content Type Visualization

**File:** `output/<doc_name>/plots/<doc>_page<NNN>_layout_bboxes.png`

Renders the PDF page at 144 DPI (scale=2.0) and overlays each detected content block as a
semi-transparent colored rectangle (alpha=0.18). A small type label appears at the top-left
corner of each box, and a legend is added to the upper-right corner.

### Color scheme

| Content Type | Color |
|-------------|-------|
| `text` (body) | `#4CAF50` (green) |
| `title` (h1–h4) | `#2196F3` (blue) |
| `image` | `#FF9800` (orange) |
| `table` | `#F44336` (red) |
| `chart` | `#E91E63` (pink) |
| `equation` | `#9C27B0` (purple) |
| `code` | `#00BCD4` (cyan) |
| `list` | `#8BC34A` (light green) |
| `header` / `footer` | `#607D8B` (blue-grey) |
| `page_number` | `#9E9E9E` (grey) |
| `aside_text` | `#FF5722` (deep orange) |
| `page_footnote` | `#795548` (brown) |
| `seal` | `#FFEB3B` (yellow) |

### Key details
- Bounding boxes are in 0–1000 normalized space and converted to pixels via `_bbox_to_pixels()`.
- Labels are only drawn on boxes taller than 8 pixels (to avoid clutter on tiny elements).
- Saved at 150 DPI with `bbox_inches="tight"`.

---

## 2. Reading-Order Visualization

**File:** `output/<doc_name>/plots/<doc>_page<NNN>_reading_order.png`

Renders the PDF page and overlays numbered circles on each content block. The numbering
follows MinerU's reading order (the order items appear in `content_list.json` for that page).

### Visual encoding
- Circle fill color follows a **plasma colormap** gradient: early blocks are yellow-green,
  late blocks are purple, making the flow direction immediately visible.
- A thin dashed grey line connects consecutive block centers to trace the reading path.
- A colorbar on the right side shows the "Reading order" scale from 1 to N.
- Circles are drawn at radius 7 with the block index (1-based) in white text inside.

### Interpreting the output
If circles follow a natural top-to-bottom, left-to-right progression and the gradient flows
smoothly, MinerU's reading-order detection is correct. Jumps (e.g., column-spanning blocks
re-ordered incorrectly) are visible as sudden color discontinuities.

---

## 3. Document Tree (Static PNG)

**File:** `output/<doc_name>/plots/<doc>_document_tree.png`

A hierarchical indented-list diagram showing the document's title structure with per-section
content counts. This is a static matplotlib figure, not interactive.

### Tree building: `build_doc_tree()`

Converts the flat `content_list` into a nested tree:
- `type=text` with `text_level >= 1` → internal title nodes
- All other types → leaf nodes attached to the nearest enclosing title
- Content before any title → attached directly to the document root

**Skipped heading levels:** If an H3 appears with no H2 ancestor, it attaches to the H1 (the
nearest higher-level ancestor). Phantom intermediate nodes are never inserted.

### Render: `_flatten_tree_for_render()`

DFS flattens the tree into `(depth, type, level, display_text, page_idx)` tuples for rendering.
Title rows show a summary of their descendant leaf counts (e.g., `"Introduction  [3 text  1 image]"`).
Up to 250 rows are rendered; longer documents are truncated with a `"... (truncated)"` marker.

### Color and indentation
Each row is colored by content type. Titles are drawn in blue with indentation proportional to
their level. Leaf counts appear inline in grey italics.

---

## 4. Interactive DOM Graph (D3 HTML)

**Files:**
- `output/<doc_name>/plots/<doc>_dom_full.html` — full document, leaves collapsed
- `output/<doc_name>/plots/<doc>_page<NNN>_dom.html` — single page, fully expanded

A self-contained HTML file embedding a **D3 v7 collapsible tree** (loaded from CDN).
The file is completely self-contained except for the D3 CDN dependency.

### Controls
| Action | Effect |
|--------|--------|
| Click a node | Toggle expand/collapse of its children |
| Scroll | Zoom in/out |
| Drag | Pan the view |
| Reset View button | Fit all visible nodes to viewport |

### Layout
The tree uses a **horizontal left-to-right layout** (depth grows rightward, siblings stack
vertically). Constants:
- `ROW_H = 22 px` — row spacing between siblings (≥ font size 11px, guarantees no overlap)
- `COL_W = 290 px` — column spacing between depth levels (~40 characters × 7px)
- `MAX_LBL = 40` — label characters before truncation with `…`

### Node types

| Node label prefix | Represents | Color |
|-------------------|-----------|-------|
| `[DOC] ...` | Document root | `#37474F` |
| `[H1] ... p.N` | H1 title | `#2196F3` (blue) |
| `[H2] ... p.N` | H2 title | same |
| `leaf_cluster` summary | Collapsed leaf group under a title | `#B0BEC5` (grey) |
| `typegroup` | All leaves of one type under a cluster | type color |
| Individual leaf | Single content block | type color, smaller circle |

### Collapse behavior
- Nodes beyond **depth 1** start collapsed on load.
- `fitVisible()` is called 350ms after render to auto-scale the initial view to fit all
  visible (non-collapsed) nodes.

### Full-doc vs per-page mode

**Full-doc** (`page_idx=None`): leaves under each title section are **collapsed** into a
summary cluster node → type-group intermediate nodes → individual leaf nodes (3 levels).
This keeps the graph navigable for large documents (100+ sections).

**Per-page** (`page_idx=N`): all nodes are **fully expanded** with individual leaves visible.
Used for inspecting a single page's structure in detail.

### Pipeline: `draw_dom_graph_viz()`

```
content_list → build_doc_tree() → _build_networkx_graph() → _nx_to_d3_json() → _write_d3_tree_html()
```

1. `build_doc_tree()` — builds the nested dict tree.
2. `_build_networkx_graph()` — converts to a `nx.DiGraph` with node attributes (`label`,
   `title` tooltip, `color`, `size`, `shape`). In full-doc mode, leaf nodes are collapsed
   into cluster/typegroup intermediates.
3. `_nx_to_d3_json()` — DFS converts the NetworkX graph to a nested D3 hierarchy dict.
   Circle radius: `r = clamp(int(size * 0.55), 6, 20)`.
4. `_write_d3_tree_html()` — substitutes `__DATA__` and `__TITLE__` placeholders in the
   `_D3_TREE_TEMPLATE` string and writes the HTML file.

---

## Using `generate_all_visualizations()`

The convenience function `generate_all_visualizations()` in `src/doc_maestro/visualizer.py` calls all four
render functions in sequence. It's the primary entry point used by both `scripts/visualize.py`
and `scripts/run_extraction.py`.

```python
from doc_maestro.visualizer import generate_all_visualizations

saved_paths = generate_all_visualizations(
    pdf_path=Path("data/pdfs/macdonals.pdf"),
    content_list=content_list,
    doc_name="macdonals",
    output_dir=Path("output"),
    pages=[0, 1, 2],       # which pages to render; None = first 3 with content
    dom=True,              # also generate HTML DOM files
)
```

Outputs land in `output/macdonals/plots/`. Returns a list of all saved `Path` objects.

---

## Standalone Visualization CLI

See [cli-reference.md](cli-reference.md) for full usage of `scripts/visualize.py`, which
lets you re-visualize any already-extracted document without re-running MinerU.
