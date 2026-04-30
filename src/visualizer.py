"""Document visualization: layout bboxes, reading order, and document tree."""

import math

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pypdfium2 as pdfium
from PIL import Image

# ── Color scheme ─────────────────────────────────────────────────────────────

CONTENT_TYPE_COLORS: dict[str, str] = {
    "text": "#4CAF50",
    "image": "#FF9800",
    "table": "#F44336",
    "chart": "#E91E63",
    "equation": "#9C27B0",
    "code": "#00BCD4",
    "list": "#8BC34A",
    "header": "#607D8B",
    "footer": "#607D8B",
    "page_number": "#9E9E9E",
    "aside_text": "#FF5722",
    "page_footnote": "#795548",
    "seal": "#FFEB3B",
    "index": "#90A4AE",
    "title": "#2196F3",
}
DEFAULT_COLOR = "#BDBDBD"

CONTENT_TYPE_ICONS: dict[str, str] = {
    "text": "P",
    "title": "S",
    "image": "[img]",
    "table": "[tbl]",
    "chart": "[cht]",
    "equation": "[eq]",
    "code": "[code]",
    "list": "[list]",
    "header": "[hdr]",
    "footer": "[ftr]",
    "page_number": "[#]",
    "aside_text": "[aside]",
    "page_footnote": "[fn]",
    "seal": "[seal]",
}


def _type_color(content_type: str, text_level: int = 0) -> str:
    if content_type == "text" and text_level >= 1:
        return CONTENT_TYPE_COLORS["title"]
    return CONTENT_TYPE_COLORS.get(content_type, DEFAULT_COLOR)


def _type_label(content_type: str, text_level: int = 0) -> str:
    if content_type == "text" and text_level >= 1:
        return f"h{text_level}"
    return content_type


# ── PDF rendering ─────────────────────────────────────────────────────────────


def render_pdf_page(pdf_path: Path, page_idx: int, scale: float = 2.0) -> Image.Image:
    """Render a single PDF page to a PIL Image.

    Args:
        pdf_path: Path to the PDF file.
        page_idx: Zero-based page index.
        scale: Render scale factor (2.0 ≈ 144 DPI). Higher = more pixels.

    Returns:
        PIL Image of the rendered page in RGB.
    """
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        page = doc[page_idx]
        bitmap = page.render(scale=scale)
        return bitmap.to_pil().convert("RGB")
    finally:
        doc.close()


def _bbox_to_pixels(bbox: list, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert 0-1000 normalized bbox to absolute pixel coordinates.

    content_list.json bboxes use top-left origin with x,y in [0, 1000].
    The rendered page image also uses top-left origin, so we scale directly.

    Args:
        bbox: [x0, y0, x1, y1] with values in 0-1000 range.
        img_w: Rendered image width in pixels.
        img_h: Rendered image height in pixels.

    Returns:
        (x0_px, y0_px, x1_px, y1_px) in pixel space.
    """
    x0, y0, x1, y1 = bbox
    return (
        x0 / 1000 * img_w,
        y0 / 1000 * img_h,
        x1 / 1000 * img_w,
        y1 / 1000 * img_h,
    )


# ── Layout BBox visualization ─────────────────────────────────────────────────


def draw_layout_viz(
    pdf_path: Path,
    content_list: list[dict],
    page_idx: int,
    output_path: Path,
    *,
    scale: float = 2.0,
) -> None:
    """Render one PDF page overlaid with color-coded bounding boxes by content type.

    Args:
        pdf_path: Path to the source PDF.
        content_list: Flat content list from content_list.json.
        page_idx: Zero-based page index to visualize.
        output_path: File path for the saved PNG.
        scale: PDF render scale factor.
    """
    page_items = [x for x in content_list if x.get("page_idx") == page_idx and x.get("bbox")]
    page_img = render_pdf_page(pdf_path, page_idx, scale=scale)
    img_w, img_h = page_img.size

    fig, ax = plt.subplots(1, 1, figsize=(img_w / 150, img_h / 150))
    ax.imshow(np.array(page_img))
    ax.set_axis_off()

    seen_types: set[str] = set()
    for item in page_items:
        ctype = item.get("type", "text")
        tlevel = item.get("text_level", 0) or 0
        color = _type_color(ctype, tlevel)
        label = _type_label(ctype, tlevel)
        x0, y0, x1, y1 = _bbox_to_pixels(item["bbox"], img_w, img_h)
        w, h = x1 - x0, y1 - y0

        rect = mpatches.FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="square,pad=0",
            linewidth=1.5,
            edgecolor=color,
            facecolor=color,
            alpha=0.18,
        )
        ax.add_patch(rect)

        # Type label at top-left of box (only if box is tall enough)
        if h > 8:
            ax.text(
                x0 + 2,
                y0 + 2,
                label,
                fontsize=5,
                color=color,
                va="top",
                ha="left",
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, pad=0.5, edgecolor="none"),
            )
        seen_types.add(label)

    # Legend
    legend_handles = []
    for lbl in sorted(seen_types):
        raw_type = "text" if lbl.startswith("h") else lbl
        c = CONTENT_TYPE_COLORS.get(
            raw_type, CONTENT_TYPE_COLORS.get("title") if lbl.startswith("h") else DEFAULT_COLOR
        )
        legend_handles.append(mpatches.Patch(facecolor=c, alpha=0.6, label=lbl))
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=6,
            framealpha=0.85,
            ncol=2,
            title="Content type",
            title_fontsize=6,
        )

    ax.set_title(
        f"Page {page_idx} — Layout & Content Types  ({len(page_items)} blocks)",
        fontsize=9,
        pad=4,
    )
    fig.tight_layout(pad=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Reading-order visualization ───────────────────────────────────────────────


def draw_reading_order_viz(
    pdf_path: Path,
    content_list: list[dict],
    page_idx: int,
    output_path: Path,
    *,
    scale: float = 2.0,
) -> None:
    """Render one PDF page with numbered reading-order circles on each content block.

    Each block is numbered in reading order (1-indexed). Circle color is taken
    from a viridis gradient so early blocks are yellow-green and late blocks
    are purple, making the flow direction immediately visible.

    A thin grey line connects consecutive block centers to trace the path.

    Args:
        pdf_path: Path to the source PDF.
        content_list: Flat content list from content_list.json.
        page_idx: Zero-based page index to visualize.
        output_path: File path for the saved PNG.
        scale: PDF render scale factor.
    """
    page_items = [x for x in content_list if x.get("page_idx") == page_idx and x.get("bbox")]
    if not page_items:
        return

    page_img = render_pdf_page(pdf_path, page_idx, scale=scale)
    img_w, img_h = page_img.size

    fig, ax = plt.subplots(1, 1, figsize=(img_w / 150, img_h / 150))
    ax.imshow(np.array(page_img))
    ax.set_axis_off()

    n = len(page_items)
    cmap = plt.get_cmap("plasma")
    centers: list[tuple[float, float]] = []

    for i, item in enumerate(page_items):
        x0, y0, x1, y1 = _bbox_to_pixels(item["bbox"], img_w, img_h)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        centers.append((cx, cy))

        frac = i / max(n - 1, 1)
        color = cmap(frac)

        # Subtle bbox outline
        rect = mpatches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=1,
            edgecolor=color,
            facecolor="none",
            alpha=0.5,
        )
        ax.add_patch(rect)

        # Numbered circle at top-left corner of the block
        circle_x, circle_y = x0 + 6, y0 + 6
        circle = plt.Circle((circle_x, circle_y), radius=7, color=color, zorder=4)
        ax.add_patch(circle)
        ax.text(
            circle_x,
            circle_y,
            str(i + 1),
            fontsize=4.5,
            color="white",
            va="center",
            ha="center",
            fontweight="bold",
            zorder=5,
        )

    # Thin connecting lines between consecutive block centers
    if len(centers) > 1:
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        ax.plot(xs, ys, color="grey", linewidth=0.4, alpha=0.35, zorder=2, linestyle="--")

    # Colorbar legend (order indicator)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=n))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Reading order", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.set_title(
        f"Page {page_idx} — Reading Order  ({n} blocks)",
        fontsize=9,
        pad=4,
    )
    fig.tight_layout(pad=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Document tree ─────────────────────────────────────────────────────────────


def build_doc_tree(content_list: list[dict], doc_name: str = "document") -> dict:
    """Build a hierarchical document tree from a flat content list.

    Titles create internal nodes; all other content types are leaves attached
    to the nearest enclosing title. When a heading level is skipped (e.g.
    H1 followed immediately by H3 with no H2), the H3 is attached to the H1
    — the nearest higher-level ancestor — rather than inserting phantom nodes.

    Args:
        content_list: Flat list from content_list.json (reading order).
        doc_name: Label for the root node.

    Returns:
        Nested dict tree: root → children (titles + leaves).
    """
    root: dict = {"type": "document", "text": doc_name, "level": 0, "children": []}
    # stack holds (title_level, node); root acts as level-0 anchor
    stack: list[tuple[int, dict]] = [(0, root)]

    for item in content_list:
        ctype = item.get("type", "text")
        text_level = item.get("text_level", 0) or 0
        is_title = ctype == "text" and text_level >= 1

        if is_title:
            # Pop until we find a strictly lower-level ancestor
            while len(stack) > 1 and stack[-1][0] >= text_level:
                stack.pop()
            node: dict = {
                "type": "title",
                "level": text_level,
                "text": item.get("text", "").strip()[:80],
                "page_idx": item.get("page_idx", 0),
                "bbox": item.get("bbox"),
                "children": [],
            }
            stack[-1][1]["children"].append(node)
            stack.append((text_level, node))
        else:
            leaf: dict = {
                "type": ctype,
                "text": item.get("text", "").strip()[:60],
                "page_idx": item.get("page_idx", 0),
                "bbox": item.get("bbox"),
                "children": [],
            }
            stack[-1][1]["children"].append(leaf)

    return root


def _count_leaves(node: dict) -> dict[str, int]:
    """Recursively count direct non-title children by type.

    Args:
        node: A tree node dict with a 'children' list.

    Returns:
        Dict mapping content type to count of direct leaf children.
    """
    counts: dict[str, int] = {}
    for child in node.get("children", []):
        if child.get("type") != "title":
            t = child.get("type", "text")
            counts[t] = counts.get(t, 0) + 1
    return counts


def _all_descendant_leaf_counts(node: dict) -> dict[str, int]:
    """Recursively count all non-title descendants of a node by content type.

    Args:
        node: Tree node with a 'children' list.

    Returns:
        Dict mapping content type to total descendant leaf count.
    """
    counts: dict[str, int] = {}
    for child in node.get("children", []):
        if child.get("type") != "title":
            t = child.get("type", "text")
            counts[t] = counts.get(t, 0) + 1
        else:
            for k, v in _all_descendant_leaf_counts(child).items():
                counts[k] = counts.get(k, 0) + v
    return counts


def _build_networkx_graph(tree: dict, *, collapse_leaves: bool = True) -> nx.DiGraph:
    """Build a directed NetworkX graph from the document tree.

    In collapse_leaves mode (full-doc), leaf nodes under each title are merged
    into a single summary node to keep the graph readable for large documents.
    In expanded mode (per-page), every node is shown individually.

    Args:
        tree: Root node from build_doc_tree().
        collapse_leaves: If True, collapse direct leaf children of titles into
            one summary node per parent. Title children are always expanded.

    Returns:
        Directed graph where each node carries: label, title (tooltip),
        color, size, shape attributes.
    """
    G: nx.DiGraph = nx.DiGraph()
    _counter: list[int] = [0]

    def _next_id() -> str:
        _counter[0] += 1
        return f"n{_counter[0]}"

    def _walk(node: dict, parent_id: str | None) -> None:
        ntype = node.get("type", "text")
        node_id: str = "root" if ntype == "document" else _next_id()
        text = node.get("text", "")
        level = node.get("level", 0)
        page_idx = node.get("page_idx", 0)
        bbox = node.get("bbox")
        children = node.get("children", [])

        desc_counts = _all_descendant_leaf_counts(node)
        total_desc = sum(desc_counts.values())
        counts_str = "  ".join(f"{v} {k}" for k, v in sorted(desc_counts.items()) if v > 0)
        bbox_str = f"bbox: {bbox}" if bbox else "bbox: none"

        if ntype == "document":
            label = f"[DOC] {text[:30]}"
            shape = "star"
            color = "#37474F"
            size = 30
            tooltip = f"type: document\n{text}\n{counts_str}"
        elif ntype == "title":
            label = f"[H{level}] {text[:30]}"
            shape = "dot"
            color = CONTENT_TYPE_COLORS["title"]
            size = min(50, int(10 + 3 * math.sqrt(max(total_desc, 1))))
            tooltip = f"type: h{level}\n{text[:80]}\np.{page_idx}\n{bbox_str}\n{counts_str}"
        else:
            icon = CONTENT_TYPE_ICONS.get(ntype, "?")
            label = f"[{icon}] {text[:30]}"
            shape = "square"
            color = CONTENT_TYPE_COLORS.get(ntype, DEFAULT_COLOR)
            size = 8
            tooltip = f"type: {ntype}\n{text[:80]}\np.{page_idx}\n{bbox_str}"

        G.add_node(node_id, label=label, title=tooltip, color=color, size=size, shape=shape)

        if parent_id is not None:
            if ntype == "title":
                weight = 2.0 if parent_id == "root" else 1.5
                G.add_edge(parent_id, node_id, weight=weight)
            else:
                G.add_edge(parent_id, node_id, weight=0.8, dashes=True)

        if collapse_leaves and ntype in ("document", "title"):
            title_children = [c for c in children if c.get("type") == "title"]
            leaf_children = [c for c in children if c.get("type") != "title"]

            for child in title_children:
                _walk(child, parent_id=node_id)

            if leaf_children:
                leaf_counts: dict[str, int] = {}
                for c in leaf_children:
                    t = c.get("type", "text")
                    leaf_counts[t] = leaf_counts.get(t, 0) + 1
                summary = " · ".join(f"{v} {k}" for k, v in sorted(leaf_counts.items()))
                cluster_id = f"leaf_cluster_{node_id}"
                G.add_node(
                    cluster_id,
                    label=summary,
                    title=f"leaf cluster\n{summary}\np.{page_idx}",
                    color="#B0BEC5",
                    size=8,
                    shape="square",
                )
                G.add_edge(node_id, cluster_id, weight=1.0, dashes=True)
        else:
            for child in children:
                _walk(child, parent_id=node_id)

    _walk(tree, parent_id=None)
    return G


def _nx_to_pyvis(G: nx.DiGraph, output_path: Path) -> None:
    """Render a NetworkX directed graph to a self-contained PyVis HTML file.

    Uses force_atlas_2based physics so title clusters naturally group by
    section proximity while leaf summary nodes orbit their parent titles.

    Args:
        G: Directed graph with node attributes: label, title, color, size, shape.
        output_path: Destination .html file path (created if missing).
    """
    from pyvis.network import Network  # deferred: large optional dep

    net = Network(
        directed=True,
        height="900px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        notebook=False,
    )

    for node_id, attrs in G.nodes(data=True):
        net.add_node(
            str(node_id),
            label=attrs.get("label", str(node_id)),
            title=attrs.get("title", ""),
            color=attrs.get("color", DEFAULT_COLOR),
            size=attrs.get("size", 10),
            shape=attrs.get("shape", "dot"),
        )

    for src, dst, attrs in G.edges(data=True):
        edge_kwargs: dict = {"width": attrs.get("weight", 1.0)}
        if attrs.get("dashes"):
            edge_kwargs["dashes"] = True
        net.add_edge(str(src), str(dst), **edge_kwargs)

    net.set_options("""{
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "springLength": 120
        },
        "solver": "forceAtlas2Based"
      }
    }""")

    net.write_html(str(output_path))


def draw_dom_graph_viz(
    content_list: list[dict],
    doc_name: str,
    output_path: Path,
    *,
    page_idx: int | None = None,
    max_nodes: int = 500,
) -> None:
    """Render the document DOM as an interactive directed graph saved as HTML.

    Full-document mode (page_idx=None): title nodes shown individually,
    direct leaf children collapsed into one summary node per section to keep
    the graph readable for documents with hundreds of blocks.

    Per-page mode (page_idx=N): content list filtered to that page, all nodes
    expanded with full granularity so every block is a visible graph node.

    Args:
        content_list: Flat content list from content_list.json.
        doc_name: Document name used as the root node label.
        output_path: Destination path for the .html file.
        page_idx: If set, render only nodes from this page (0-based).
        max_nodes: Reserved for future pruning; not enforced in this version.
    """
    items = (
        [x for x in content_list if x.get("page_idx") == page_idx]
        if page_idx is not None
        else content_list
    )
    tree = build_doc_tree(items, doc_name)
    G = _build_networkx_graph(tree, collapse_leaves=(page_idx is None))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _nx_to_pyvis(G, output_path)


def _flatten_tree_for_render(
    node: dict,
    depth: int = 0,
    result: list | None = None,
    max_items: int = 300,
    _at_root: bool = True,
) -> list[tuple[int, str, int, str, int]]:
    """DFS flatten to (depth, type, level, display_text, page_idx) tuples.

    Title nodes form the structural skeleton. Leaf content under titles is
    collapsed into a count summary on the parent title's row. Only root-level
    leaves (content before the first title) are shown individually.

    Args:
        node: Current tree node.
        depth: Current render depth (0 = root children).
        result: Accumulator list (mutated in place).
        max_items: Maximum rows to emit before truncating.
        _at_root: True only for the document root's direct children.

    Returns:
        List of (depth, type, level, display_text, page_idx) tuples.
    """
    if result is None:
        result = []

    for child in node.get("children", []):
        if len(result) >= max_items:
            result.append((depth, "truncated", 0, "... (truncated)", -1))
            break

        ctype = child.get("type", "text")

        if ctype == "title":
            leaf_counts = _all_descendant_leaf_counts(child)
            summary = "  ".join(f"{v} {k}" for k, v in sorted(leaf_counts.items()) if v > 0)
            display = child.get("text", "")[:70]
            if summary:
                display += f"  [{summary}]"
            result.append(
                (depth, "title", child.get("level", 1), display, child.get("page_idx", 0))
            )
            # Recurse into this title's children — but mark as not-root so
            # leaves inside titles are suppressed
            _flatten_tree_for_render(child, depth + 1, result, max_items, _at_root=False)
        elif _at_root:
            # Pre-title root-level content (cover images, front matter)
            icon = CONTENT_TYPE_ICONS.get(ctype, "-")
            text = child.get("text", "")[:55] or f"({ctype})"
            result.append((depth, ctype, 0, f"{icon}  {text}", child.get("page_idx", 0)))
        # else: leaf under a title — suppressed (already counted in parent summary)

    return result


def draw_doc_tree_viz(
    content_list: list[dict],
    doc_name: str,
    output_path: Path,
    *,
    max_items: int = 250,
) -> None:
    """Render the document tree as a hierarchical indented diagram.

    Titles form the structural backbone with per-section leaf-type counts.
    Content blocks appear as leaves under their enclosing title.

    Args:
        content_list: Flat content list from content_list.json.
        doc_name: Document name used as root label and figure title.
        output_path: File path for the saved PNG.
        max_items: Maximum rows to render before truncating.
    """
    tree = build_doc_tree(content_list, doc_name)
    rows = _flatten_tree_for_render(tree, depth=0, max_items=max_items)

    if not rows:
        return

    n_rows = len(rows) + 2  # +2 for root label
    row_h_inches = 0.28
    fig_h = max(8, n_rows * row_h_inches)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows)
    ax.set_axis_off()

    indent_unit = 0.020
    row_h = 1.0
    y = n_rows - 1

    # Root label
    ax.text(
        0,
        y,
        f"[DOC]  {doc_name}",
        fontsize=9,
        fontweight="bold",
        color="#212121",
        va="top",
        fontfamily="monospace",
    )
    y -= row_h

    for depth, ctype, level, text, page_idx in rows:
        if ctype == "title":
            color = CONTENT_TYPE_COLORS["title"]
            weight = "bold"
            prefix = ">> " + ("-" * level) + " "
            fs = max(7, 9 - level * 0.5)
        elif ctype == "truncated":
            color = "#9E9E9E"
            weight = "normal"
            prefix = "  "
            fs = 7
        else:
            color = _type_color(ctype)
            weight = "normal"
            prefix = "  "
            fs = 7

        x = indent_unit * (depth + 1)
        page_tag = f"  p.{page_idx}" if page_idx >= 0 else ""
        label = f"{prefix}{text}{page_tag}"

        ax.text(
            x,
            y,
            label,
            fontsize=fs,
            color=color,
            fontweight=weight,
            va="top",
            ha="left",
            fontfamily="monospace",
        )
        y -= row_h

    # Type legend
    legend_handles = [
        mpatches.Patch(color=CONTENT_TYPE_COLORS["title"], label="title / heading"),
    ]
    for t in ["text", "image", "table", "chart", "equation", "code", "list"]:
        if any(r[1] == t for r in rows):
            legend_handles.append(
                mpatches.Patch(color=CONTENT_TYPE_COLORS.get(t, DEFAULT_COLOR), label=t)
            )
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=6,
        framealpha=0.9,
        title="Content type",
        title_fontsize=6,
    )

    ax.set_title(f"Document Structure Tree — {doc_name}", fontsize=10, pad=6)
    fig.tight_layout(pad=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Batch runner ──────────────────────────────────────────────────────────────


def generate_all_visualizations(
    pdf_path: Path,
    content_list: list[dict],
    doc_name: str,
    output_dir: Path,
    pages: list[int] | None = None,
) -> list[Path]:
    """Generate all three visualization types for the given document.

    Produces layout-bbox PNGs and reading-order PNGs per page, plus one
    document-tree PNG for the full document.

    Args:
        pdf_path: Path to the source PDF.
        content_list: Flat content list from content_list.json.
        doc_name: Document name used in filenames and titles.
        output_dir: Root output directory (files go under output_dir/plots/).
        pages: List of zero-based page indices to visualize. If None, uses
            the first 3 pages that have content.

    Returns:
        List of Paths to all saved PNG files.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if pages is None:
        all_pages = sorted({x.get("page_idx", 0) for x in content_list if x.get("bbox")})
        pages = all_pages[:3]

    saved: list[Path] = []

    for page_idx in pages:
        layout_path = plots_dir / f"{doc_name}_page{page_idx:03d}_layout_bboxes.png"
        draw_layout_viz(pdf_path, content_list, page_idx, layout_path)
        print(f"  saved: {layout_path.name}")
        saved.append(layout_path)

        order_path = plots_dir / f"{doc_name}_page{page_idx:03d}_reading_order.png"
        draw_reading_order_viz(pdf_path, content_list, page_idx, order_path)
        print(f"  saved: {order_path.name}")
        saved.append(order_path)

    tree_path = plots_dir / f"{doc_name}_document_tree.png"
    draw_doc_tree_viz(content_list, doc_name, tree_path)
    print(f"  saved: {tree_path.name}")
    saved.append(tree_path)

    return saved
