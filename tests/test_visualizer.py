"""Unit tests for visualizer.py — tree building and color/label helpers."""

from src.visualizer import (
    CONTENT_TYPE_COLORS,
    _build_networkx_graph,
    _count_leaves,
    _flatten_tree_for_render,
    _type_color,
    _type_label,
    build_doc_tree,
    draw_dom_graph_viz,
)


class TestTypeColor:
    def test_plain_text_is_green(self):
        assert _type_color("text", 0) == CONTENT_TYPE_COLORS["text"]

    def test_title_level_1_is_blue(self):
        assert _type_color("text", 1) == CONTENT_TYPE_COLORS["title"]

    def test_title_level_3_is_blue(self):
        assert _type_color("text", 3) == CONTENT_TYPE_COLORS["title"]

    def test_image_color(self):
        assert _type_color("image") == CONTENT_TYPE_COLORS["image"]

    def test_table_color(self):
        assert _type_color("table") == CONTENT_TYPE_COLORS["table"]

    def test_unknown_type_returns_default(self):
        color = _type_color("future_unknown_type")
        assert color == "#BDBDBD"


class TestTypeLabel:
    def test_plain_text_returns_text(self):
        assert _type_label("text", 0) == "text"

    def test_h1_label(self):
        assert _type_label("text", 1) == "h1"

    def test_h3_label(self):
        assert _type_label("text", 3) == "h3"

    def test_image_label(self):
        assert _type_label("image") == "image"


class TestBuildDocTree:
    def _make_item(self, type_, text="", text_level=0, page_idx=0):
        item = {"type": type_, "text": text, "page_idx": page_idx}
        if text_level:
            item["text_level"] = text_level
        return item

    def test_empty_list_produces_root_with_no_children(self):
        tree = build_doc_tree([])
        assert tree["type"] == "document"
        assert tree["children"] == []

    def test_root_label(self):
        tree = build_doc_tree([], doc_name="my_doc")
        assert tree["text"] == "my_doc"

    def test_single_title_becomes_child_of_root(self):
        items = [self._make_item("text", "Introduction", text_level=1)]
        tree = build_doc_tree(items)
        assert len(tree["children"]) == 1
        assert tree["children"][0]["type"] == "title"
        assert tree["children"][0]["level"] == 1

    def test_leaf_under_title(self):
        items = [
            self._make_item("text", "Intro", text_level=1),
            self._make_item("text", "Body paragraph"),
        ]
        tree = build_doc_tree(items)
        h1 = tree["children"][0]
        assert len(h1["children"]) == 1
        assert h1["children"][0]["type"] == "text"

    def test_nested_h2_under_h1(self):
        items = [
            self._make_item("text", "Chapter 1", text_level=1),
            self._make_item("text", "Section 1.1", text_level=2),
            self._make_item("text", "Body text"),
        ]
        tree = build_doc_tree(items)
        h1 = tree["children"][0]
        h2 = h1["children"][0]
        assert h2["type"] == "title"
        assert h2["level"] == 2
        assert h2["children"][0]["type"] == "text"

    def test_second_h1_is_sibling_not_child(self):
        items = [
            self._make_item("text", "Chapter 1", text_level=1),
            self._make_item("text", "Chapter 2", text_level=1),
        ]
        tree = build_doc_tree(items)
        assert len(tree["children"]) == 2
        assert all(c["level"] == 1 for c in tree["children"])

    def test_skipped_level_attaches_to_nearest_ancestor(self):
        """H3 with no H2 ancestor should attach to H1, not create phantom H2."""
        items = [
            self._make_item("text", "Chapter 1", text_level=1),
            self._make_item("text", "Subsub", text_level=3),
        ]
        tree = build_doc_tree(items)
        h1 = tree["children"][0]
        # H3 should be a child of H1 (nearest higher-level ancestor)
        assert len(h1["children"]) == 1
        assert h1["children"][0]["level"] == 3

    def test_content_before_any_title_goes_to_root(self):
        items = [
            self._make_item("image", "cover"),
            self._make_item("text", "Chapter 1", text_level=1),
        ]
        tree = build_doc_tree(items)
        # First child is the image (leaf under root), second is the title
        assert tree["children"][0]["type"] == "image"
        assert tree["children"][1]["type"] == "title"

    def test_table_under_section(self):
        items = [
            self._make_item("text", "Results", text_level=1),
            self._make_item("table", ""),
        ]
        tree = build_doc_tree(items)
        h1 = tree["children"][0]
        assert h1["children"][0]["type"] == "table"

    def test_multiple_leaves_under_title(self):
        items = [
            self._make_item("text", "Section", text_level=1),
            self._make_item("text", "para1"),
            self._make_item("text", "para2"),
            self._make_item("image", "fig1"),
        ]
        tree = build_doc_tree(items)
        h1 = tree["children"][0]
        assert len(h1["children"]) == 3


class TestCountLeaves:
    def test_counts_direct_non_title_children(self):
        node = {
            "children": [
                {"type": "text", "children": []},
                {"type": "text", "children": []},
                {"type": "image", "children": []},
                {"type": "title", "children": []},  # should NOT be counted
            ]
        }
        counts = _count_leaves(node)
        assert counts == {"text": 2, "image": 1}

    def test_empty_node_returns_empty_counts(self):
        assert _count_leaves({"children": []}) == {}

    def test_only_title_children_returns_empty(self):
        node = {"children": [{"type": "title", "children": []}]}
        assert _count_leaves(node) == {}


class TestFlattenTreeForRender:
    def test_flat_list_has_one_row_per_child(self):
        tree = build_doc_tree(
            [
                {"type": "text", "text": "H1", "text_level": 1, "page_idx": 0},
                {"type": "text", "text": "H2", "text_level": 2, "page_idx": 1},
            ]
        )
        rows = _flatten_tree_for_render(tree)
        assert len(rows) == 2

    def test_depth_increases_for_nested_titles(self):
        tree = build_doc_tree(
            [
                {"type": "text", "text": "H1", "text_level": 1, "page_idx": 0},
                {"type": "text", "text": "H2", "text_level": 2, "page_idx": 0},
            ]
        )
        rows = _flatten_tree_for_render(tree)
        h1_depth = rows[0][0]
        h2_depth = rows[1][0]
        assert h2_depth > h1_depth

    def test_max_items_truncates(self):
        items = [
            {"type": "text", "text": f"H{i}", "text_level": 1, "page_idx": i}
            for i in range(20)
        ]
        tree = build_doc_tree(items)
        rows = _flatten_tree_for_render(tree, max_items=5)
        assert len(rows) <= 6  # 5 items + possible truncation marker


class TestDomGraphViz:
    def _make_items(self) -> list[dict]:
        return [
            {"type": "text", "text": "Chapter 1", "text_level": 1, "page_idx": 0, "bbox": [10, 10, 500, 50]},
            {"type": "text", "text": "Body paragraph", "page_idx": 0, "bbox": [10, 60, 500, 100]},
            {"type": "image", "text": "", "page_idx": 0, "bbox": [10, 110, 500, 300]},
            {"type": "text", "text": "Chapter 2", "text_level": 1, "page_idx": 1, "bbox": [10, 10, 500, 50]},
            {"type": "text", "text": "More text", "page_idx": 1, "bbox": [10, 60, 500, 100]},
        ]

    def test_full_doc_graph_has_root_node(self):
        tree = build_doc_tree(self._make_items(), "test_doc")
        G = _build_networkx_graph(tree, collapse_leaves=True)
        assert "root" in G.nodes

    def test_title_nodes_present(self):
        tree = build_doc_tree(self._make_items(), "test_doc")
        G = _build_networkx_graph(tree, collapse_leaves=True)
        title_nodes = [n for n, d in G.nodes(data=True) if "[H" in d.get("label", "")]
        assert len(title_nodes) >= 2

    def test_leaf_cluster_in_full_doc(self):
        tree = build_doc_tree(self._make_items(), "test_doc")
        G = _build_networkx_graph(tree, collapse_leaves=True)
        cluster_nodes = [n for n in G.nodes if str(n).startswith("leaf_cluster_")]
        assert len(cluster_nodes) >= 1

    def test_per_page_no_leaf_clusters(self):
        tree = build_doc_tree(self._make_items(), "test_doc")
        G = _build_networkx_graph(tree, collapse_leaves=False)
        cluster_nodes = [n for n in G.nodes if str(n).startswith("leaf_cluster_")]
        assert len(cluster_nodes) == 0

    def test_edges_are_directed_parent_to_child(self):
        tree = build_doc_tree(self._make_items(), "test_doc")
        G = _build_networkx_graph(tree, collapse_leaves=True)
        assert G.in_degree("root") == 0
        assert G.out_degree("root") > 0

    def test_tooltip_contains_page_idx(self):
        tree = build_doc_tree(self._make_items(), "test_doc")
        G = _build_networkx_graph(tree, collapse_leaves=True)
        title_tooltips = [
            d.get("title", "")
            for _, d in G.nodes(data=True)
            if "[H" in d.get("label", "")
        ]
        assert all("p." in t for t in title_tooltips)

    def test_empty_content_list_produces_root_only(self):
        tree = build_doc_tree([], "empty_doc")
        G = _build_networkx_graph(tree, collapse_leaves=True)
        assert list(G.nodes) == ["root"]

    def test_output_html_saved(self, tmp_path):
        out = tmp_path / "test_dom.html"
        draw_dom_graph_viz(self._make_items(), "test_doc", out)
        assert out.exists()
        assert out.stat().st_size > 0
