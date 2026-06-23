"""Tests for xml_ops shared GT helpers."""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET

from src.task_composer import xml_ops


def _page() -> ET.Element:
    page = ET.Element("page", {"idx": "2", "width": "600", "height": "800"})
    t = ET.SubElement(page, "title", {"bbox": "[0, 0, 600, 30]", "level": "1"})
    t.text = "Intro"
    tb = ET.SubElement(page, "text-block", {"bbox": "[0, 40, 600, 80]"})
    tb.text = "Body text."
    fig = ET.SubElement(page, "figure", {"bbox": "[0, 90, 300, 300]"})
    ET.SubElement(fig, "img", {"src": "images/f1.jpg"})
    cap = ET.SubElement(fig, "caption")
    cap.text = "Figure one"
    hdr = ET.SubElement(page, "page-header", {"bbox": "[0, 0, 600, 10]"})
    hdr.text = "running head"
    return page


def test_page_idx():
    assert xml_ops.page_idx(_page()) == 2


def test_page_reading_text_concatenates_in_order_and_skips_noise():
    text = xml_ops.page_reading_text(_page())
    assert text == "Intro\nBody text.\nFigure one"
    assert "running head" not in text


def test_text_blocks_with_bbox_only_text_bearing_top_level():
    blocks = xml_ops.text_blocks_with_bbox(_page())
    assert {"bbox": "[0, 0, 600, 30]", "tag": "title", "text": "Intro"} in blocks
    assert all(b["tag"] != "page-header" for b in blocks)
    assert all(b["tag"] != "figure" for b in blocks)  # figure has no direct text


def test_layout_elements_includes_every_bboxed_element():
    layout = xml_ops.layout_elements(_page())
    tags = {e["tag"] for e in layout}
    assert {"title", "text-block", "figure"} <= tags
    assert "page-header" not in tags  # noise excluded


def test_xml_to_markdown_maps_title_and_figure():
    md = xml_ops.xml_to_markdown(_page())
    assert "# Intro" in md
    assert "![Figure one](images/f1.jpg)" in md
    assert "Body text." in md


def test_strip_bbox_removes_all_bbox_attrs_keeps_text():
    out = xml_ops.strip_bbox(_page())
    assert "bbox" not in out
    assert "Body text." in out
    assert "Intro" in out


def test_mutate_tree_changes_structure_deterministically():
    clean = xml_ops.strip_bbox(_page())
    a = xml_ops.mutate_tree(clean, random.Random(0))
    b = xml_ops.mutate_tree(clean, random.Random(0))
    assert a == b  # deterministic for a fixed seed
    assert a != clean  # something changed
