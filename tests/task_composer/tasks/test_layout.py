# tests/task_composer/tasks/test_layout.py
"""Tests for LayoutComposer."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from src.task_composer.base import DocumentContext
from src.task_composer.tasks.layout import LayoutComposer


def _ctx(pages, tmp_path: Path) -> DocumentContext:
    return DocumentContext(doc_name="doc", pages=pages, img_root=tmp_path)


def test_task_name():
    assert LayoutComposer.task_name == "layout"


def test_includes_figures_and_text(tmp_path):
    page = ET.Element("page", {"idx": "0"})
    ET.SubElement(page, "text-block", {"bbox": "[0, 0, 1, 1]"}).text = "x"
    ET.SubElement(page, "figure", {"bbox": "[2, 2, 3, 3]"})
    samples = LayoutComposer().compose_page(page, _ctx([page], tmp_path))
    layout = samples[0].payload["target.json"]
    assert {"tag": "figure", "bbox": "[2, 2, 3, 3]"} in layout
    assert {"tag": "text-block", "bbox": "[0, 0, 1, 1]"} in layout


def test_empty_page_yields_no_samples(tmp_path):
    page = ET.Element("page", {"idx": "0"})
    assert LayoutComposer().compose_page(page, _ctx([page], tmp_path)) == []
