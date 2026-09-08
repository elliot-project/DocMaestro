# tests/task_composer/tasks/test_dom_edition.py
"""Tests for DomEditionComposer."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from src.task_composer.base import DocumentContext
from src.task_composer.tasks.dom_edition import DomEditionComposer


def _page(n: int = 4) -> ET.Element:
    page = ET.Element("page", {"idx": "0"})
    for i in range(n):
        ET.SubElement(page, "text-block", {"bbox": f"[0, {i}, 1, {i + 1}]"}).text = f"t{i}"
    return page


def _ctx(pages, tmp_path: Path) -> DocumentContext:
    return DocumentContext(doc_name="doc", pages=pages, img_root=tmp_path)


def test_task_name():
    assert DomEditionComposer.task_name == "dom_edition"


def test_input_corrupted_target_clean(tmp_path):
    page = _page()
    samples = DomEditionComposer(seed=0).compose_page(page, _ctx([page], tmp_path))
    s = samples[0]
    assert "bbox" not in s.payload["input.txt"]
    assert "bbox" not in s.payload["target.txt"]
    assert s.payload["input.txt"] != s.payload["target.txt"]


def test_page_with_one_child_yields_no_samples(tmp_path):
    page = _page(n=1)
    assert DomEditionComposer(seed=0).compose_page(page, _ctx([page], tmp_path)) == []
