# tests/task_composer/tasks/test_page_tree.py
"""Tests for PageTreeComposer."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from doc_maestro.task_composer.base import DocumentContext
from doc_maestro.task_composer.tasks.page_tree import PageTreeComposer


def _ctx(pages, tmp_path: Path) -> DocumentContext:
    return DocumentContext(doc_name="doc", pages=pages, img_root=tmp_path)


def test_task_name():
    assert PageTreeComposer.task_name == "page_tree"


def test_target_is_bbox_free_xml(tmp_path):
    page = ET.Element("page", {"idx": "0"})
    ET.SubElement(page, "text-block", {"bbox": "[0, 0, 1, 1]"}).text = "hi"
    samples = PageTreeComposer().compose_page(page, _ctx([page], tmp_path))
    target = samples[0].payload["target.txt"]
    assert "bbox" not in target
    assert "hi" in target


def test_empty_page_yields_no_samples(tmp_path):
    page = ET.Element("page", {"idx": "0"})
    assert PageTreeComposer().compose_page(page, _ctx([page], tmp_path)) == []
