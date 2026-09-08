# tests/task_composer/tasks/test_doc_reconstruction.py
"""Tests for DocReconstructionComposer."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from src.task_composer.base import DocumentContext
from src.task_composer.tasks.doc_reconstruction import DocReconstructionComposer


def _page(n: int = 3) -> ET.Element:
    page = ET.Element("page", {"idx": "0"})
    for i in range(n):
        el = ET.SubElement(page, "text-block", {"bbox": f"[0, {i * 10}, 50, {i * 10 + 9}]"})
        el.text = f"block {i}"
    return page


def _ctx(pages, tmp_path: Path) -> DocumentContext:
    return DocumentContext(doc_name="doc", pages=pages, img_root=tmp_path)


def test_task_name():
    assert DocReconstructionComposer.task_name == "doc_reconstruction"


def test_target_bbox_aligns_with_shuffled_input(tmp_path):
    page = _page(3)
    samples = DocReconstructionComposer(seed=0).compose_page(page, _ctx([page], tmp_path))
    s = samples[0]
    shuffled = s.payload["input.json"]
    target = s.payload["target.json"]
    assert len(shuffled) == len(target) == 3
    # Each input item maps to a real bbox from the page.
    page_bboxes = {f"[0, {i * 10}, 50, {i * 10 + 9}]" for i in range(3)}
    assert set(target) <= page_bboxes
    assert [item["idx"] for item in shuffled] == [0, 1, 2]


def test_page_with_one_element_yields_no_samples(tmp_path):
    page = _page(1)
    assert DocReconstructionComposer(seed=0).compose_page(page, _ctx([page], tmp_path)) == []
