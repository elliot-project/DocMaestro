# tests/task_composer/tasks/test_page_sorting.py
"""Tests for PageSortingComposer."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from doc_maestro.task_composer.tasks.page_sorting import PageSortingComposer


def _doc(n_pages: int) -> ET.Element:
    root = ET.Element("document", {"name": "doc"})
    for i in range(n_pages):
        page = ET.SubElement(root, "page", {"idx": str(i)})
        ET.SubElement(page, "page-number", {"bbox": "[0, 0, 10, 10]"}).text = str(i + 1)
    return root


def test_task_name():
    assert PageSortingComposer.task_name == "page_sorting"


def test_single_sample_restores_order(tmp_path: Path):
    root = _doc(4)
    samples = PageSortingComposer(seed=0).compose_document(root, tmp_path)
    assert len(samples) == 1
    s = samples[0]
    shuffled = s.payload["input.json"]
    target = s.payload["target.json"]
    restored = [shuffled[target[i]]["orig_page_idx"] for i in range(len(shuffled))]
    assert restored == [0, 1, 2, 3]
    assert "page_number_bboxes" in shuffled[0]


def test_single_page_doc_yields_no_samples(tmp_path: Path):
    assert PageSortingComposer(seed=0).compose_document(_doc(1), tmp_path) == []
