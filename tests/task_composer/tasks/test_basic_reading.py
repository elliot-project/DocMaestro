# tests/task_composer/tasks/test_basic_reading.py
"""Tests for BasicReadingComposer."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from src.task_composer.base import DocumentContext
from src.task_composer.tasks.basic_reading import BasicReadingComposer


def _page(idx: int = 0) -> ET.Element:
    page = ET.Element("page", {"idx": str(idx)})
    tb = ET.SubElement(page, "text-block", {"bbox": "[0, 0, 10, 10]"})
    tb.text = "Hello world"
    return page


def _ctx(pages, tmp_path: Path) -> DocumentContext:
    return DocumentContext(doc_name="doc", pages=pages, img_root=tmp_path)


def test_task_name():
    assert BasicReadingComposer.task_name == "basic_reading"


def test_one_sample_with_full_text(tmp_path):
    page = _page(1)
    samples = BasicReadingComposer().compose_page(page, _ctx([page], tmp_path))
    assert len(samples) == 1
    s = samples[0]
    assert s.payload["target.txt"] == "Hello world"
    assert s.payload["json"]["page_idx"] == 1
    assert s.key == "doc_p001_basic_read_0"


def test_empty_page_yields_no_samples(tmp_path):
    page = ET.Element("page", {"idx": "0"})
    samples = BasicReadingComposer().compose_page(page, _ctx([page], tmp_path))
    assert samples == []
