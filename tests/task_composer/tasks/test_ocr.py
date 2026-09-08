# tests/task_composer/tasks/test_ocr.py
"""Tests for OcrComposer."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from src.task_composer.base import DocumentContext
from src.task_composer.tasks.ocr import OcrComposer


def _ctx(pages, tmp_path: Path) -> DocumentContext:
    return DocumentContext(doc_name="doc", pages=pages, img_root=tmp_path)


def test_task_name():
    assert OcrComposer.task_name == "ocr"


def test_emits_blocks_with_bbox(tmp_path):
    page = ET.Element("page", {"idx": "0"})
    tb = ET.SubElement(page, "text-block", {"bbox": "[1, 2, 3, 4]"})
    tb.text = "abc"
    samples = OcrComposer().compose_page(page, _ctx([page], tmp_path))
    assert len(samples) == 1
    blocks = samples[0].payload["target.json"]
    assert blocks == [{"bbox": "[1, 2, 3, 4]", "tag": "text-block", "text": "abc"}]


def test_no_text_blocks_yields_no_samples(tmp_path):
    page = ET.Element("page", {"idx": "0"})
    ET.SubElement(page, "figure", {"bbox": "[0, 0, 1, 1]"})
    assert OcrComposer().compose_page(page, _ctx([page], tmp_path)) == []
