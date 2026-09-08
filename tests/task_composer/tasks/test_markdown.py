# tests/task_composer/tasks/test_markdown.py
"""Tests for MarkdownComposer."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from src.task_composer.base import DocumentContext
from src.task_composer.tasks.markdown import MarkdownComposer


def _ctx(pages, tmp_path: Path) -> DocumentContext:
    return DocumentContext(doc_name="doc", pages=pages, img_root=tmp_path)


def test_task_name():
    assert MarkdownComposer.task_name == "markdown"


def test_title_becomes_heading(tmp_path):
    page = ET.Element("page", {"idx": "0"})
    ET.SubElement(page, "title", {"level": "2"}).text = "Methods"
    samples = MarkdownComposer().compose_page(page, _ctx([page], tmp_path))
    assert "## Methods" in samples[0].payload["target.txt"]


def test_empty_page_yields_no_samples(tmp_path):
    page = ET.Element("page", {"idx": "0"})
    assert MarkdownComposer().compose_page(page, _ctx([page], tmp_path)) == []
