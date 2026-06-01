"""Tests for Sample and DocumentContext data types."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from src.task_composer.base import DocumentContext, Sample


class TestSample:
    def test_valid_sample_construction(self):
        s = Sample(key="doc_p001_task_0", payload={"__key__": "doc_p001_task_0", "target.txt": "hello"})
        assert s.key == "doc_p001_task_0"
        assert s.payload["target.txt"] == "hello"

    def test_missing_key_in_payload_raises(self):
        with pytest.raises(ValueError, match="missing '__key__'"):
            Sample(key="doc_p001_task_0", payload={"target.txt": "hello"})

    def test_payload_accepts_bytes(self):
        s = Sample(key="k", payload={"__key__": "k", "image.jpg": b"\xff\xd8\xff"})
        assert isinstance(s.payload["image.jpg"], bytes)


class TestDocumentContext:
    def test_construction(self, tmp_path):
        page = ET.Element("page", {"idx": "0", "width": "600", "height": "800"})
        ctx = DocumentContext(doc_name="test_doc", pages=[page], img_root=tmp_path)
        assert ctx.doc_name == "test_doc"
        assert len(ctx.pages) == 1
        assert ctx.img_root == tmp_path

    def test_pages_preserves_order(self):
        pages = [ET.Element("page", {"idx": str(i)}) for i in range(5)]
        ctx = DocumentContext(doc_name="d", pages=pages, img_root=Path("/tmp"))
        for i, p in enumerate(ctx.pages):
            assert p.attrib["idx"] == str(i)
