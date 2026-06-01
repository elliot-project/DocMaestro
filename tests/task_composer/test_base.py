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

    def test_key_mismatch_raises(self):
        with pytest.raises(ValueError, match="key mismatch"):
            Sample(key="foo", payload={"__key__": "bar", "target.txt": "x"})


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


from src.task_composer.base import BaseTaskComposer


class _ConcreteComposer(BaseTaskComposer):
    """Minimal concrete subclass that returns one sample per page."""

    task_name = "test_task"

    def compose_page(self, page: ET.Element, context: DocumentContext) -> list[Sample]:
        key = f"{context.doc_name}_p{page.attrib['idx']}_test_0"
        return [Sample(key=key, payload={"__key__": key, "target.txt": "y"})]


class TestBaseTaskComposer:
    def _make_root(self, n_pages: int = 3) -> ET.Element:
        root = ET.Element("document", {"name": "testdoc", "page_count": str(n_pages), "backend": "hybrid"})
        for i in range(n_pages):
            ET.SubElement(root, "page", {"idx": str(i), "width": "600", "height": "800"})
        return root

    def test_task_name_declared(self):
        assert _ConcreteComposer.task_name == "test_task"

    def test_cannot_instantiate_without_task_name(self):
        class _BadComposer(BaseTaskComposer):
            def compose_page(self, page, context):
                return []

        with pytest.raises(TypeError):
            _BadComposer()

    def test_compose_document_iterates_all_pages(self, tmp_path):
        root = self._make_root(n_pages=3)
        composer = _ConcreteComposer()
        samples = composer.compose_document(root, img_root=tmp_path)
        assert len(samples) == 3

    def test_compose_document_builds_context_correctly(self, tmp_path):
        root = self._make_root(n_pages=2)
        composer = _ConcreteComposer()
        samples = composer.compose_document(root, img_root=tmp_path)
        assert samples[0].key.startswith("testdoc_p0")
        assert samples[1].key.startswith("testdoc_p1")

    def test_compose_document_passes_img_root(self, tmp_path):
        """img_root must be reachable from context inside compose_page."""
        captured: list[Path] = []

        class _CapturingComposer(BaseTaskComposer):
            task_name = "capturing"

            def compose_page(self, page, context):
                captured.append(context.img_root)
                return []

        root = self._make_root(n_pages=1)
        _CapturingComposer().compose_document(root, img_root=tmp_path)
        assert captured[0] == tmp_path
