"""Tests for ReadingOrderComposer."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from src.task_composer.base import DocumentContext
from src.task_composer.tasks.reading_order import ReadingOrderComposer


def _make_page(idx: int = 0, n_text_blocks: int = 5) -> ET.Element:
    page = ET.Element("page", {"idx": str(idx), "width": "600", "height": "800"})
    for i in range(n_text_blocks):
        tb = ET.SubElement(page, "text-block", {"bbox": f"[0,{i*30},{600},{(i+1)*30}]"})
        tb.text = f"paragraph {i}"
    return page


def _make_context(pages: list[ET.Element], tmp_path: Path) -> DocumentContext:
    return DocumentContext(doc_name="testdoc", pages=pages, img_root=tmp_path)


class TestTaskName:
    def test_task_name(self):
        assert ReadingOrderComposer.task_name == "reading_order"


class TestNoSamples:
    def test_page_with_fewer_than_n_elements_returns_empty(self, tmp_path):
        page = _make_page(n_text_blocks=4)
        ctx = _make_context([page], tmp_path)
        samples = ReadingOrderComposer(n_elements=5, seed=0).compose_page(page, ctx)
        assert samples == []

    def test_page_headers_footers_excluded_from_eligible(self, tmp_path):
        page = ET.Element("page", {"idx": "0", "width": "600", "height": "800"})
        for i in range(3):
            tb = ET.SubElement(page, "text-block")
            tb.text = f"text {i}"
        header = ET.SubElement(page, "page-header")
        header.text = "header"
        footer = ET.SubElement(page, "page-footer")
        footer.text = "footer"
        ctx = _make_context([page], tmp_path)
        # Only 3 eligible elements — need 5
        samples = ReadingOrderComposer(n_elements=5, seed=0).compose_page(page, ctx)
        assert samples == []

    def test_page_number_and_abandon_excluded_from_eligible(self, tmp_path):
        page = ET.Element("page", {"idx": "0", "width": "600", "height": "800"})
        for i in range(3):
            tb = ET.SubElement(page, "text-block")
            tb.text = f"text {i}"
        pn = ET.SubElement(page, "page-number")
        pn.text = "1"
        ab = ET.SubElement(page, "abandon")
        ab.text = "watermark"
        ctx = _make_context([page], tmp_path)
        # Only 3 eligible elements — page-number and abandon are not counted
        samples = ReadingOrderComposer(n_elements=5, seed=0).compose_page(page, ctx)
        assert samples == []


class TestSampleStructure:
    def test_one_sample_per_eligible_page(self, tmp_path):
        page = _make_page(n_text_blocks=5)
        ctx = _make_context([page], tmp_path)
        samples = ReadingOrderComposer(n_elements=5, seed=0).compose_page(page, ctx)
        assert len(samples) == 1

    def test_input_has_n_elements(self, tmp_path):
        page = _make_page(n_text_blocks=8)
        ctx = _make_context([page], tmp_path)
        samples = ReadingOrderComposer(n_elements=5, seed=0).compose_page(page, ctx)
        assert len(samples[0].payload["input.json"]) == 5

    def test_target_has_n_elements(self, tmp_path):
        page = _make_page(n_text_blocks=5)
        ctx = _make_context([page], tmp_path)
        samples = ReadingOrderComposer(n_elements=5, seed=0).compose_page(page, ctx)
        assert len(samples[0].payload["target.json"]) == 5

    def test_target_is_valid_permutation(self, tmp_path):
        page = _make_page(n_text_blocks=5)
        ctx = _make_context([page], tmp_path)
        samples = ReadingOrderComposer(n_elements=5, seed=0).compose_page(page, ctx)
        target = samples[0].payload["target.json"]
        assert sorted(target) == list(range(5))

    def test_target_restores_correct_order(self, tmp_path):
        """Applying target indices to shuffled input must yield elements in
        their original document order (sorted by original position)."""
        page = _make_page(n_text_blocks=7)
        ctx = _make_context([page], tmp_path)
        samples = ReadingOrderComposer(n_elements=5, seed=42).compose_page(page, ctx)
        shuffled = samples[0].payload["input.json"]
        target = samples[0].payload["target.json"]
        restored = [shuffled[i] for i in target]
        # The restored texts must be in ascending paragraph order
        texts = [item["text"] for item in restored]
        assert texts == sorted(texts)

    def test_input_items_have_required_fields(self, tmp_path):
        page = _make_page(n_text_blocks=5)
        ctx = _make_context([page], tmp_path)
        samples = ReadingOrderComposer(n_elements=5, seed=0).compose_page(page, ctx)
        for item in samples[0].payload["input.json"]:
            assert "idx" in item
            assert "tag" in item
            assert "text" in item

    def test_idx_field_matches_shuffled_position(self, tmp_path):
        page = _make_page(n_text_blocks=5)
        ctx = _make_context([page], tmp_path)
        samples = ReadingOrderComposer(n_elements=5, seed=0).compose_page(page, ctx)
        for pos, item in enumerate(samples[0].payload["input.json"]):
            assert item["idx"] == pos

    def test_key_format(self, tmp_path):
        page = _make_page(idx=3, n_text_blocks=5)
        ctx = _make_context([page], tmp_path)
        samples = ReadingOrderComposer(n_elements=5, seed=0).compose_page(page, ctx)
        assert samples[0].key == "testdoc_p003_read_ord_0"

    def test_json_metadata_fields(self, tmp_path):
        page = _make_page(idx=1, n_text_blocks=5)
        ctx = _make_context([page], tmp_path)
        samples = ReadingOrderComposer(n_elements=5, seed=0).compose_page(page, ctx)
        meta = samples[0].payload["json"]
        assert meta["task_type"] == "reading_order"
        assert meta["doc_name"] == "testdoc"
        assert meta["page_idx"] == 1
        assert meta["n_elements"] == 5

    def test_payload_key_matches_sample_key(self, tmp_path):
        page = _make_page(n_text_blocks=5)
        ctx = _make_context([page], tmp_path)
        samples = ReadingOrderComposer(n_elements=5, seed=0).compose_page(page, ctx)
        assert samples[0].payload["__key__"] == samples[0].key


class TestDeterminism:
    def test_same_seed_produces_same_shuffle(self, tmp_path):
        page = _make_page(n_text_blocks=7)
        ctx = _make_context([page], tmp_path)
        s1 = ReadingOrderComposer(n_elements=5, seed=99).compose_page(page, ctx)
        s2 = ReadingOrderComposer(n_elements=5, seed=99).compose_page(page, ctx)
        assert s1[0].payload["target.json"] == s2[0].payload["target.json"]
