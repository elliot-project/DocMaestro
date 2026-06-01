"""Tests for CaptionPredictionComposer."""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from src.task_composer.base import DocumentContext
from src.task_composer.tasks.caption_prediction import CaptionPredictionComposer


def _make_page(idx: int = 0, elements: list[ET.Element] | None = None) -> ET.Element:
    page = ET.Element("page", {"idx": str(idx), "width": "600", "height": "800"})
    for el in (elements or []):
        page.append(el)
    return page


def _text_block(text: str, bbox: str = "[0, 0, 100, 20]") -> ET.Element:
    el = ET.Element("text-block", {"bbox": bbox})
    el.text = text
    return el


def _title(text: str) -> ET.Element:
    el = ET.Element("title", {"bbox": "[0, 0, 200, 30]", "level": "1"})
    el.text = text
    return el


def _figure_with_caption(
    caption: str,
    img_src: str | None = "images/fig.jpg",
    bbox: str = "[100, 100, 400, 300]",
) -> ET.Element:
    fig = ET.Element("figure", {"bbox": bbox})
    if img_src:
        img = ET.SubElement(fig, "img")
        img.set("src", img_src)
    cap = ET.SubElement(fig, "caption")
    cap.text = caption
    return fig


def _chart_with_caption(caption: str) -> ET.Element:
    chart = ET.Element("chart", {"bbox": "[100, 100, 400, 300]"})
    img = ET.SubElement(chart, "img")
    img.set("src", "images/chart.jpg")
    cap = ET.SubElement(chart, "caption")
    cap.text = caption
    return chart


def _table_with_caption(caption: str) -> ET.Element:
    table = ET.Element("table", {"bbox": "[100, 100, 400, 300]"})
    cap = ET.SubElement(table, "caption")
    cap.text = caption
    return table


def _make_context(pages: list[ET.Element], tmp_path: Path) -> DocumentContext:
    return DocumentContext(doc_name="testdoc", pages=pages, img_root=tmp_path)


def _write_dummy_image(tmp_path: Path, rel_path: str) -> Path:
    p = tmp_path / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(128, 64, 32)).save(p)
    return p


class TestTaskName:
    def test_task_name(self):
        assert CaptionPredictionComposer.task_name == "caption_prediction"


class TestNoSamples:
    def test_page_with_no_captionable_elements_returns_empty(self, tmp_path):
        page = _make_page(elements=[_text_block("hello"), _title("Section")])
        ctx = _make_context([page], tmp_path)
        samples = CaptionPredictionComposer().compose_page(page, ctx)
        assert samples == []

    def test_figure_without_caption_returns_empty(self, tmp_path):
        fig = ET.Element("figure", {"bbox": "[0,0,100,100]"})
        img = ET.SubElement(fig, "img")
        img.set("src", "images/fig.jpg")
        page = _make_page(elements=[fig])
        ctx = _make_context([page], tmp_path)
        samples = CaptionPredictionComposer().compose_page(page, ctx)
        assert samples == []

    def test_figure_with_empty_caption_returns_empty(self, tmp_path):
        fig = _figure_with_caption(caption="  ")
        page = _make_page(elements=[fig])
        ctx = _make_context([page], tmp_path)
        samples = CaptionPredictionComposer().compose_page(page, ctx)
        assert samples == []


class TestSampleStructure:
    def test_one_sample_per_captioned_figure(self, tmp_path):
        page = _make_page(elements=[_figure_with_caption("Fig 1. Results")])
        ctx = _make_context([page], tmp_path)
        samples = CaptionPredictionComposer().compose_page(page, ctx)
        assert len(samples) == 1

    def test_three_captionable_types_all_produce_samples(self, tmp_path):
        page = _make_page(elements=[
            _figure_with_caption("Fig 1"),
            _chart_with_caption("Chart 2"),
            _table_with_caption("Table 3"),
        ])
        ctx = _make_context([page], tmp_path)
        samples = CaptionPredictionComposer().compose_page(page, ctx)
        assert len(samples) == 3

    def test_target_is_caption_text(self, tmp_path):
        page = _make_page(elements=[_figure_with_caption("Revenue chart Q4")])
        ctx = _make_context([page], tmp_path)
        samples = CaptionPredictionComposer().compose_page(page, ctx)
        assert samples[0].payload["target.txt"] == "Revenue chart Q4"

    def test_key_format(self, tmp_path):
        page = _make_page(idx=5, elements=[_figure_with_caption("cap")])
        ctx = _make_context([page], tmp_path)
        samples = CaptionPredictionComposer().compose_page(page, ctx)
        assert samples[0].key == "testdoc_p005_caption_pred_0"

    def test_json_metadata_fields(self, tmp_path):
        page = _make_page(idx=2, elements=[_figure_with_caption("cap")])
        ctx = _make_context([page], tmp_path)
        samples = CaptionPredictionComposer().compose_page(page, ctx)
        meta = samples[0].payload["json"]
        assert meta["task_type"] == "caption_prediction"
        assert meta["doc_name"] == "testdoc"
        assert meta["page_idx"] == 2
        assert meta["element_tag"] == "figure"

    def test_payload_key_matches_sample_key(self, tmp_path):
        page = _make_page(elements=[_figure_with_caption("cap")])
        ctx = _make_context([page], tmp_path)
        samples = CaptionPredictionComposer().compose_page(page, ctx)
        assert samples[0].payload["__key__"] == samples[0].key


class TestContextWindow:
    def test_preceding_up_to_3_elements_closest_first(self, tmp_path):
        elements = [
            _text_block("A"),  # idx 0
            _text_block("B"),  # idx 1
            _text_block("C"),  # idx 2
            _figure_with_caption("cap"),  # idx 3
            _text_block("D"),  # idx 4
        ]
        page = _make_page(elements=elements)
        ctx = _make_context([page], tmp_path)
        samples = CaptionPredictionComposer().compose_page(page, ctx)
        preceding = samples[0].payload["input.json"]["preceding"]
        assert len(preceding) == 3
        # Closest-first: C, B, A
        assert preceding[0]["text"] == "C"
        assert preceding[1]["text"] == "B"
        assert preceding[2]["text"] == "A"

    def test_following_up_to_3_elements_closest_first(self, tmp_path):
        elements = [
            _figure_with_caption("cap"),  # idx 0
            _text_block("A"),             # idx 1
            _text_block("B"),             # idx 2
            _text_block("C"),             # idx 3
            _text_block("D"),             # idx 4
        ]
        page = _make_page(elements=elements)
        ctx = _make_context([page], tmp_path)
        samples = CaptionPredictionComposer().compose_page(page, ctx)
        following = samples[0].payload["input.json"]["following"]
        assert len(following) == 3
        # Closest-first: A, B, C
        assert following[0]["text"] == "A"
        assert following[1]["text"] == "B"
        assert following[2]["text"] == "C"

    def test_fewer_than_3_preceding_returns_all_available(self, tmp_path):
        elements = [_text_block("only"), _figure_with_caption("cap")]
        page = _make_page(elements=elements)
        ctx = _make_context([page], tmp_path)
        samples = CaptionPredictionComposer().compose_page(page, ctx)
        assert len(samples[0].payload["input.json"]["preceding"]) == 1

    def test_page_header_excluded_from_context(self, tmp_path):
        header = ET.Element("page-header", {"bbox": "[0,0,600,30]"})
        header.text = "Annual Report"
        elements = [header, _text_block("real context"), _figure_with_caption("cap")]
        page = _make_page(elements=elements)
        ctx = _make_context([page], tmp_path)
        samples = CaptionPredictionComposer().compose_page(page, ctx)
        preceding = samples[0].payload["input.json"]["preceding"]
        assert all(e["tag"] != "page-header" for e in preceding)


class TestImageHandling:
    def test_image_included_when_file_exists(self, tmp_path):
        _write_dummy_image(tmp_path, "images/fig.jpg")
        page = _make_page(elements=[_figure_with_caption("cap", img_src="images/fig.jpg")])
        ctx = _make_context([page], tmp_path)
        samples = CaptionPredictionComposer().compose_page(page, ctx)
        assert "image.jpg" in samples[0].payload
        assert isinstance(samples[0].payload["image.jpg"], bytes)

    def test_image_omitted_when_file_missing_and_warning_emitted(self, tmp_path):
        page = _make_page(elements=[_figure_with_caption("cap", img_src="images/missing.jpg")])
        ctx = _make_context([page], tmp_path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            samples = CaptionPredictionComposer().compose_page(page, ctx)
        assert "image.jpg" not in samples[0].payload
        assert len(caught) == 1
        assert "missing.jpg" in str(caught[0].message)

    def test_sample_still_written_when_no_img_child(self, tmp_path):
        fig = ET.Element("figure", {"bbox": "[0,0,100,100]"})
        cap = ET.SubElement(fig, "caption")
        cap.text = "No image figure"
        page = _make_page(elements=[fig])
        ctx = _make_context([page], tmp_path)
        samples = CaptionPredictionComposer().compose_page(page, ctx)
        assert len(samples) == 1
        assert "image.jpg" not in samples[0].payload

    def test_image_omitted_when_img_src_is_empty(self, tmp_path):
        fig = ET.Element("figure", {"bbox": "[0,0,100,100]"})
        img = ET.SubElement(fig, "img")  # <img/> with no src attribute
        cap = ET.SubElement(fig, "caption")
        cap.text = "empty src figure"
        page = _make_page(elements=[fig])
        ctx = _make_context([page], tmp_path)
        # Should not raise IsADirectoryError — empty src is silently skipped
        samples = CaptionPredictionComposer().compose_page(page, ctx)
        assert len(samples) == 1
        assert "image.jpg" not in samples[0].payload
