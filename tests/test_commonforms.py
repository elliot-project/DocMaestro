"""Tests for the CommonForms dataset preparation logic."""

from __future__ import annotations

import io

import pytest

from src.datasets.commonforms import (
    CATEGORY_NAMES,
    build_fields_sidecar,
    build_pdf,
    coco_to_xyxy,
    group_pages,
    parse_file_name,
)


def _jpeg_bytes(width: int, height: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(255, 255, 255)).save(buf, format="JPEG")
    return buf.getvalue()


class TestParseFileName:
    def test_basic_name_splits_into_root_and_page(self):
        assert parse_file_name("0005676-1.jpeg") == ("0005676", 1)

    def test_multi_dash_root_kept_intact(self):
        assert parse_file_name("00056-76-2.jpeg") == ("00056-76", 2)

    def test_double_digit_page(self):
        assert parse_file_name("0005676-12.jpeg") == ("0005676", 12)

    def test_missing_hyphen_raises(self):
        with pytest.raises(ValueError):
            parse_file_name("0005676.jpeg")

    def test_non_integer_page_raises(self):
        with pytest.raises(ValueError):
            parse_file_name("0005676-a.jpeg")


class TestCocoToXyxy:
    def test_converts_xywh_to_xyxy(self):
        assert coco_to_xyxy([10, 20, 30, 40]) == [10, 20, 40, 60]

    def test_preserves_float_coordinates(self):
        assert coco_to_xyxy([1.5, 2.5, 3.0, 4.0]) == [1.5, 2.5, 4.5, 6.5]


class TestGroupPages:
    def test_groups_by_root_and_sorts_by_page(self):
        records = [
            {"file_name": "doc_a-2.jpeg"},
            {"file_name": "doc_b-1.jpeg"},
            {"file_name": "doc_a-1.jpeg"},
            {"file_name": "doc_a-10.jpeg"},
        ]
        grouped = group_pages(records)
        assert set(grouped) == {"doc_a", "doc_b"}
        assert [r["file_name"] for r in grouped["doc_a"]] == [
            "doc_a-1.jpeg",
            "doc_a-2.jpeg",
            "doc_a-10.jpeg",
        ]
        assert [r["file_name"] for r in grouped["doc_b"]] == ["doc_b-1.jpeg"]


class TestBuildFieldsSidecar:
    def _page(self):
        return {
            "page": 1,
            "width": 1700,
            "height": 2200,
            "objects": {
                "bbox": [[10, 20, 30, 40], [5, 5, 10, 10]],
                "category_id": [0, 2],
                "area": [1200.0, 100.0],
                "id": [11, 12],
            },
        }

    def test_top_level_schema(self):
        sidecar = build_fields_sidecar("0005676", [self._page()])
        assert sidecar["doc_id"] == "0005676"
        assert sidecar["source"] == "commonforms"
        assert sidecar["bbox_format"] == "xyxy_pixels"
        assert sidecar["categories"] == CATEGORY_NAMES
        assert len(sidecar["pages"]) == 1

    def test_fields_converted_and_named(self):
        sidecar = build_fields_sidecar("0005676", [self._page()])
        page = sidecar["pages"][0]
        assert page["page"] == 1
        assert page["width"] == 1700
        assert page["height"] == 2200
        first, second = page["fields"]
        assert first == {
            "category_id": 0,
            "category": "text",
            "bbox": [10, 20, 40, 60],
            "area": 1200.0,
            "id": 11,
        }
        assert second["category"] == "signature"
        assert second["bbox"] == [5, 5, 15, 15]

    def test_page_with_no_objects_yields_empty_fields(self):
        empty_page = {
            "page": 1,
            "width": 100,
            "height": 100,
            "objects": {"bbox": [], "category_id": [], "area": [], "id": []},
        }
        sidecar = build_fields_sidecar("doc", [empty_page])
        assert sidecar["pages"][0]["fields"] == []


class TestBuildPdf:
    def test_page_count_order_and_dimensions(self):
        import fitz

        images = [
            (_jpeg_bytes(100, 200), 100, 200),
            (_jpeg_bytes(50, 75), 50, 75),
        ]
        pdf_bytes = build_pdf(images)

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            assert doc.page_count == 2
            # 1 pixel == 1 point, order preserved.
            assert (round(doc[0].rect.width), round(doc[0].rect.height)) == (100, 200)
            assert (round(doc[1].rect.width), round(doc[1].rect.height)) == (50, 75)
        finally:
            doc.close()

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            build_pdf([])
