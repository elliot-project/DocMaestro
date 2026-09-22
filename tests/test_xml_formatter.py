"""Unit tests for xml_formatter.py — document XML synthesis."""

import xml.etree.ElementTree as ET

from doc_maestro.xml_formatter import build_document_xml

MINIMAL_MIDDLE_JSON = {
    "_backend": "pipeline",
    "_version_name": "3.1.6",
    "pdf_info": [
        {
            "page_idx": 0,
            "page_size": [612.0, 792.0],
            "para_blocks": [],
            "images": [],
            "tables": [],
            "interline_equations": [],
            "discarded_blocks": [],
        }
    ],
}


def _parse_xml(xml_str: str) -> ET.Element:
    """Strip XML declaration line and parse to ElementTree."""
    lines = xml_str.strip().splitlines()
    body = "\n".join(line for line in lines if not line.strip().startswith("<?xml"))
    return ET.fromstring(body)


class TestDocumentRoot:
    def test_root_tag_is_document(self):
        root = _parse_xml(build_document_xml([], MINIMAL_MIDDLE_JSON, "test.pdf"))
        assert root.tag == "document"

    def test_document_name_attribute(self):
        root = _parse_xml(build_document_xml([], MINIMAL_MIDDLE_JSON, "test.pdf"))
        assert root.get("name") == "test.pdf"

    def test_document_backend_attribute(self):
        root = _parse_xml(build_document_xml([], MINIMAL_MIDDLE_JSON, "test.pdf"))
        assert root.get("backend") == "pipeline"

    def test_page_count_from_middle_json(self):
        root = _parse_xml(build_document_xml([], MINIMAL_MIDDLE_JSON, "test.pdf"))
        assert root.get("page_count") == "1"


class TestPageElement:
    def test_page_element_created(self):
        content = [{"type": "text", "text": "Hello", "bbox": [0, 0, 100, 20], "page_idx": 0}]
        root = _parse_xml(build_document_xml(content, MINIMAL_MIDDLE_JSON, "doc.pdf"))
        pages = root.findall("page")
        assert len(pages) == 1

    def test_page_idx_attribute(self):
        content = [{"type": "text", "text": "Hello", "bbox": [0, 0, 100, 20], "page_idx": 0}]
        root = _parse_xml(build_document_xml(content, MINIMAL_MIDDLE_JSON, "doc.pdf"))
        assert root.find("page").get("idx") == "0"

    def test_page_size_attributes(self):
        content = [{"type": "text", "text": "Hello", "bbox": [0, 0, 100, 20], "page_idx": 0}]
        root = _parse_xml(build_document_xml(content, MINIMAL_MIDDLE_JSON, "doc.pdf"))
        page = root.find("page")
        assert page.get("width") == "612.0"
        assert page.get("height") == "792.0"


class TestTextBlocks:
    def test_plain_text_tag(self):
        content = [{"type": "text", "text": "Body text.", "bbox": [0, 0, 100, 20], "page_idx": 0}]
        root = _parse_xml(build_document_xml(content, MINIMAL_MIDDLE_JSON, "doc.pdf"))
        el = root.find(".//text-block")
        assert el is not None
        assert el.text == "Body text."

    def test_text_block_bbox_attribute(self):
        content = [{"type": "text", "text": "X", "bbox": [10, 20, 200, 40], "page_idx": 0}]
        root = _parse_xml(build_document_xml(content, MINIMAL_MIDDLE_JSON, "doc.pdf"))
        el = root.find(".//text-block")
        assert el.get("bbox") == "[10, 20, 200, 40]"

    def test_title_level_1(self):
        content = [
            {
                "type": "text",
                "text": "Introduction",
                "text_level": 1,
                "bbox": [50, 100, 400, 130],
                "page_idx": 0,
            }
        ]
        root = _parse_xml(build_document_xml(content, MINIMAL_MIDDLE_JSON, "doc.pdf"))
        title = root.find(".//title")
        assert title is not None
        assert title.get("level") == "1"
        assert title.text == "Introduction"

    def test_title_level_2(self):
        content = [
            {"type": "text", "text": "Sub", "text_level": 2, "bbox": [0, 0, 100, 20], "page_idx": 0}
        ]
        root = _parse_xml(build_document_xml(content, MINIMAL_MIDDLE_JSON, "doc.pdf"))
        assert root.find(".//title").get("level") == "2"


class TestFigureElement:
    def test_figure_tag(self):
        content = [
            {
                "type": "image",
                "img_path": "images/abc.jpg",
                "image_caption": ["Fig. 1"],
                "image_footnote": [],
                "bbox": [62, 480, 946, 904],
                "page_idx": 0,
            }
        ]
        root = _parse_xml(build_document_xml(content, MINIMAL_MIDDLE_JSON, "doc.pdf"))
        figure = root.find(".//figure")
        assert figure is not None

    def test_figure_bbox(self):
        content = [
            {
                "type": "image",
                "img_path": "images/abc.jpg",
                "image_caption": [],
                "image_footnote": [],
                "bbox": [100, 150, 400, 350],
                "page_idx": 0,
            }
        ]
        root = _parse_xml(build_document_xml(content, MINIMAL_MIDDLE_JSON, "doc.pdf"))
        figure = root.find(".//figure")
        assert figure.get("bbox") == "[100, 150, 400, 350]"

    def test_figure_img_src(self):
        content = [
            {
                "type": "image",
                "img_path": "images/abc.jpg",
                "image_caption": [],
                "image_footnote": [],
                "bbox": [0, 0, 100, 100],
                "page_idx": 0,
            }
        ]
        root = _parse_xml(build_document_xml(content, MINIMAL_MIDDLE_JSON, "doc.pdf"))
        img = root.find(".//figure/img")
        assert img.get("src") == "images/abc.jpg"

    def test_figure_caption(self):
        content = [
            {
                "type": "image",
                "img_path": "images/abc.jpg",
                "image_caption": ["Fig. 1. Annual flow curves."],
                "image_footnote": [],
                "bbox": [0, 0, 100, 100],
                "page_idx": 0,
            }
        ]
        root = _parse_xml(build_document_xml(content, MINIMAL_MIDDLE_JSON, "doc.pdf"))
        caption = root.find(".//figure/caption")
        assert caption.text == "Fig. 1. Annual flow curves."


class TestTableElement:
    def test_table_tag(self):
        content = [
            {
                "type": "table",
                "img_path": "images/tbl.jpg",
                "table_caption": ["Table 1"],
                "table_footnote": [],
                "table_body": "<html><body><table></table></body></html>",
                "bbox": [200, 300, 700, 500],
                "page_idx": 0,
            }
        ]
        root = _parse_xml(build_document_xml(content, MINIMAL_MIDDLE_JSON, "doc.pdf"))
        table = root.find(".//table")
        assert table is not None

    def test_table_bbox_format(self):
        content = [
            {
                "type": "table",
                "img_path": "images/tbl.jpg",
                "table_caption": [],
                "table_footnote": [],
                "table_body": "",
                "bbox": [200, 300, 700, 500],
                "page_idx": 0,
            }
        ]
        root = _parse_xml(build_document_xml(content, MINIMAL_MIDDLE_JSON, "doc.pdf"))
        assert root.find(".//table").get("bbox") == "[200, 300, 700, 500]"

    def test_table_body_html(self):
        html = "<html><body><table><tr><td>A</td></tr></table></body></html>"
        content = [
            {
                "type": "table",
                "img_path": "",
                "table_caption": [],
                "table_footnote": [],
                "table_body": html,
                "bbox": [0, 0, 100, 100],
                "page_idx": 0,
            }
        ]
        root = _parse_xml(build_document_xml(content, MINIMAL_MIDDLE_JSON, "doc.pdf"))
        body = root.find(".//table/table-body")
        assert body.text == html


class TestMathBlock:
    def test_math_block_tag(self):
        content = [
            {
                "type": "equation",
                "text": "$$Q_{\\%} = f(P)$$",
                "img_path": "images/eq.jpg",
                "bbox": [50, 600, 800, 650],
                "page_idx": 0,
            }
        ]
        root = _parse_xml(build_document_xml(content, MINIMAL_MIDDLE_JSON, "doc.pdf"))
        eq = root.find(".//math-block")
        assert eq is not None

    def test_math_block_bbox(self):
        content = [
            {
                "type": "equation",
                "text": "$$x^2$$",
                "bbox": [50, 600, 800, 650],
                "page_idx": 0,
            }
        ]
        root = _parse_xml(build_document_xml(content, MINIMAL_MIDDLE_JSON, "doc.pdf"))
        assert root.find(".//math-block").get("bbox") == "[50, 600, 800, 650]"

    def test_math_block_latex_content(self):
        content = [
            {
                "type": "equation",
                "text": "$$E = mc^2$$",
                "bbox": [0, 0, 100, 50],
                "page_idx": 0,
            }
        ]
        root = _parse_xml(build_document_xml(content, MINIMAL_MIDDLE_JSON, "doc.pdf"))
        assert root.find(".//math-block").text == "$$E = mc^2$$"


class TestCodeBlock:
    def test_code_block_tag(self):
        content = [
            {
                "type": "code",
                "text": "def foo(): pass",
                "sub_type": "code",
                "bbox": [0, 0, 200, 50],
                "page_idx": 0,
            }
        ]
        root = _parse_xml(build_document_xml(content, MINIMAL_MIDDLE_JSON, "doc.pdf"))
        el = root.find(".//code-block")
        assert el is not None
        assert el.text == "def foo(): pass"

    def test_code_block_sub_type_attribute(self):
        content = [
            {
                "type": "code",
                "text": "algorithm step 1",
                "sub_type": "algorithm",
                "bbox": [0, 0, 200, 50],
                "page_idx": 0,
            }
        ]
        root = _parse_xml(build_document_xml(content, MINIMAL_MIDDLE_JSON, "doc.pdf"))
        el = root.find(".//code-block")
        assert el.get("sub-type") == "algorithm"


class TestMultiPage:
    def test_multiple_pages_created(self):
        middle = {
            "_backend": "pipeline",
            "pdf_info": [
                {"page_idx": 0, "page_size": [612.0, 792.0], "para_blocks": []},
                {"page_idx": 1, "page_size": [612.0, 792.0], "para_blocks": []},
            ],
        }
        content = [
            {"type": "text", "text": "Page 0", "bbox": [0, 0, 100, 20], "page_idx": 0},
            {"type": "text", "text": "Page 1", "bbox": [0, 0, 100, 20], "page_idx": 1},
        ]
        root = _parse_xml(build_document_xml(content, middle, "multi.pdf"))
        pages = root.findall("page")
        assert len(pages) == 2
        assert pages[0].get("idx") == "0"
        assert pages[1].get("idx") == "1"

    def test_content_routed_to_correct_page(self):
        middle = {
            "_backend": "pipeline",
            "pdf_info": [
                {"page_idx": 0, "page_size": [612.0, 792.0], "para_blocks": []},
                {"page_idx": 1, "page_size": [612.0, 792.0], "para_blocks": []},
            ],
        }
        content = [
            {"type": "text", "text": "First", "bbox": [0, 0, 100, 20], "page_idx": 0},
            {"type": "text", "text": "Second", "bbox": [0, 0, 100, 20], "page_idx": 1},
        ]
        root = _parse_xml(build_document_xml(content, middle, "multi.pdf"))
        page_0_texts = [el.text for el in root.find("page[@idx='0']").findall("text-block")]
        page_1_texts = [el.text for el in root.find("page[@idx='1']").findall("text-block")]
        assert page_0_texts == ["First"]
        assert page_1_texts == ["Second"]


class TestBboxEdgeCases:
    def test_missing_bbox_no_attribute(self):
        content = [{"type": "text", "text": "No bbox", "page_idx": 0}]
        root = _parse_xml(build_document_xml(content, MINIMAL_MIDDLE_JSON, "doc.pdf"))
        el = root.find(".//text-block")
        assert el.get("bbox") is None

    def test_float_bbox_truncated(self):
        content = [{"type": "text", "text": "X", "bbox": [10.7, 20.3, 200.9, 40.1], "page_idx": 0}]
        root = _parse_xml(build_document_xml(content, MINIMAL_MIDDLE_JSON, "doc.pdf"))
        el = root.find(".//text-block")
        assert el.get("bbox") == "[10, 20, 200, 40]"
