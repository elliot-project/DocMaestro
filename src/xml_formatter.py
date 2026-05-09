"""XML synthesis from MinerU content_list.json and middle.json outputs."""

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from xml.dom import minidom

from src.xml_schema import format_bbox, resolve_tag

# Matches characters that are illegal in XML 1.0 (null bytes, C0 controls except
# tab/LF/CR, and DEL).  These sometimes appear in MinerU's OCR output.
_INVALID_XML_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def _clean(text: str) -> str:
    """Strip XML-illegal control characters from a string."""
    return _INVALID_XML_RE.sub("", text) if text else text


def _set_bbox_attr(element: ET.Element, bbox: list | None) -> None:
    """Attach a bbox attribute to an element if bbox is present."""
    formatted = format_bbox(bbox)
    if formatted is not None:
        element.set("bbox", formatted)


def _build_figure_element(item: dict) -> ET.Element:
    """Build a <figure> element from an image content block.

    Args:
        item: A content_list.json entry with type='image'.

    Returns:
        ET.Element representing the figure.
    """
    el = ET.Element("figure")
    _set_bbox_attr(el, item.get("bbox"))

    if img_path := item.get("img_path"):
        img_el = ET.SubElement(el, "img")
        img_el.set("src", img_path)

    for caption_text in item.get("image_caption", []):
        cap_el = ET.SubElement(el, "caption")
        cap_el.text = _clean(caption_text)

    for footnote_text in item.get("image_footnote", []):
        fn_el = ET.SubElement(el, "footnote")
        fn_el.text = _clean(footnote_text)

    return el


def _build_chart_element(item: dict) -> ET.Element:
    """Build a <chart> element from a chart content block.

    Args:
        item: A content_list.json entry with type='chart'.

    Returns:
        ET.Element representing the chart.
    """
    el = ET.Element("chart")
    _set_bbox_attr(el, item.get("bbox"))

    if img_path := item.get("img_path"):
        img_el = ET.SubElement(el, "img")
        img_el.set("src", img_path)

    for caption_text in item.get("chart_caption", []):
        cap_el = ET.SubElement(el, "caption")
        cap_el.text = _clean(caption_text)

    return el


def _build_table_element(item: dict) -> ET.Element:
    """Build a <table> element from a table content block.

    The table_body is HTML from MinerU and is stored inside a <table-body> child.

    Args:
        item: A content_list.json entry with type='table'.

    Returns:
        ET.Element representing the table.
    """
    el = ET.Element("table")
    _set_bbox_attr(el, item.get("bbox"))

    if img_path := item.get("img_path"):
        img_el = ET.SubElement(el, "img")
        img_el.set("src", img_path)

    for caption_text in item.get("table_caption", []):
        cap_el = ET.SubElement(el, "caption")
        cap_el.text = _clean(caption_text)

    if table_body := item.get("table_body"):
        body_el = ET.SubElement(el, "table-body")
        body_el.text = _clean(table_body)

    for footnote_text in item.get("table_footnote", []):
        fn_el = ET.SubElement(el, "footnote")
        fn_el.text = _clean(footnote_text)

    return el


def _build_equation_element(item: dict) -> ET.Element:
    """Build a <math-block> element from an equation content block.

    Args:
        item: A content_list.json entry with type='equation'.

    Returns:
        ET.Element representing the math block.
    """
    el = ET.Element("math-block")
    _set_bbox_attr(el, item.get("bbox"))
    el.text = _clean(item.get("text", ""))

    if img_path := item.get("img_path"):
        img_el = ET.SubElement(el, "img")
        img_el.set("src", img_path)

    return el


def _build_code_element(item: dict) -> ET.Element:
    """Build a <code-block> element from a code content block.

    Args:
        item: A content_list.json entry with type='code'.

    Returns:
        ET.Element representing the code block.
    """
    el = ET.Element("code-block")
    _set_bbox_attr(el, item.get("bbox"))
    if sub_type := item.get("sub_type"):
        el.set("sub-type", sub_type)
    el.text = _clean(item.get("text", ""))
    return el


def _build_generic_text_element(item: dict) -> ET.Element:
    """Build a text-bearing element (text-block, title, list, etc.).

    Args:
        item: A content_list.json entry with a text-bearing type.

    Returns:
        ET.Element with tag resolved from content type and text_level.
    """
    content_type = item.get("type", "text")
    text_level = item.get("text_level", 0) or 0
    tag_config = resolve_tag(content_type, text_level)

    el = ET.Element(tag_config.tag)
    for attr_name, attr_value in tag_config.extra_attrs.items():
        el.set(attr_name, attr_value)
    _set_bbox_attr(el, item.get("bbox"))
    el.text = _clean(item.get("text", ""))
    return el


_ELEMENT_BUILDERS = {
    "image": _build_figure_element,
    "chart": _build_chart_element,
    "table": _build_table_element,
    "equation": _build_equation_element,
    "code": _build_code_element,
}


def _build_element_for_item(item: dict) -> ET.Element:
    """Dispatch to the correct element builder for a content_list item.

    Args:
        item: A single entry from content_list.json.

    Returns:
        ET.Element for the content block.
    """
    content_type = item.get("type", "text")
    builder = _ELEMENT_BUILDERS.get(content_type)
    if builder:
        return builder(item)
    return _build_generic_text_element(item)


def build_document_xml(
    content_list: list[dict],
    middle_json: dict,
    doc_name: str,
) -> str:
    """Synthesize a structured XML document from MinerU extraction outputs.

    Combines the flat reading-order content from content_list.json with page
    geometry from middle.json to produce a hierarchical document XML that
    embeds bounding boxes on every content element.

    The coordinate system follows middle.json conventions: [x0, y0, x1, y1]
    in page-pixel space for pipeline/hybrid backends, or in 0-1000 normalized
    space for content_list.json (which MinerU normalizes to 0-1000).

    Args:
        content_list: Parsed list from content_list.json (flat, reading-order).
        middle_json: Parsed dict from middle.json (topology + page geometry).
        doc_name: Source document name used for the root element attribute.

    Returns:
        Pretty-printed XML string.
    """
    pdf_info: list[dict] = middle_json.get("pdf_info", [])
    page_sizes: dict[int, tuple[float, float]] = {}
    for page in pdf_info:
        idx = page.get("page_idx", 0)
        size = page.get("page_size", [0, 0])
        page_sizes[idx] = (float(size[0]), float(size[1]))

    # Group content blocks by page
    pages: dict[int, list[dict]] = defaultdict(list)
    for item in content_list:
        page_idx = item.get("page_idx", 0)
        pages[page_idx].append(item)

    root = ET.Element("document")
    root.set("name", doc_name)
    root.set("page_count", str(len(page_sizes) or len(pages)))
    root.set("backend", middle_json.get("_backend", "unknown"))

    for page_idx in sorted(pages.keys()):
        page_el = ET.SubElement(root, "page")
        page_el.set("idx", str(page_idx))
        size = page_sizes.get(page_idx)
        if size and size[0]:
            page_el.set("width", str(size[0]))
            page_el.set("height", str(size[1]))

        for item in pages[page_idx]:
            child_el = _build_element_for_item(item)
            page_el.append(child_el)

    xml_str = ET.tostring(root, encoding="unicode")
    return minidom.parseString(xml_str).toprettyxml(indent="  ", encoding=None)


def save_document_xml(xml_str: str, output_path: Path) -> None:
    """Write the XML string to disk.

    Args:
        xml_str: Pretty-printed XML document string.
        output_path: Destination file path (will be created with parents).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xml_str, encoding="utf-8")
