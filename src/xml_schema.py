"""XML tag schema for MinerU content-type to XML tag mapping."""

from dataclasses import dataclass

# Maps content_list.json `type` values to XML element tag names.
CONTENT_TYPE_TO_TAG: dict[str, str] = {
    "text": "text-block",
    "image": "figure",
    "table": "table",
    "chart": "chart",
    "equation": "math-block",
    "code": "code-block",
    "list": "list",
    "header": "page-header",
    "footer": "page-footer",
    "page_number": "page-number",
    "aside_text": "aside",
    "page_footnote": "page-footnote",
    "seal": "seal",
    "index": "index",
}

# Text nodes with text_level >= 1 are mapped to title tags instead.
TITLE_TAG = "title"
FALLBACK_TAG = "block"


@dataclass(frozen=True)
class XMLTagConfig:
    """Resolved XML tag and attributes for a single content block.

    Args:
        tag: XML element tag name.
        extra_attrs: Additional XML attributes (e.g., level for titles).
    """

    tag: str
    extra_attrs: dict[str, str]


def resolve_tag(content_type: str, text_level: int = 0) -> XMLTagConfig:
    """Resolve the XML tag and attributes for a MinerU content block.

    Args:
        content_type: The `type` field from content_list.json.
        text_level: The `text_level` field (0 = body text, 1+ = heading level).

    Returns:
        XMLTagConfig with the resolved tag and any extra attributes.
    """
    if content_type == "text" and text_level >= 1:
        return XMLTagConfig(tag=TITLE_TAG, extra_attrs={"level": str(text_level)})

    tag = CONTENT_TYPE_TO_TAG.get(content_type, FALLBACK_TAG)
    return XMLTagConfig(tag=tag, extra_attrs={})


def format_bbox(bbox: list[float] | list[int] | None) -> str | None:
    """Format a bounding box list as the canonical '[x1, y1, x2, y2]' string.

    Args:
        bbox: List of four numeric coordinates [x0, y0, x1, y1], or None.

    Returns:
        Formatted bbox string, or None if bbox is absent.
    """
    if not bbox or len(bbox) != 4:
        return None
    coords = [int(v) for v in bbox]
    return f"[{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}]"
