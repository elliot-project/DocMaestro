"""CommonForms dataset preparation: page grouping, PDF assembly, GT sidecars.

The CommonForms dataset stores one document page per row as a JPEG. Pages of the
same document share a root file name (``0005676-1.jpeg``, ``0005676-2.jpeg``, ...)
and carry COCO-style ground-truth form-field annotations. This module holds the
pure logic that turns those rows into per-document multi-page PDFs (consumable by
the existing MinerU pipeline) plus an additive GT sidecar.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

# Authoritative class labels from the dataset's ClassLabel feature
# (['Text', 'CheckBox', 'Signature']), normalized to lowercase.
CATEGORY_NAMES: dict[str, str] = {
    "0": "text",
    "1": "checkbox",
    "2": "signature",
}

_SOURCE = "commonforms"


def parse_file_name(file_name: str) -> tuple[str, int]:
    """Split a CommonForms page file name into its document root and page number.

    The document root is everything before the LAST hyphen; the page number is
    the integer after it. This keeps roots that themselves contain hyphens intact
    (``00056-76-2.jpeg`` -> root ``00056-76``, page ``2``).

    Args:
        file_name: Page file name, e.g. ``"0005676-1.jpeg"``.

    Returns:
        Tuple of (document_root, page_number).

    Raises:
        ValueError: If the name has no hyphen or the page segment is not an int.
    """
    stem = file_name.rsplit(".", 1)[0]
    if "-" not in stem:
        raise ValueError(f"No page hyphen in file name: {file_name!r}")
    root, _, page_part = stem.rpartition("-")
    try:
        page = int(page_part)
    except ValueError as exc:
        raise ValueError(f"Non-integer page in file name: {file_name!r}") from exc
    return root, page


def coco_to_xyxy(bbox: list[float]) -> list[float]:
    """Convert a COCO ``[x, y, w, h]`` bbox to ``[x1, y1, x2, y2]``.

    Args:
        bbox: COCO-format box ``[x, y, width, height]``.

    Returns:
        Box as ``[x1, y1, x2, y2]`` where ``x2 = x + w`` and ``y2 = y + h``.
    """
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def group_pages(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group page records by document root, each group sorted by page number.

    Each record must carry a ``"file_name"`` key (e.g. ``"0005676-1.jpeg"``).
    Records are grouped by the document root and ordered ascending by page number
    within each group.

    Args:
        records: Page records, each with at least a ``"file_name"`` key.

    Returns:
        Mapping of document root to its page records, sorted by page number.
    """
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for record in records:
        root, page = parse_file_name(record["file_name"])
        groups[root].append((page, record))
    return {
        root: [record for _, record in sorted(pages, key=lambda pr: pr[0])]
        for root, pages in groups.items()
    }


def build_fields_sidecar(doc_id: str, pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the additive ground-truth form-field sidecar for one document.

    Converts each page's COCO ``objects`` (parallel lists of ``bbox``,
    ``category_id``, ``area``, ``id``) into a list of field records with
    ``[x1, y1, x2, y2]`` pixel coordinates and human-readable category names. The
    coordinate space matches the materialized PDF (1 pixel = 1 point), so the
    sidecar can later be merged into the MinerU XML.

    Args:
        doc_id: Document root identifier.
        pages: Page dicts, each with ``page``, ``width``, ``height``, and an
            ``objects`` dict of parallel COCO lists.

    Returns:
        Sidecar dict with top-level metadata and a per-page ``fields`` list.
    """
    out_pages = []
    for page in pages:
        objects = page["objects"]
        fields = [
            {
                "category_id": category_id,
                "category": CATEGORY_NAMES.get(str(category_id), "unknown"),
                "bbox": coco_to_xyxy(bbox),
                "area": area,
                "id": field_id,
            }
            for bbox, category_id, area, field_id in zip(
                objects["bbox"],
                objects["category_id"],
                objects["area"],
                objects["id"],
            )
        ]
        out_pages.append(
            {
                "page": page["page"],
                "width": page["width"],
                "height": page["height"],
                "fields": fields,
            }
        )
    return {
        "doc_id": doc_id,
        "source": _SOURCE,
        "bbox_format": "xyxy_pixels",
        "categories": CATEGORY_NAMES,
        "pages": out_pages,
    }


def build_pdf(images: list[tuple[bytes, int, int]]) -> bytes:
    """Assemble image pages into a single multi-page PDF, lossless.

    Each page is created with an explicit rectangle equal to the image's pixel
    dimensions and the JPEG bytes are embedded directly (no rasterization or
    re-encoding). This makes 1 pixel equal 1 PDF point, so ground-truth pixel
    bounding boxes align with the rendered page. Page order matches input order.

    Args:
        images: Ordered list of ``(jpeg_bytes, width_px, height_px)`` tuples.

    Returns:
        The assembled PDF as bytes.

    Raises:
        ValueError: If ``images`` is empty.
    """
    if not images:
        raise ValueError("Cannot build a PDF from zero pages.")

    import fitz  # lazy — heavy native dep

    doc = fitz.open()
    try:
        for image_bytes, width, height in images:
            page = doc.new_page(width=width, height=height)
            page.insert_image(fitz.Rect(0, 0, width, height), stream=image_bytes)
        return doc.tobytes()
    finally:
        doc.close()
