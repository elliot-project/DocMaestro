# src/task_composer/tasks/layout.py
"""DocTask 3: layout (image -> class labels + bounding boxes)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from src.task_composer.base import BaseTaskComposer, DocumentContext, Sample
from src.task_composer.xml_ops import layout_elements, page_idx


class LayoutComposer(BaseTaskComposer):
    """SSL task: identify layout elements and their spatial locations.

    One sample per page; the target lists ``{tag, bbox}`` for every bbox-bearing
    element (figures and tables included), using the XML tag as the layout class.
    """

    task_name = "layout"

    def compose_page(self, page: ET.Element, context: DocumentContext) -> list[Sample]:
        """Produce one layout sample for the page, or none if it has no boxes."""
        elements = layout_elements(page)
        if not elements:
            return []
        idx = page_idx(page)
        key = f"{context.doc_name}_p{idx:03d}_layout_0"
        payload: dict[str, Any] = {
            "__key__": key,
            "target.json": elements,
            "json": {
                "task_type": self.task_name,
                "doc_name": context.doc_name,
                "page_idx": idx,
            },
        }
        return [Sample(key=key, payload=payload)]
