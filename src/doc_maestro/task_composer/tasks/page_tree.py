# src/task_composer/tasks/page_tree.py
"""DocTask 5: page image -> logical XML tree (bbox-free)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from doc_maestro.task_composer.base import BaseTaskComposer, DocumentContext, Sample
from doc_maestro.task_composer.xml_ops import page_idx, strip_bbox


class PageTreeComposer(BaseTaskComposer):
    """SSL task: produce the logical XML tree of a page without bboxes.

    One sample per page that has children; the target is the sanitized,
    bbox-free serialization of the page element.
    """

    task_name = "page_tree"

    def compose_page(self, page: ET.Element, context: DocumentContext) -> list[Sample]:
        """Produce one tree sample for the page, or none if it has no children."""
        if len(list(page)) == 0:
            return []
        idx = page_idx(page)
        key = f"{context.doc_name}_p{idx:03d}_page_tree_0"
        payload: dict[str, Any] = {
            "__key__": key,
            "target.txt": strip_bbox(page),
            "json": {
                "task_type": self.task_name,
                "doc_name": context.doc_name,
                "page_idx": idx,
            },
        }
        return [Sample(key=key, payload=payload)]
