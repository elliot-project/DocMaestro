# src/task_composer/tasks/markdown.py
"""DocTask 4: image -> semantic markdown."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from src.task_composer.base import BaseTaskComposer, DocumentContext, Sample
from src.task_composer.xml_ops import page_idx, xml_to_markdown


class MarkdownComposer(BaseTaskComposer):
    """SSL task: render a page as semantic markdown.

    One sample per page; the target is the markdown produced by
    ``xml_ops.xml_to_markdown`` (headings, figures, raw table HTML, math, code).
    """

    task_name = "markdown"

    def compose_page(self, page: ET.Element, context: DocumentContext) -> list[Sample]:
        """Produce one markdown sample for the page, or none if empty."""
        md = xml_to_markdown(page)
        if not md:
            return []
        idx = page_idx(page)
        key = f"{context.doc_name}_p{idx:03d}_markdown_0"
        payload: dict[str, Any] = {
            "__key__": key,
            "target.txt": md,
            "json": {
                "task_type": self.task_name,
                "doc_name": context.doc_name,
                "page_idx": idx,
            },
        }
        return [Sample(key=key, payload=payload)]
