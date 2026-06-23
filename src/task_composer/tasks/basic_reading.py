# src/task_composer/tasks/basic_reading.py
"""DocTask 1: basic reading (image -> full-page text)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from src.task_composer.base import BaseTaskComposer, DocumentContext, Sample
from src.task_composer.xml_ops import page_idx, page_reading_text


class BasicReadingComposer(BaseTaskComposer):
    """SSL task: extract all page text in reading order.

    Emits one sample per page whose target is the concatenated text of every
    text-bearing element in document order. Pages with no readable text are
    skipped.
    """

    task_name = "basic_reading"

    def compose_page(self, page: ET.Element, context: DocumentContext) -> list[Sample]:
        """Produce a single full-text sample for the page, or none if empty."""
        text = page_reading_text(page)
        if not text:
            return []
        idx = page_idx(page)
        key = f"{context.doc_name}_p{idx:03d}_basic_read_0"
        payload: dict[str, Any] = {
            "__key__": key,
            "target.txt": text,
            "json": {
                "task_type": self.task_name,
                "doc_name": context.doc_name,
                "page_idx": idx,
            },
        }
        return [Sample(key=key, payload=payload)]
