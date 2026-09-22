# src/task_composer/tasks/ocr.py
"""DocTask 2: OCR (image -> text + bounding boxes, block level)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from doc_maestro.task_composer.base import BaseTaskComposer, DocumentContext, Sample
from doc_maestro.task_composer.xml_ops import page_idx, text_blocks_with_bbox


class OcrComposer(BaseTaskComposer):
    """SSL task: extract text and its block-level bounding box per element.

    One sample per page; the target is the ordered list of ``{bbox, tag, text}``
    for every text-bearing block. Word-level boxes are future work (see
    ``docs/DocTasks.md`` Task 2 note).
    """

    task_name = "ocr"

    def compose_page(self, page: ET.Element, context: DocumentContext) -> list[Sample]:
        """Produce one OCR sample for the page, or none if it has no text."""
        blocks = text_blocks_with_bbox(page)
        if not blocks:
            return []
        idx = page_idx(page)
        key = f"{context.doc_name}_p{idx:03d}_ocr_0"
        payload: dict[str, Any] = {
            "__key__": key,
            "target.json": blocks,
            "json": {
                "task_type": self.task_name,
                "doc_name": context.doc_name,
                "page_idx": idx,
            },
        }
        return [Sample(key=key, payload=payload)]
