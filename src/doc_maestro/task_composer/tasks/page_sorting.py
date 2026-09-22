# src/task_composer/tasks/page_sorting.py
"""DocTask 8: page sorting (recover reading order of shuffled pages)."""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from doc_maestro.task_composer.base import BaseTaskComposer, DocumentContext, Sample
from doc_maestro.task_composer.xml_ops import page_idx


class PageSortingComposer(BaseTaskComposer):
    """SSL task: restore the original order of shuffled document pages.

    Produces one sample per document. Each page carries its page-number bboxes
    so the visualizer (and training pipeline) can black them out and prevent the
    model from trivializing the task. The target is the permutation that
    restores the original page order, matching ``reading_order`` semantics:
    ``[shuffled[target[i]] for i in range(n)]`` is the correctly ordered list.

    Args:
        seed: RNG seed for reproducible shuffling.
    """

    task_name = "page_sorting"

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self._rng = random.Random(seed)

    def compose_page(self, page: ET.Element, context: DocumentContext) -> list[Sample]:
        """Unused — page sorting is a document-level task."""
        return []

    def compose_document(self, root: ET.Element, img_root: Path) -> list[Sample]:
        """Produce a single page-ordering sample for the document."""
        pages = list(root)
        if len(pages) < 3:
            return []

        # Sample a random continuous subset of 3 to 5 pages
        n_sample = self._rng.randint(3, min(5, len(pages)))
        start_idx = self._rng.randint(0, len(pages) - n_sample)
        pages = pages[start_idx : start_idx + n_sample]

        doc_name = root.attrib["name"]
        ordered = [
            {
                "orig_page_idx": page_idx(p),
                "page_number_bboxes": [
                    e.attrib["bbox"] for e in p if e.tag == "page-number" and "bbox" in e.attrib
                ],
            }
            for p in pages
        ]

        n = len(ordered)
        perm = list(range(n))
        self._rng.shuffle(perm)
        shuffled = [ordered[orig_pos] for orig_pos in perm]
        target = [perm.index(i) for i in range(n)]

        key = f"{doc_name}_page_sort_0"
        payload: dict[str, Any] = {
            "__key__": key,
            "input.json": shuffled,
            "target.json": target,
            "json": {
                "task_type": self.task_name,
                "doc_name": doc_name,
                "n_pages": n,
            },
        }
        return [Sample(key=key, payload=payload)]
