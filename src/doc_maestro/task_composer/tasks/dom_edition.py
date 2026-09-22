# src/task_composer/tasks/dom_edition.py
"""DocTask 6: DOM tree edition (correct a corrupted logical tree)."""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from typing import Any

from doc_maestro.task_composer.base import BaseTaskComposer, DocumentContext, Sample
from doc_maestro.task_composer.xml_ops import mutate_tree, page_idx, strip_bbox


class DomEditionComposer(BaseTaskComposer):
    """SSL task: repair a structurally mutated page tree.

    The target is the clean, bbox-free page XML; the input is that XML with one
    structural mutation injected (delete/retag/shuffle). Pages with fewer than
    two children are skipped — mutation needs something to act on.

    Args:
        seed: RNG seed for reproducible mutation selection.
    """

    task_name = "dom_edition"

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self._rng = random.Random(seed)

    def compose_page(self, page: ET.Element, context: DocumentContext) -> list[Sample]:
        """Produce one (corrupted, clean) tree pair for the page, or none."""
        if len(list(page)) < 2:
            return []
        clean = strip_bbox(page)
        corrupted = mutate_tree(clean, self._rng)
        idx = page_idx(page)
        key = f"{context.doc_name}_p{idx:03d}_dom_edit_0"
        payload: dict[str, Any] = {
            "__key__": key,
            "input.txt": corrupted,
            "target.txt": clean,
            "json": {
                "task_type": self.task_name,
                "doc_name": context.doc_name,
                "page_idx": idx,
            },
        }
        return [Sample(key=key, payload=payload)]
