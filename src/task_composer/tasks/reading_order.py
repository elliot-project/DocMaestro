"""Reading order SSL task composer."""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from typing import Any

from src.task_composer.base import BaseTaskComposer, DocumentContext, Sample

_SKIP_TAGS = {"page-header", "page-footer", "page-number", "abandon"}


class ReadingOrderComposer(BaseTaskComposer):
    """SSL task: restore the correct reading order of shuffled page elements.

    Samples ``n_elements`` elements from eligible page content (excluding
    headers, footers, and abandoned blocks), shuffles them, and asks for
    the permutation that restores the original document order.

    ``input.json`` is the shuffled list; each item has ``idx`` = its position
    in the shuffled list (0-indexed), ``tag``, and ``text`` (truncated to 120
    chars). ``target.json`` is a list of indices into the shuffled list that
    restores the correct order:
    ``[shuffled[target[i]] for i in range(n)] == correct_order``.

    Args:
        n_elements: Number of elements to sample per page (default 5).
        seed: RNG seed for reproducible element sampling and shuffling.
    """

    task_name = "reading_order"

    def __init__(self, n_elements: int = 5, seed: int | None = None) -> None:
        super().__init__()
        self.n_elements = n_elements
        self._rng = random.Random(seed)

    def compose_page(self, page: ET.Element, context: DocumentContext) -> list[Sample]:
        """Produce reading order task samples for a single page.

        Selects ``n_elements`` eligible elements (skipping headers, footers,
        page numbers, and abandoned blocks), shuffles them, and constructs a
        sample whose target is the permutation index list that restores the
        original document order.

        Args:
            page: The ``<page>`` ET.Element to compose samples from.
            context: Full document context (doc_name, pages, img_root).

        Returns:
            A list containing one ``Sample`` if the page has at least
            ``n_elements`` eligible elements, otherwise an empty list.
        """
        page_idx = int(page.attrib["idx"])

        eligible = [
            (orig_idx, elem)
            for orig_idx, elem in enumerate(page)
            if elem.tag not in _SKIP_TAGS
        ]

        if len(eligible) < self.n_elements:
            return []

        # Sample n_elements, preserving original document order
        chosen = sorted(
            self._rng.sample(eligible, self.n_elements),
            key=lambda x: x[0],
        )

        # Build descriptions in correct (original) order; local idx 0..n-1
        correct_descriptions = [
            {
                "idx": local_idx,
                "tag": elem.tag,
                "text": (elem.text or "").strip()[:120],
            }
            for local_idx, (_, elem) in enumerate(chosen)
        ]

        # Shuffle: produce a permutation of indices 0..n-1
        shuffled_perm = list(range(self.n_elements))
        self._rng.shuffle(shuffled_perm)

        # shuffled[i] = correct_descriptions[shuffled_perm[i]], with idx updated to i
        shuffled = [
            {**correct_descriptions[orig_pos], "idx": shuffled_pos}
            for shuffled_pos, orig_pos in enumerate(shuffled_perm)
        ]

        # target[i] = position in shuffled that holds the i-th correct element
        # i.e., [shuffled[target[i]] for i in range(n)] == correct_descriptions (by content)
        target = [shuffled_perm.index(i) for i in range(self.n_elements)]

        key = f"{context.doc_name}_p{page_idx:03d}_read_ord_0"

        payload: dict[str, Any] = {
            "__key__": key,
            "input.json": shuffled,
            "target.json": target,
            "json": {
                "task_type": self.task_name,
                "doc_name": context.doc_name,
                "page_idx": page_idx,
                "n_elements": self.n_elements,
            },
        }

        return [Sample(key=key, payload=payload)]
