# src/task_composer/tasks/doc_reconstruction.py
"""DocTask 9: document reconstruction (place shuffled elements on a canvas)."""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from typing import Any

from doc_maestro.task_composer.base import BaseTaskComposer, DocumentContext, Sample
from doc_maestro.task_composer.xml_ops import NOISE_TAGS, page_idx


class DocReconstructionComposer(BaseTaskComposer):
    """SSL task: reassemble a page from its isolated, shuffled elements.

    For each page with at least two bbox-bearing, non-noise elements, shuffles
    them and asks for the target bounding box of each. ``input.json[i]`` carries
    ``{idx, tag, text}`` and ``target.json[i]`` is the bbox where that element
    belongs, so the two lists are positionally aligned.

    Args:
        seed: RNG seed for reproducible shuffling.
    """

    task_name = "doc_reconstruction"

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self._rng = random.Random(seed)

    def compose_page(self, page: ET.Element, context: DocumentContext) -> list[Sample]:
        """Produce one reconstruction sample for the page, or none."""
        elements = [el for el in page if el.tag not in NOISE_TAGS and "bbox" in el.attrib]
        if len(elements) < 2:
            return []

        order = list(range(len(elements)))
        self._rng.shuffle(order)
        shuffled_elems = [elements[i] for i in order]

        input_items = [
            {
                "idx": pos,
                "tag": el.tag,
                "text": (el.text or "").strip()[:80],
            }
            for pos, el in enumerate(shuffled_elems)
        ]
        target = [el.attrib["bbox"] for el in shuffled_elems]

        idx = page_idx(page)
        key = f"{context.doc_name}_p{idx:03d}_doc_recon_0"
        payload: dict[str, Any] = {
            "__key__": key,
            "input.json": input_items,
            "target.json": target,
            "json": {
                "task_type": self.task_name,
                "doc_name": context.doc_name,
                "page_idx": idx,
            },
        }
        return [Sample(key=key, payload=payload)]
