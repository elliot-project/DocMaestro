# src/task_composer/tasks/figure_ordering.py
"""DocTask 10: ordering figures (map each figure to its origin page)."""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from src.task_composer.base import BaseTaskComposer, DocumentContext, Sample
from src.task_composer.xml_ops import page_idx


class FigureOrderingComposer(BaseTaskComposer):
    """SSL task: ground isolated figures to the pages they came from.

    Produces one sample per document. The input lists the document's page
    indices and its figures (shuffled, by ``src``); the target maps each figure
    to its origin page index. Requires at least two pages and one figure.

    Args:
        seed: RNG seed for reproducible figure shuffling.
    """

    task_name = "figure_ordering"

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self._rng = random.Random(seed)

    def compose_page(self, page: ET.Element, context: DocumentContext) -> list[Sample]:
        """Unused — figure ordering is a document-level task."""
        return []

    def compose_document(self, root: ET.Element, img_root: Path) -> list[Sample]:
        """Produce a single figure-to-page mapping sample for the document."""
        pages = list(root)
        if len(pages) < 2:
            return []

        doc_name = root.attrib["name"]
        page_indices = [page_idx(p) for p in pages]

        figures: list[dict] = []
        for p in pages:
            origin = page_idx(p)
            for el in p:
                if el.tag == "figure":
                    img = el.find("img")
                    if img is not None and img.attrib.get("src"):
                        figures.append({"src": img.attrib["src"], "page_origin": origin})

        if not figures:
            return []

        self._rng.shuffle(figures)
        input_figures = [{"idx": i, "src": f["src"]} for i, f in enumerate(figures)]
        target = [f["page_origin"] for f in figures]

        key = f"{doc_name}_fig_order_0"
        payload: dict[str, Any] = {
            "__key__": key,
            "input.json": {"pages": page_indices, "figures": input_figures},
            "target.json": target,
            "json": {
                "task_type": self.task_name,
                "doc_name": doc_name,
                "n_figures": len(figures),
            },
        }
        return [Sample(key=key, payload=payload)]
