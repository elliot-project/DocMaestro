"""Figure selection SSL task composer."""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from typing import Any

from src.task_composer.base import BaseTaskComposer, DocumentContext, Sample


class FigureSelectionComposer(BaseTaskComposer):
    """SSL task: given page text, identify which figure belongs to this page.

    For each page with at least one figure, produces one sample per figure.
    The candidates list contains the correct figure plus ``n_distractors``
    figures drawn randomly from other pages in the same document.

    Args:
        n_distractors: Number of distractor figures per sample (default 2,
            giving 3 total candidates).
        seed: RNG seed for reproducible distractor sampling and shuffling.
    """

    task_name = "figure_selection"

    def __init__(self, n_distractors: int = 2, seed: int | None = None) -> None:
        super().__init__()
        self.n_distractors = n_distractors
        self._rng = random.Random(seed)

    def compose_page(self, page: ET.Element, context: DocumentContext) -> list[Sample]:
        """Produce figure-selection samples for a single page.

        For each figure on the page, builds a candidate set of (1 correct +
        n_distractors) figures, shuffles them, and records which index holds
        the correct figure as the label.

        Args:
            page: The ``<page>`` ET.Element to compose samples from.
            context: Full document context for cross-page distractor lookup.

        Returns:
            List of ``Sample`` objects — one per figure on this page — or
            empty list if the page has no figures or there are not enough
            distractors available from other pages.
        """
        page_idx = int(page.attrib["idx"])

        page_figures = [
            e for e in page
            if e.tag == "figure" and e.find("img") is not None
        ]
        if not page_figures:
            return []

        # Collect all figures from other pages: (element, page_idx_of_origin)
        other_figures: list[tuple[ET.Element, int]] = [
            (fig, int(other_page.attrib["idx"]))
            for other_page in context.pages
            if int(other_page.attrib["idx"]) != page_idx
            for fig in other_page
            if fig.tag == "figure" and fig.find("img") is not None
        ]

        if len(other_figures) < self.n_distractors:
            return []

        page_text = "\n".join(
            (e.text or "").strip()
            for e in page
            if e.tag in {"text-block", "title"} and (e.text or "").strip()
        )

        samples: list[Sample] = []
        for sample_idx, correct_fig in enumerate(page_figures):
            correct_src = correct_fig.find("img").attrib.get("src", "")
            correct_candidate = {"src": correct_src, "page_origin": page_idx}

            distractors = self._rng.sample(other_figures, self.n_distractors)
            distractor_candidates = [
                {"src": dfig.find("img").attrib.get("src", ""), "page_origin": dpage_idx}
                for dfig, dpage_idx in distractors
            ]

            candidates = [correct_candidate] + distractor_candidates
            self._rng.shuffle(candidates)
            label = candidates.index(correct_candidate)

            key = f"{context.doc_name}_p{page_idx:03d}_fig_sel_{sample_idx}"

            payload: dict[str, Any] = {
                "__key__": key,
                "input.txt": page_text,
                "json": {
                    "task_type": self.task_name,
                    "doc_name": context.doc_name,
                    "page_idx": page_idx,
                    "candidates": candidates,
                    "label": label,
                },
            }

            samples.append(Sample(key=key, payload=payload))

        return samples
