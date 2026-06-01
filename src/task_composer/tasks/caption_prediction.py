"""Caption prediction SSL task composer."""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from typing import Any

from src.task_composer.base import BaseTaskComposer, DocumentContext, Sample

_CAPTIONABLE_TAGS = {"figure", "chart", "table"}
_SKIP_CONTEXT_TAGS = {"page-header", "page-footer", "page-number", "abandon"}


def _elem_to_dict(elem: ET.Element) -> dict[str, str]:
    return {
        "tag": elem.tag,
        "text": (elem.text or "").strip()[:120],
        "bbox": elem.attrib.get("bbox", ""),
    }


class CaptionPredictionComposer(BaseTaskComposer):
    """SSL task: given an image and surrounding page context, predict the caption.

    For each ``<figure>``, ``<chart>``, or ``<table>`` with a non-empty
    ``<caption>`` child, produces one sample per page. The input carries:

    - Up to 3 sibling elements **before** the target in document order,
      listed closest-first (reverse reading order).
    - Up to 3 sibling elements **after** the target in document order,
      listed closest-first.
    - The target element's image bytes (omitted with a warning if unresolvable).

    The target is the raw caption text.
    """

    task_name = "caption_prediction"

    def compose_page(self, page: ET.Element, context: DocumentContext) -> list[Sample]:
        elements = list(page)
        page_idx = int(page.attrib["idx"])
        samples: list[Sample] = []
        sample_idx = 0

        # Build a filtered sibling list (excludes skip tags) with original positions
        sibling_positions = [
            (orig_idx, elem)
            for orig_idx, elem in enumerate(elements)
            if elem.tag not in _SKIP_CONTEXT_TAGS
        ]

        for orig_idx, elem in enumerate(elements):
            if elem.tag not in _CAPTIONABLE_TAGS:
                continue

            caption_el = elem.find("caption")
            if caption_el is None or not (caption_el.text or "").strip():
                continue

            caption_text = caption_el.text.strip()

            # Find position of this element in the sibling list.
            # None is unreachable: _CAPTIONABLE_TAGS and _SKIP_CONTEXT_TAGS are disjoint.
            sib_pos = next(
                (i for i, (o, _) in enumerate(sibling_positions) if o == orig_idx),
                None,
            )
            if sib_pos is None:  # pragma: no cover
                continue

            # Siblings excluding the target itself
            before = [e for i, (_, e) in enumerate(sibling_positions) if i < sib_pos]
            after = [e for i, (_, e) in enumerate(sibling_positions) if i > sib_pos]

            preceding = before[-3:][::-1]  # up to 3, closest-first
            following = after[:3]          # up to 3, closest-first

            key = f"{context.doc_name}_p{page_idx:03d}_caption_pred_{sample_idx}"

            payload: dict[str, Any] = {
                "__key__": key,
                "json": {
                    "task_type": self.task_name,
                    "doc_name": context.doc_name,
                    "page_idx": page_idx,
                    "element_tag": elem.tag,
                    "bbox": elem.attrib.get("bbox", ""),
                },
                "input.json": {
                    "preceding": [_elem_to_dict(e) for e in preceding],
                    "following": [_elem_to_dict(e) for e in following],
                },
                "target.txt": caption_text,
            }

            img_el = elem.find("img")
            if img_el is not None:
                img_src = img_el.attrib.get("src", "")
                img_path = context.img_root / img_src
                if img_path.exists():
                    payload["image.jpg"] = img_path.read_bytes()
                else:
                    warnings.warn(
                        f"Image not found: {img_path} — omitting image.jpg from sample",
                        stacklevel=2,
                    )

            samples.append(Sample(key=key, payload=payload))
            sample_idx += 1

        return samples
