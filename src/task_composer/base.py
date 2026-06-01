"""Core data types and abstract base class for the Task Composer module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


@dataclass
class Sample:
    """A single WebDataset-compatible task sample.

    Args:
        key: Unique sample identifier, e.g. ``adidas_p002_caption_pred_0``.
        payload: WebDataset entry dict. Must contain ``"__key__"``. All other
            keys are file extensions (``"json"``, ``"input.txt"``, ``"image.jpg"``
            etc.) whose values are ``str``, ``dict``/``list``, or ``bytes``.
    """

    key: str
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if "__key__" not in self.payload:
            raise ValueError(
                f"Sample payload missing '__key__': {self.key!r}. "
                "Add payload['__key__'] = self.key before constructing."
            )


@dataclass
class DocumentContext:
    """Snapshot of a full document, passed to every compose_page call.

    Args:
        doc_name: Document identifier (matches XML root ``name`` attribute).
        pages: All page ``ET.Element`` objects in document order.
        img_root: Base directory for resolving relative ``<img src>`` paths.
    """

    doc_name: str
    pages: list[ET.Element]
    img_root: Path
