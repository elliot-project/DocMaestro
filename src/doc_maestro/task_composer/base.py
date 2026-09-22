"""Core data types and abstract base class for the Task Composer module."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar


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
        if self.payload["__key__"] != self.key:
            raise ValueError(
                f"Sample key mismatch: key={self.key!r} but payload['__key__']={self.payload['__key__']!r}. "
                "They must be identical."
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
    pages: list[ET.Element]  # document order; do not mutate after construction
    img_root: Path


class BaseTaskComposer(ABC):
    """Abstract base class for all SSL task composers.

    Subclasses must declare a ``task_name`` class variable and implement
    ``compose_page``. The ``compose_document`` default iterates every page
    and collects samples — override it only if the task requires multi-page
    coordination during composition.
    """

    task_name: ClassVar[str]

    def __init__(self) -> None:
        # Truthiness check intentionally rejects both missing and blank task_name
        if not getattr(type(self), "task_name", None):
            raise TypeError(
                f"{type(self).__name__} must define a non-empty 'task_name' class variable"
            )

    @abstractmethod
    def compose_page(
        self,
        page: ET.Element,
        context: DocumentContext,
    ) -> list[Sample]:
        """Produce all task samples for a single page.

        Args:
            page: The ``<page>`` ET.Element to compose samples from.
            context: Full document context for cross-page lookups.

        Returns:
            List of ``Sample`` objects, possibly empty if no eligible
            elements exist on this page.
        """

    def compose_document(
        self,
        root: ET.Element,
        img_root: Path,
    ) -> list[Sample]:
        """Compose samples for every page in a document.

        Default implementation iterates all pages in document order.
        Override if a task needs coordination across pages during composition.

        Args:
            root: The ``<document>`` root ET.Element.
            img_root: Base directory for resolving image src paths.

        Returns:
            Flat list of all samples produced across all pages.
        """
        ctx = DocumentContext(
            doc_name=root.attrib["name"],
            pages=list(root),
            img_root=img_root,
        )
        samples: list[Sample] = []
        for page in ctx.pages:
            samples.extend(self.compose_page(page, ctx))
        return samples
