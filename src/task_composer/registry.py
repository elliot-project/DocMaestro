"""Central registry of SSL task composers.

Single source of truth consumed by the visualizer and any batch runner. Each
task is a ``TaskSpec`` carrying the composer factory, its default config, and a
``view_type`` string the frontend uses to pick a renderer. Adding a task = add
its composer module and append one ``TaskSpec`` to ``_SPECS``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.task_composer.base import BaseTaskComposer
from src.task_composer.tasks.basic_reading import BasicReadingComposer
from src.task_composer.tasks.caption_prediction import CaptionPredictionComposer
from src.task_composer.tasks.doc_reconstruction import DocReconstructionComposer
from src.task_composer.tasks.dom_edition import DomEditionComposer
from src.task_composer.tasks.figure_ordering import FigureOrderingComposer
from src.task_composer.tasks.figure_selection import FigureSelectionComposer
from src.task_composer.tasks.form_filling import FormFillingComposer
from src.task_composer.tasks.layout import LayoutComposer
from src.task_composer.tasks.markdown import MarkdownComposer
from src.task_composer.tasks.ocr import OcrComposer
from src.task_composer.tasks.page_sorting import PageSortingComposer
from src.task_composer.tasks.page_tree import PageTreeComposer
from src.task_composer.tasks.reading_order import ReadingOrderComposer


@dataclass(frozen=True)
class TaskSpec:
    """Registry entry describing one SSL task.

    Args:
        task_name: Stable identifier, matches the composer's ``task_name``.
        composer_factory: Callable that builds the composer from config kwargs.
        view_type: Frontend renderer key (see Plan 2).
        default_config: Default kwargs merged under caller overrides.
        description: One-line human summary.
    """

    task_name: str
    composer_factory: Callable[..., BaseTaskComposer]
    view_type: str
    default_config: dict
    description: str


_SPECS: list[TaskSpec] = [
    TaskSpec(
        "reading_order",
        ReadingOrderComposer,
        "reading_order",
        {"seed": 42},
        "Restore the reading order of shuffled page elements.",
    ),
    TaskSpec(
        "caption_prediction",
        CaptionPredictionComposer,
        "caption",
        {},
        "Predict a figure/table caption from surrounding context.",
    ),
    TaskSpec(
        "figure_selection",
        FigureSelectionComposer,
        "image_grid",
        {"seed": 42},
        "DocTask 7: contrastive image selection from page text.",
    ),
    TaskSpec(
        "basic_reading",
        BasicReadingComposer,
        "text",
        {},
        "DocTask 1: extract full-page text in reading order.",
    ),
    TaskSpec(
        "ocr",
        OcrComposer,
        "bbox_overlay",
        {},
        "DocTask 2: OCR text + block bboxes.",
    ),
    TaskSpec(
        "layout",
        LayoutComposer,
        "bbox_overlay",
        {},
        "DocTask 3: layout class labels + bboxes.",
    ),
    TaskSpec(
        "markdown",
        MarkdownComposer,
        "markdown",
        {},
        "DocTask 4: image to semantic markdown.",
    ),
    TaskSpec(
        "page_tree",
        PageTreeComposer,
        "xml_tree",
        {},
        "DocTask 5: page to bbox-free XML tree.",
    ),
    TaskSpec(
        "dom_edition",
        DomEditionComposer,
        "xml_diff",
        {"seed": 42},
        "DocTask 6: repair a corrupted DOM tree.",
    ),
    TaskSpec(
        "page_sorting",
        PageSortingComposer,
        "page_sequence",
        {"seed": 42},
        "DocTask 8: recover page reading order.",
    ),
    TaskSpec(
        "doc_reconstruction",
        DocReconstructionComposer,
        "canvas_layout",
        {"seed": 42},
        "DocTask 9: reconstruct page layout from shuffled elements.",
    ),
    TaskSpec(
        "figure_ordering",
        FigureOrderingComposer,
        "figure_page_map",
        {"seed": 42},
        "DocTask 10: map figures to origin pages.",
    ),
    TaskSpec(
        "form_filling",
        FormFillingComposer,
        "kv_form",
        {},
        "DocTask 11: form key-value extraction from CommonForms.",
    ),
]

REGISTRY: dict[str, TaskSpec] = {spec.task_name: spec for spec in _SPECS}


def get_composer(task_name: str, **overrides) -> BaseTaskComposer:
    """Instantiate a composer by task name with optional config overrides.

    Args:
        task_name: Key into ``REGISTRY``.
        **overrides: Kwargs merged over the spec's ``default_config``.

    Returns:
        A ready ``BaseTaskComposer`` instance.

    Raises:
        KeyError: If ``task_name`` is not registered.
    """
    spec = REGISTRY[task_name]
    config = {**spec.default_config, **overrides}
    return spec.composer_factory(**config)


def list_tasks() -> list[TaskSpec]:
    """Return all registered task specs in registration order."""
    return list(REGISTRY.values())
