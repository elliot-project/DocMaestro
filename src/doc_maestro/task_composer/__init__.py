"""Task Composer — WebDataset SSL task generation from parsed document XML."""

from doc_maestro.task_composer.base import BaseTaskComposer, DocumentContext, Sample
from doc_maestro.task_composer.registry import REGISTRY, TaskSpec, get_composer, list_tasks
from doc_maestro.task_composer.shard_writer import ShardWriter
from doc_maestro.task_composer.tasks.basic_reading import BasicReadingComposer
from doc_maestro.task_composer.tasks.caption_prediction import CaptionPredictionComposer
from doc_maestro.task_composer.tasks.doc_reconstruction import DocReconstructionComposer
from doc_maestro.task_composer.tasks.dom_edition import DomEditionComposer
from doc_maestro.task_composer.tasks.figure_ordering import FigureOrderingComposer
from doc_maestro.task_composer.tasks.figure_selection import FigureSelectionComposer
from doc_maestro.task_composer.tasks.form_filling import FormFillingComposer
from doc_maestro.task_composer.tasks.layout import LayoutComposer
from doc_maestro.task_composer.tasks.markdown import MarkdownComposer
from doc_maestro.task_composer.tasks.ocr import OcrComposer
from doc_maestro.task_composer.tasks.page_sorting import PageSortingComposer
from doc_maestro.task_composer.tasks.page_tree import PageTreeComposer
from doc_maestro.task_composer.tasks.reading_order import ReadingOrderComposer

__all__ = [
    "BaseTaskComposer",
    "DocumentContext",
    "Sample",
    "ShardWriter",
    "REGISTRY",
    "TaskSpec",
    "get_composer",
    "list_tasks",
    "BasicReadingComposer",
    "CaptionPredictionComposer",
    "DocReconstructionComposer",
    "DomEditionComposer",
    "FigureOrderingComposer",
    "FigureSelectionComposer",
    "FormFillingComposer",
    "LayoutComposer",
    "MarkdownComposer",
    "OcrComposer",
    "PageSortingComposer",
    "PageTreeComposer",
    "ReadingOrderComposer",
]
