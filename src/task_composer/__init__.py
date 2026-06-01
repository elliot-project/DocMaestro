"""Task Composer — WebDataset SSL task generation from parsed document XML."""

from src.task_composer.base import BaseTaskComposer, DocumentContext, Sample
from src.task_composer.shard_writer import ShardWriter
from src.task_composer.tasks.caption_prediction import CaptionPredictionComposer
from src.task_composer.tasks.figure_selection import FigureSelectionComposer
from src.task_composer.tasks.reading_order import ReadingOrderComposer

__all__ = [
    "BaseTaskComposer",
    "DocumentContext",
    "Sample",
    "ShardWriter",
    "CaptionPredictionComposer",
    "FigureSelectionComposer",
    "ReadingOrderComposer",
]
