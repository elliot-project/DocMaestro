# tests/task_composer/test_registry.py
"""Tests for the task registry."""

from __future__ import annotations

from doc_maestro.task_composer import registry
from doc_maestro.task_composer.base import BaseTaskComposer


def test_existing_tasks_registered():
    names = {s.task_name for s in registry.list_tasks()}
    assert {
        "reading_order",
        "caption_prediction",
        "figure_selection",
        "basic_reading",
        "ocr",
        "layout",
        "markdown",
        "page_tree",
        "dom_edition",
        "page_sorting",
        "doc_reconstruction",
        "figure_ordering",
        "form_filling",
    } <= names


def test_every_spec_has_nonempty_view_type():
    for spec in registry.list_tasks():
        assert spec.view_type, f"{spec.task_name} missing view_type"


def test_get_composer_returns_instance():
    comp = registry.get_composer("figure_selection")
    assert isinstance(comp, BaseTaskComposer)


def test_get_composer_applies_overrides():
    comp = registry.get_composer("reading_order", n_elements=3)
    assert comp.n_elements == 3
