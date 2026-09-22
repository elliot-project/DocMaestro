"""Tests for src/dataset/contract.py — per-task input/output modality contract."""

from __future__ import annotations

import pytest

from src.dataset.contract import (
    CONTRACTS,
    AssetPolicy,
    TargetFormat,
    geometric_augmentation_allowed,
    get_contract,
)
from src.task_composer import registry


class TestCoverage:
    def test_every_registered_task_has_a_contract(self) -> None:
        registered = {spec.task_name for spec in registry.list_tasks()}
        assert registered == set(CONTRACTS)

    def test_get_contract_returns_matching_task_name(self) -> None:
        assert get_contract("markdown").task_name == "markdown"

    def test_unknown_task_raises(self) -> None:
        with pytest.raises(KeyError):
            get_contract("not_a_task")


class TestPageLevelTasks:
    @pytest.mark.parametrize(
        "task", ["basic_reading", "ocr", "layout", "markdown", "page_tree", "form_filling"]
    )
    def test_single_page_no_figures(self, task: str) -> None:
        c = get_contract(task)
        assert c.page_images is AssetPolicy.SINGLE
        assert c.figure_images is AssetPolicy.NONE


class TestLeakAvoidance:
    """The four deliberate departures: the naive input contains the answer."""

    def test_doc_reconstruction_gets_no_page_image(self) -> None:
        # Showing the page turns layout inference into coordinate reading.
        c = get_contract("doc_reconstruction")
        assert c.page_images is AssetPolicy.NONE
        assert c.figure_images is AssetPolicy.NONE

    def test_caption_prediction_gets_crop_not_page(self) -> None:
        # The caption is printed on the page beside the figure.
        c = get_contract("caption_prediction")
        assert c.page_images is AssetPolicy.NONE
        assert c.figure_images is AssetPolicy.SINGLE

    def test_figure_selection_gets_crops_not_page(self) -> None:
        # The correct figure is visibly on that page.
        c = get_contract("figure_selection")
        assert c.page_images is AssetPolicy.NONE
        assert c.figure_images is AssetPolicy.MULTI

    def test_page_sorting_redacts_page_numbers(self) -> None:
        # Without redaction the task degenerates to OCR of a page number.
        c = get_contract("page_sorting")
        assert c.page_images is AssetPolicy.MULTI
        assert c.redact_page_numbers is True

    def test_page_sorting_is_the_only_redacting_task(self) -> None:
        redacting = {n for n, c in CONTRACTS.items() if c.redact_page_numbers}
        assert redacting == {"page_sorting"}


class TestMultiAssetTasks:
    def test_figure_ordering_needs_both_kinds(self) -> None:
        c = get_contract("figure_ordering")
        assert c.page_images is AssetPolicy.MULTI
        assert c.figure_images is AssetPolicy.MULTI

    def test_dom_edition_is_page_plus_text_input(self) -> None:
        c = get_contract("dom_edition")
        assert c.page_images is AssetPolicy.SINGLE
        assert c.text_input is True

    def test_reading_order_is_page_plus_text_input(self) -> None:
        c = get_contract("reading_order")
        assert c.page_images is AssetPolicy.SINGLE
        assert c.text_input is True

    def test_basic_reading_has_no_text_input(self) -> None:
        assert get_contract("basic_reading").text_input is False


class TestBboxBearing:
    def test_exactly_the_coordinate_emitting_tasks(self) -> None:
        bearing = {n for n, c in CONTRACTS.items() if c.bbox_bearing}
        assert bearing == {"ocr", "layout", "doc_reconstruction", "caption_prediction"}

    def test_geometric_augmentation_forbidden_on_bbox_tasks(self) -> None:
        for task in ("ocr", "layout", "doc_reconstruction", "caption_prediction"):
            assert geometric_augmentation_allowed(task) is False

    def test_geometric_augmentation_allowed_on_markdown(self) -> None:
        assert geometric_augmentation_allowed("markdown") is True

    def test_geometric_augmentation_unknown_task_raises(self) -> None:
        with pytest.raises(KeyError):
            geometric_augmentation_allowed("not_a_task")


class TestTargetFormats:
    @pytest.mark.parametrize(
        ("task", "fmt"),
        [
            ("basic_reading", TargetFormat.TEXT),
            ("caption_prediction", TargetFormat.TEXT),
            ("markdown", TargetFormat.MARKDOWN),
            ("page_tree", TargetFormat.XML),
            ("dom_edition", TargetFormat.XML),
            ("ocr", TargetFormat.JSON),
            ("layout", TargetFormat.JSON),
            ("reading_order", TargetFormat.JSON),
            ("doc_reconstruction", TargetFormat.JSON),
            ("figure_selection", TargetFormat.JSON),
            ("figure_ordering", TargetFormat.JSON),
            ("page_sorting", TargetFormat.JSON),
            ("form_filling", TargetFormat.JSON),
        ],
    )
    def test_target_format(self, task: str, fmt: TargetFormat) -> None:
        assert get_contract(task).target_format is fmt


class TestImmutability:
    def test_contract_is_frozen(self) -> None:
        with pytest.raises(Exception):
            get_contract("markdown").task_name = "other"  # type: ignore[misc]
