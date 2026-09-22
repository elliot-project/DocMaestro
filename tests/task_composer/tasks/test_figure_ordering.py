# tests/task_composer/tasks/test_figure_ordering.py
"""Tests for FigureOrderingComposer."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from doc_maestro.task_composer.tasks.figure_ordering import FigureOrderingComposer

# Figure src → origin page index, mirroring the layout built by _doc().
SRC_TO_PAGE = {
    "images/a.jpg": 0,
    "images/b.jpg": 0,
    "images/c.jpg": 1,
    "images/d.jpg": 1,
}


def _page_with_figures(root: ET.Element, idx: int, srcs: list[str]) -> ET.Element:
    page = ET.SubElement(root, "page", {"idx": str(idx)})
    for src in srcs:
        fig = ET.SubElement(page, "figure", {"bbox": "[0, 0, 1, 1]"})
        ET.SubElement(fig, "img", {"src": src})
    return page


def _doc() -> ET.Element:
    """Two pages, four figures — enough to clear the composer's 3-figure minimum."""
    root = ET.Element("document", {"name": "doc"})
    _page_with_figures(root, 0, ["images/a.jpg", "images/b.jpg"])
    _page_with_figures(root, 1, ["images/c.jpg", "images/d.jpg"])
    return root


def test_task_name():
    assert FigureOrderingComposer.task_name == "figure_ordering"


def test_maps_each_figure_to_origin_page(tmp_path: Path):
    samples = FigureOrderingComposer(seed=0).compose_document(_doc(), tmp_path)
    assert len(samples) == 1
    s = samples[0]
    figures = s.payload["input.json"]["figures"]
    target = s.payload["target.json"]
    for i, fig in enumerate(figures):
        assert target[i] == SRC_TO_PAGE[fig["src"]]
    assert s.payload["input.json"]["pages"] == [0, 1]


def test_samples_between_three_and_five_figures(tmp_path: Path):
    """The composer subsamples 3–5 figures rather than emitting all of them."""
    root = ET.Element("document", {"name": "doc"})
    _page_with_figures(root, 0, [f"images/{i}.jpg" for i in range(8)])
    _page_with_figures(root, 1, [f"images/{i}.jpg" for i in range(8, 16)])
    samples = FigureOrderingComposer(seed=0).compose_document(root, tmp_path)
    assert len(samples) == 1
    figures = samples[0].payload["input.json"]["figures"]
    assert 3 <= len(figures) <= 5


def test_doc_with_fewer_than_three_figures_yields_no_samples(tmp_path: Path):
    """Below the 3-figure minimum the task is trivial, so nothing is emitted."""
    root = ET.Element("document", {"name": "doc"})
    _page_with_figures(root, 0, ["images/a.jpg"])
    _page_with_figures(root, 1, ["images/c.jpg"])
    assert FigureOrderingComposer(seed=0).compose_document(root, tmp_path) == []


def test_single_page_doc_yields_no_samples(tmp_path: Path):
    root = ET.Element("document", {"name": "doc"})
    _page_with_figures(root, 0, ["images/a.jpg", "images/b.jpg", "images/c.jpg"])
    assert FigureOrderingComposer(seed=0).compose_document(root, tmp_path) == []


def test_doc_without_figures_yields_no_samples(tmp_path: Path):
    root = ET.Element("document", {"name": "doc"})
    ET.SubElement(root, "page", {"idx": "0"})
    ET.SubElement(root, "page", {"idx": "1"})
    assert FigureOrderingComposer(seed=0).compose_document(root, tmp_path) == []
