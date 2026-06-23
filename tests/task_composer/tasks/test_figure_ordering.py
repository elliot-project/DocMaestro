# tests/task_composer/tasks/test_figure_ordering.py
"""Tests for FigureOrderingComposer."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from src.task_composer.tasks.figure_ordering import FigureOrderingComposer


def _doc() -> ET.Element:
    root = ET.Element("document", {"name": "doc"})
    p0 = ET.SubElement(root, "page", {"idx": "0"})
    f0 = ET.SubElement(p0, "figure", {"bbox": "[0, 0, 1, 1]"})
    ET.SubElement(f0, "img", {"src": "images/a.jpg"})
    p1 = ET.SubElement(root, "page", {"idx": "1"})
    f1 = ET.SubElement(p1, "figure", {"bbox": "[0, 0, 1, 1]"})
    ET.SubElement(f1, "img", {"src": "images/b.jpg"})
    return root


def test_task_name():
    assert FigureOrderingComposer.task_name == "figure_ordering"


def test_maps_each_figure_to_origin_page(tmp_path: Path):
    samples = FigureOrderingComposer(seed=0).compose_document(_doc(), tmp_path)
    assert len(samples) == 1
    s = samples[0]
    figures = s.payload["input.json"]["figures"]
    target = s.payload["target.json"]
    src_to_page = {"images/a.jpg": 0, "images/b.jpg": 1}
    for i, fig in enumerate(figures):
        assert target[i] == src_to_page[fig["src"]]
    assert s.payload["input.json"]["pages"] == [0, 1]


def test_doc_without_figures_yields_no_samples(tmp_path: Path):
    root = ET.Element("document", {"name": "doc"})
    ET.SubElement(root, "page", {"idx": "0"})
    ET.SubElement(root, "page", {"idx": "1"})
    assert FigureOrderingComposer(seed=0).compose_document(root, tmp_path) == []
