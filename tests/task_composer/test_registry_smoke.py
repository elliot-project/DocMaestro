# tests/task_composer/test_registry_smoke.py
"""Smoke test: every registered composer runs over a synthetic document."""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

from src.task_composer import registry
from src.task_composer.base import Sample


def _doc() -> ET.Element:
    root = ET.Element("document", {"name": "smoke", "page_count": "2"})
    for pi in range(2):
        page = ET.SubElement(root, "page", {"idx": str(pi), "width": "600", "height": "800"})
        ET.SubElement(page, "title", {"bbox": "[0, 0, 600, 30]", "level": "1"}).text = f"H{pi}"
        for j in range(4):
            y2 = j * 40 + 30
            tb = ET.SubElement(
                page, "text-block", {"bbox": f"[0, {j * 40}, 600, {y2}]"}
            )
            tb.text = f"para {pi}-{j}"
        fig = ET.SubElement(page, "figure", {"bbox": "[0, 200, 300, 400]"})
        ET.SubElement(fig, "img", {"src": f"images/p{pi}.jpg"})
        cap = ET.SubElement(fig, "caption")
        cap.text = f"caption {pi}"
        ET.SubElement(page, "page-number", {"bbox": "[280, 780, 320, 800]"}).text = str(pi + 1)
    return root


def test_every_registered_composer_runs(tmp_path: Path):
    root = _doc()
    for spec in registry.list_tasks():
        composer = registry.get_composer(spec.task_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples = composer.compose_document(root, tmp_path)
        assert isinstance(samples, list), f"{spec.task_name} did not return a list"
        assert all(isinstance(s, Sample) for s in samples), (
            f"{spec.task_name} returned non-Sample objects"
        )


def test_registry_has_all_thirteen_tasks():
    names = {s.task_name for s in registry.list_tasks()}
    expected = {
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
    }
    assert names == expected
