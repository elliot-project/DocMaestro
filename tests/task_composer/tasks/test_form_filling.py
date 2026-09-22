# tests/task_composer/tasks/test_form_filling.py
"""Tests for FormFillingComposer."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from doc_maestro.task_composer.tasks.form_filling import FormFillingComposer


def _write_sidecar(root: Path, doc_name: str) -> None:
    sidecar = {
        "doc_id": doc_name,
        "source": "commonforms",
        "bbox_format": "xyxy_pixels",
        "categories": {"1": "text_field"},
        "pages": [
            {
                "page": 0,
                "width": 600,
                "height": 800,
                "fields": [
                    {
                        "category_id": 1,
                        "category": "text_field",
                        "bbox": [10, 20, 110, 60],
                        "area": 4000,
                        "id": 1,
                    }
                ],
            }
        ],
    }
    (root / f"{doc_name}.fields.json").write_text(json.dumps(sidecar))


def _doc(doc_name: str = "form1") -> ET.Element:
    return ET.Element("document", {"name": doc_name})


def test_task_name():
    assert FormFillingComposer.task_name == "form_filling"


def test_emits_fields_from_sidecar(tmp_path: Path):
    _write_sidecar(tmp_path, "form1")
    comp = FormFillingComposer(sidecar_root=tmp_path)
    samples = comp.compose_document(_doc("form1"), tmp_path)
    assert len(samples) == 1
    fields = samples[0].payload["target.json"]["fields"]
    assert fields == [{"category": "text_field", "bbox": [10, 20, 110, 60]}]
    assert samples[0].payload["json"]["page_idx"] == 0


def test_missing_sidecar_yields_no_samples(tmp_path: Path):
    comp = FormFillingComposer(sidecar_root=tmp_path)
    assert comp.compose_document(_doc("absent"), tmp_path) == []


def test_no_sidecar_root_yields_no_samples(tmp_path: Path):
    assert FormFillingComposer().compose_document(_doc("form1"), tmp_path) == []
