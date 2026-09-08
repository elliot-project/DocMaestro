# src/task_composer/tasks/form_filling.py
"""DocTask 11: form filling (key-value information extraction from forms).

Unlike the other composers, the ground truth here does not come from
DocMaestro XML. It is pulled from the pre-processed CommonForms partition's
field sidecar (built by ``src/datasets/commonforms.py::build_fields_sidecar``).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from src.task_composer.base import BaseTaskComposer, DocumentContext, Sample


class FormFillingComposer(BaseTaskComposer):
    """SSL task: extract structured form fields from a form page.

    Resolves ``{sidecar_root}/{doc_name}.fields.json`` and emits one sample per
    page that has fields. The target is the page's field structure
    (``{category, bbox}`` per field). Best-effort: with no ``sidecar_root`` or a
    missing file, yields nothing.

    Args:
        sidecar_root: Directory holding ``<doc_name>.fields.json`` sidecars.
    """

    task_name = "form_filling"

    def __init__(self, sidecar_root: Path | None = None) -> None:
        super().__init__()
        self.sidecar_root = Path(sidecar_root) if sidecar_root is not None else None

    def compose_page(self, page: ET.Element, context: DocumentContext) -> list[Sample]:
        """Unused — form filling reads a sidecar, not the XML page."""
        return []

    def compose_document(self, root: ET.Element, img_root: Path) -> list[Sample]:
        """Produce one KV sample per sidecar page that has form fields."""
        if self.sidecar_root is None:
            return []
        doc_name = root.attrib["name"]
        sidecar_path = self.sidecar_root / f"{doc_name}.fields.json"
        if not sidecar_path.exists():
            return []

        sidecar = json.loads(sidecar_path.read_text())
        samples: list[Sample] = []
        for page in sidecar.get("pages", []):
            fields = [
                {"category": f["category"], "bbox": f["bbox"]}
                for f in page.get("fields", [])
            ]
            if not fields:
                continue
            page_no = int(page["page"])
            key = f"{doc_name}_p{page_no:03d}_form_fill_0"
            payload: dict[str, Any] = {
                "__key__": key,
                "target.json": {"fields": fields},
                "json": {
                    "task_type": self.task_name,
                    "doc_name": doc_name,
                    "page_idx": page_no,
                    "width": page.get("width"),
                    "height": page.get("height"),
                },
            }
            samples.append(Sample(key=key, payload=payload))
        return samples
