"""Unit tests for figure_classifier.py."""

import re
import warnings
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import torch

from src.figure_classifier import LABEL_REGISTRY, FigureClassifier

EXPECTED_LABELS = {
    "natural-image",
    "diagram",
    "scientific-medical-image",
    "illustration-artwork",
    "screenshot",
    "infographic",
    "logo-icon",
    "map",
    "signature-handwriting",
    "rasterized-table",
}


class TestLabelRegistry:
    def test_all_labels_present(self):
        assert set(LABEL_REGISTRY.keys()) == EXPECTED_LABELS

    def test_descriptions_are_non_empty_strings(self):
        for label, description in LABEL_REGISTRY.items():
            assert isinstance(description, str), f"{label} description is not a string"
            assert len(description) > 100, f"{label} description is too short"

    def test_descriptions_are_unique(self):
        descriptions = list(LABEL_REGISTRY.values())
        assert len(set(descriptions)) == len(descriptions), "Duplicate descriptions found"

    def test_keys_are_kebab_case(self):
        pattern = re.compile(r"^[a-z][a-z0-9-]+$")
        for key in LABEL_REGISTRY:
            assert pattern.match(key), f"Key '{key}' is not valid kebab-case"


def _make_figure(img_src: str | None = None) -> ET.Element:
    """Build a minimal <figure> ET.Element, optionally with an <img src=...> child."""
    el = ET.Element("figure")
    el.set("bbox", "[0, 0, 100, 100]")
    if img_src is not None:
        img = ET.SubElement(el, "img")
        img.set("src", img_src)
    return el


class TestFallbacks:
    def _make_classifier(self) -> FigureClassifier:
        """Instantiate FigureClassifier with the model loading mocked out."""
        with patch("src.figure_classifier.AutoProcessor.from_pretrained"), \
             patch("src.figure_classifier.AutoModel.from_pretrained") as mock_model_cls:
            mock_model = MagicMock()
            mock_model.get_text_features.return_value = torch.zeros(10, 768)
            mock_model_cls.return_value = mock_model
            return FigureClassifier(device="cpu", batch_size=4)

    def test_no_img_child_sets_unknown(self, tmp_path):
        clf = self._make_classifier()
        figure = _make_figure(img_src=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = clf.classify_figures([figure], img_root=tmp_path)
        assert result[figure][0] == "unknown"
        assert any("img_path" in str(w.message) or "no image" in str(w.message).lower()
                   for w in caught)

    def test_missing_file_sets_unknown(self, tmp_path):
        clf = self._make_classifier()
        figure = _make_figure(img_src="images/does_not_exist.jpg")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = clf.classify_figures([figure], img_root=tmp_path)
        assert result[figure][0] == "unknown"
        assert len(caught) >= 1

    def test_unknown_has_no_confidence(self, tmp_path):
        clf = self._make_classifier()
        figure = _make_figure(img_src=None)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = clf.classify_figures([figure], img_root=tmp_path)
        label, confidence = result[figure]
        assert label == "unknown"
        assert confidence is None
