"""Unit tests for figure_classifier.py."""

import re

from src.figure_classifier import LABEL_REGISTRY

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
