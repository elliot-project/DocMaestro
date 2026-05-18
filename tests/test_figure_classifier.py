"""Unit tests for figure_classifier.py."""

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
            assert len(description) > 20, f"{label} description is too short"

    def test_unknown_not_in_registry(self):
        assert "unknown" not in LABEL_REGISTRY
