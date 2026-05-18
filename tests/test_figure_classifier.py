"""Unit tests for figure_classifier.py."""

import math
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from src.figure_classifier import _DEFAULT_MODEL, FigureClassifier
from src.utils.labels import LABEL_REGISTRY

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

    def test_descriptions_are_lists_of_non_empty_strings(self):
        for label, descriptions in LABEL_REGISTRY.items():
            assert isinstance(descriptions, list), f"{label} value must be a list"
            assert len(descriptions) >= 1, f"{label} must have at least one description"
            for desc in descriptions:
                assert isinstance(desc, str), f"{label} contains a non-string description"
                assert len(desc) > 20, f"{label} has a description that is too short"

    def test_descriptions_are_unique(self):
        all_descriptions = [desc for descs in LABEL_REGISTRY.values() for desc in descs]
        assert len(set(all_descriptions)) == len(all_descriptions), "Duplicate descriptions found"

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


def _make_figure_with_file(tmp_path: Path, filename: str) -> ET.Element:
    """Create a <figure> element and write a 1x1 pixel JPEG at the expected path."""
    (tmp_path / "images").mkdir(exist_ok=True)
    img = Image.new("RGB", (4, 4), color=(128, 64, 32))
    img.save(tmp_path / "images" / filename)
    return _make_figure(img_src=f"images/{filename}")


class TestFallbacks:
    def _make_classifier(self) -> FigureClassifier:
        """Instantiate FigureClassifier with the model loading mocked out."""
        with (
            patch("src.figure_classifier.AutoProcessor.from_pretrained"),
            patch("src.figure_classifier.AutoModel.from_pretrained") as mock_model_cls,
        ):
            mock_model = MagicMock()
            mock_model.get_text_features.return_value = torch.zeros(50, 768)
            # Chain: from_pretrained(...).to(...).eval() must return mock_model
            mock_model_cls.return_value.to.return_value.eval.return_value = mock_model
            return FigureClassifier(device="cpu", batch_size=4)

    def test_no_img_child_sets_unknown(self, tmp_path):
        clf = self._make_classifier()
        figure = _make_figure(img_src=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = clf.classify_figures([figure], img_root=tmp_path)
        assert result[figure][0] == "unknown"
        assert any(
            "img src" in str(w.message) or "no image" in str(w.message).lower() for w in caught
        )

    def test_missing_file_sets_unknown(self, tmp_path):
        clf = self._make_classifier()
        figure = _make_figure(img_src="images/does_not_exist.jpg")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = clf.classify_figures([figure], img_root=tmp_path)
        assert result[figure][0] == "unknown"
        assert len(caught) == 1
        assert "not found" in str(caught[0].message).lower()

    def test_unknown_has_no_confidence(self, tmp_path):
        clf = self._make_classifier()
        figure = _make_figure(img_src=None)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = clf.classify_figures([figure], img_root=tmp_path)
        label, confidence = result[figure]
        assert label == "unknown"
        assert confidence is None

    def test_corrupt_image_sets_unknown(self, tmp_path):
        """A file that exists but contains garbage bytes falls back to unknown."""
        (tmp_path / "images").mkdir(exist_ok=True)
        corrupt_path = tmp_path / "images" / "corrupt.jpg"
        corrupt_path.write_bytes(b"not an image at all \x00\xff")
        fig = _make_figure(img_src="images/corrupt.jpg")
        clf = self._make_classifier()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = clf.classify_figures([fig], img_root=tmp_path)
        assert result[fig][0] == "unknown"
        assert result[fig][1] is None
        assert len(caught) >= 1

    def test_cuda_oom_emits_warning(self, tmp_path):
        """When CUDA OOM occurs inside _run_batch, a warning is emitted.

        We exercise the real _run_batch code path by patching the model's
        get_image_features to raise OOM on the first call, then patching the
        FigureClassifier constructor so the CPU fallback instantiation is cheap
        (no disk I/O) and its _run_batch returns a canned result.
        """
        fig = _make_figure_with_file(tmp_path, "fig_oom.jpg")
        clf = self._make_classifier()

        # The CPU-fallback FigureClassifier created inside _run_batch's except block
        # also needs to be cheap. We intercept its construction and wire in a mock
        # _run_batch so it returns a canned result instead of loading anything from disk.
        def patched_init(self_inner, model_name=None, device="cuda", batch_size=32):
            # Minimal state needed by _run_batch
            self_inner.model_name = model_name or _DEFAULT_MODEL
            self_inner.device = device
            self_inner.batch_size = batch_size
            self_inner.processor = MagicMock()
            mock_model = MagicMock()
            mock_model.get_image_features.return_value = torch.zeros(1, 768)
            self_inner.model = mock_model
            self_inner.text_features = torch.zeros(10, 768)

        # Make the primary clf's model raise OOM on get_image_features
        clf.model.get_image_features.side_effect = torch.cuda.OutOfMemoryError("out of memory")

        with (
            patch("src.figure_classifier.FigureClassifier.__init__", patched_init),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            try:
                clf.classify_figures([fig], img_root=tmp_path)
            except Exception:
                pass  # If CPU path also errors, we still want to check the warning

        oom_warnings = [
            w
            for w in caught
            if "oom" in str(w.message).lower() or "fallback" in str(w.message).lower()
        ]
        assert len(oom_warnings) >= 1


class TestBatchChunking:
    def _make_classifier_with_spy(self, batch_size: int) -> FigureClassifier:
        """Return a FigureClassifier whose _run_batch is replaced with a spy."""
        with (
            patch("src.figure_classifier.AutoProcessor.from_pretrained"),
            patch("src.figure_classifier.AutoModel.from_pretrained") as mock_model_cls,
        ):
            mock_model = MagicMock()
            mock_model.get_text_features.return_value = torch.zeros(50, 768)
            mock_model_cls.return_value.to.return_value.eval.return_value = mock_model
            clf = FigureClassifier(device="cpu", batch_size=batch_size)
        clf._run_batch = MagicMock(side_effect=lambda paths: [("diagram", 0.9)] * len(paths))
        return clf

    def test_single_batch_when_few_images(self, tmp_path):
        clf = self._make_classifier_with_spy(batch_size=32)
        figs = [_make_figure_with_file(tmp_path, f"img_{i}.jpg") for i in range(5)]
        clf.classify_figures(figs, img_root=tmp_path)
        assert clf._run_batch.call_count == 1

    def test_multiple_batches_when_many_images(self, tmp_path):
        clf = self._make_classifier_with_spy(batch_size=4)
        figs = [_make_figure_with_file(tmp_path, f"img_{i}.jpg") for i in range(10)]
        clf.classify_figures(figs, img_root=tmp_path)
        assert clf._run_batch.call_count == math.ceil(10 / 4)

    def test_batch_sizes_are_correct(self, tmp_path):
        clf = self._make_classifier_with_spy(batch_size=4)
        figs = [_make_figure_with_file(tmp_path, f"img_{i}.jpg") for i in range(9)]
        clf.classify_figures(figs, img_root=tmp_path)
        call_lengths = [len(c.args[0]) for c in clf._run_batch.call_args_list]
        assert call_lengths == [4, 4, 1]

    def test_results_mapped_to_correct_figures(self, tmp_path):
        """Results from distinct batches map to the correct figure elements."""
        clf = self._make_classifier_with_spy(batch_size=2)
        figs = [_make_figure_with_file(tmp_path, f"img_{i}.jpg") for i in range(4)]
        # Batch 1 returns "diagram", batch 2 returns "map"
        clf._run_batch = MagicMock(
            side_effect=[
                [("diagram", 0.9), ("diagram", 0.9)],
                [("map", 0.8), ("map", 0.8)],
            ]
        )
        result = clf.classify_figures(figs, img_root=tmp_path)
        assert result[figs[0]][0] == "diagram"
        assert result[figs[1]][0] == "diagram"
        assert result[figs[2]][0] == "map"
        assert result[figs[3]][0] == "map"

    def test_text_features_encoded_once(self, tmp_path):
        """get_text_features is called exactly once at init, not per batch."""
        with (
            patch("src.figure_classifier.AutoProcessor.from_pretrained"),
            patch("src.figure_classifier.AutoModel.from_pretrained") as mock_model_cls,
        ):
            mock_model = MagicMock()
            mock_model.get_text_features.return_value = torch.zeros(50, 768)
            mock_model_cls.return_value.to.return_value.eval.return_value = mock_model
            clf = FigureClassifier(device="cpu", batch_size=2)
            assert mock_model.get_text_features.call_count == 1
        # Run multiple batches; call count must stay at 1
        clf._run_batch = MagicMock(side_effect=lambda paths: [("diagram", 0.9)] * len(paths))
        figs = [_make_figure_with_file(tmp_path, f"img_{i}.jpg") for i in range(6)]
        clf.classify_figures(figs, img_root=tmp_path)
        assert mock_model.get_text_features.call_count == 1


class TestClassifyFiguresReturnDict:
    def _classify_with_mock(
        self,
        figures: list[ET.Element],
        img_root: Path,
        label: str = "diagram",
        conf: float = 0.9,
    ) -> dict[ET.Element, tuple[str, float | None]]:
        with (
            patch("src.figure_classifier.AutoProcessor.from_pretrained"),
            patch("src.figure_classifier.AutoModel.from_pretrained") as mock_model_cls,
        ):
            mock_model = MagicMock()
            mock_model.get_text_features.return_value = torch.zeros(50, 768)
            mock_model_cls.return_value.to.return_value.eval.return_value = mock_model
            clf = FigureClassifier(device="cpu", batch_size=32)
        clf._run_batch = MagicMock(side_effect=lambda paths: [(label, conf)] * len(paths))
        return clf.classify_figures(figures, img_root=img_root)

    def test_category_label_returned(self, tmp_path):
        fig = _make_figure_with_file(tmp_path, "fig.jpg")
        result = self._classify_with_mock([fig], tmp_path, label="diagram", conf=0.87)
        assert result[fig][0] == "diagram"

    def test_confidence_returned(self, tmp_path):
        fig = _make_figure_with_file(tmp_path, "fig.jpg")
        result = self._classify_with_mock([fig], tmp_path, label="screenshot", conf=0.923)
        label, conf = result[fig]
        assert label == "screenshot"
        assert conf == pytest.approx(0.923)
