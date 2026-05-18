"""Zero-shot figure classification using SigLIP2 text-image embeddings.

This module provides a registry of figure categories with verbose, natural-language
descriptions designed for zero-shot classification. Each label maps to a detailed
description fed directly to SigLIP2's text encoder to maximize cosine similarity
separation between categories.
"""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from transformers import AutoModel, AutoProcessor

# Maps label slugs (kebab-case) to verbose natural-language descriptions fed directly
# to the SigLIP2 text encoder. Descriptions are intentionally long and specific to
# improve zero-shot cosine similarity separation between categories.
LABEL_REGISTRY: dict[str, str] = {
    "natural-image": (
        "a photograph or camera-captured image of a real-world scene, including outdoor"
        " landscapes, indoor environments, portraits of people, animals, everyday objects,"
        " food, architecture, or any scene recorded by a physical lens without synthetic"
        " construction"
    ),
    "diagram": (
        "a technical or conceptual diagram such as a flowchart, block diagram, system"
        " architecture schematic, UML diagram, neural network topology, pipeline"
        " illustration, or process map — composed primarily of boxes, arrows, nodes, and"
        " labeled connectors on a plain or minimal background"
    ),
    "scientific-medical-image": (
        "a scientific or biomedical imaging artifact such as a microscopy scan (light,"
        " electron, fluorescence), MRI or CT slice, X-ray, histology slide, satellite or"
        " aerial remote sensing image, astronomical photograph, or any image produced by a"
        " specialized scientific instrument rather than a conventional camera"
    ),
    "illustration-artwork": (
        "a hand-drawn or digitally illustrated image including artwork, cartoons, concept"
        " art, graphic novel panels, decorative borders, icons drawn in illustrative style,"
        " clipart, or any image that is clearly synthetic and artistic rather than"
        " photographic or diagrammatic"
    ),
    "screenshot": (
        "a rasterized capture of a software interface, operating system desktop, web"
        " browser, mobile app screen, terminal or command-line output, IDE or code editor"
        " window, or any image that shows a digital user interface as it appeared on a screen"
    ),
    "infographic": (
        "a composed visual communication graphic that integrates text labels, icons,"
        " statistics, timelines, or illustrations to explain a concept or dataset —"
        " typically with a designed layout, brand colors, and mixed text-and-image elements"
        " aimed at a general audience"
    ),
    "logo-icon": (
        "a company logo, product brand mark, certification badge, institutional seal, app"
        " icon, or small symbolic graphic typically displayed at low resolution on a white"
        " or transparent background with minimal surrounding context"
    ),
    "map": (
        "a geographic, topographic, or spatial layout map including world or regional maps,"
        " city street maps, choropleth or heat maps, floor plans, site plans, or any image"
        " whose primary purpose is to encode spatial relationships between locations"
    ),
    "signature-handwriting": (
        "a handwritten signature, cursive handwriting sample, handwritten annotation,"
        " marginalia, or any image dominated by ink strokes produced by a human hand rather"
        " than printed or typeset text"
    ),
    "rasterized-table": (
        "a table or structured data grid that has been rendered as an image rather than"
        " encoded as HTML or text, showing rows, columns, cell borders, and tabular numeric"
        " or categorical content — visually distinct from charts and from text paragraphs"
    ),
}

_DEFAULT_MODEL = "google/siglip2-base-patch16-384"
_LABELS = list(LABEL_REGISTRY.keys())


class FigureClassifier:
    """Zero-shot figure classifier backed by SigLIP2.

    Loads the model once and pre-encodes all label descriptions into text
    features that are reused across every batch, avoiding redundant forward
    passes through the text encoder.

    Args:
        model_name: HuggingFace model ID for SigLIP2.
        device: PyTorch device string, e.g. "cuda" or "cpu".
        batch_size: Number of images per GPU forward pass.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cuda",
        batch_size: int = 32,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.text_features = self._encode_labels()

    def _encode_labels(self) -> torch.Tensor:
        """Pre-encode all label descriptions into a normalised [N x D] text feature matrix.

        Returns:
            Tensor of shape [len(LABEL_REGISTRY), embedding_dim] on self.device.
        """
        descriptions = list(LABEL_REGISTRY.values())
        inputs = self.processor(text=descriptions, return_tensors="pt", padding=True).to(
            self.device
        )
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return F.normalize(features, dim=-1)

    def classify_figures(
        self,
        figures: list[ET.Element],
        img_root: Path,
    ) -> dict[ET.Element, tuple[str, float | None]]:
        """Classify a list of <figure> ET elements using batched SigLIP2 inference.

        Figures without a resolvable image file receive category "unknown" and
        confidence None. A warning is emitted for each such figure.

        Args:
            figures: List of <figure> ET.Element objects from the parsed XML.
            img_root: Root directory used to resolve relative img src paths.

        Returns:
            Dict mapping each ET.Element to (label, confidence). Confidence is
            None when the label is "unknown".
        """
        results: dict[ET.Element, tuple[str, float | None]] = {}
        queue: list[tuple[ET.Element, Path]] = []

        for fig in figures:
            img_el = fig.find("img")
            if img_el is None or not img_el.get("src"):
                warnings.warn(
                    f"<figure bbox='{fig.get('bbox', 'unknown')}'> has no img_path — "
                    "setting category='unknown'",
                    stacklevel=2,
                )
                results[fig] = ("unknown", None)
                continue

            img_path = img_root / img_el.get("src")
            if not img_path.exists():
                warnings.warn(
                    f"Image file not found: {img_path} — setting category='unknown'",
                    stacklevel=2,
                )
                results[fig] = ("unknown", None)
                continue

            queue.append((fig, img_path))

        # Process queue in batches
        for i in range(0, len(queue), self.batch_size):
            batch = queue[i : i + self.batch_size]
            batch_figs = [item[0] for item in batch]
            batch_paths = [item[1] for item in batch]
            batch_results = self._run_batch(batch_paths)
            for fig, (label, conf) in zip(batch_figs, batch_results):
                results[fig] = (label, conf)

        return results

    def _run_batch(self, img_paths: list[Path]) -> list[tuple[str, float | None]]:
        """Run a single batched forward pass for a list of image paths.

        Images are loaded in parallel via ThreadPoolExecutor. A corrupt or
        unreadable image falls back to category "unknown" with confidence None.

        Args:
            img_paths: Absolute paths to image files for this batch.

        Returns:
            List of (label, confidence) tuples, one per input path.
        """

        def _load(p: Path) -> Image.Image | None:
            try:
                return Image.open(p).convert("RGB")
            except (UnidentifiedImageError, OSError) as exc:
                warnings.warn(f"Could not load image {p}: {exc} — skipping", stacklevel=1)
                return None

        with ThreadPoolExecutor() as pool:
            images_or_none = list(pool.map(_load, img_paths))

        valid_indices = [i for i, img in enumerate(images_or_none) if img is not None]
        valid_images = [images_or_none[i] for i in valid_indices]

        results: list[tuple[str, float | None]] = [("unknown", None)] * len(img_paths)

        if not valid_images:
            return results

        try:
            inputs = self.processor(images=valid_images, return_tensors="pt", padding=True).to(
                self.device
            )
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            image_features = F.normalize(image_features, dim=-1)
            scores = image_features @ self.text_features.T  # [B x N]
            probs = torch.sigmoid(scores)
            confidences, indices = probs.max(dim=-1)

            for out_idx, (conf, idx) in zip(
                valid_indices, zip(confidences.tolist(), indices.tolist())
            ):
                results[out_idx] = (_LABELS[idx], round(conf, 3))

        except torch.cuda.OutOfMemoryError:
            # Fallback to CPU. NOTE: This reloads the model from disk which is slow (~30-90s)
            # but is correct. On a healthy server this path should never be hit in production.
            warnings.warn("CUDA OOM during batch inference — falling back to CPU", stacklevel=2)
            cpu_clf = FigureClassifier(
                model_name=self.model_name,
                device="cpu",
                batch_size=max(1, len(valid_images) // 2),
            )
            fallback = cpu_clf._run_batch([img_paths[i] for i in valid_indices])
            for out_idx, res in zip(valid_indices, fallback):
                results[out_idx] = res

        return results
