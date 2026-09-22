"""Post-processing pass: enrich <figure> elements in a parsed XML file with
SigLIP2 zero-shot classification labels.

Usage:
    uv run scripts/classify_figures.py --xml output/mydoc/mydoc_hybrid.xml
    uv run scripts/classify_figures.py --xml output/mydoc/mydoc_hybrid.xml \\
        --img-root output/mydoc --batch-size 16 --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import wandb
from doc_maestro.figure_classifier import FigureClassifier


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace with xml, img_root, batch_size, device, model, wandb fields.
    """
    parser = argparse.ArgumentParser(
        description="Enrich <figure> elements in an XML file with SigLIP2 category labels."
    )
    parser.add_argument("--xml", required=True, type=Path, help="Path to XML file to enrich.")
    parser.add_argument(
        "--img-root",
        type=Path,
        default=None,
        help="Root directory for resolving img src paths. Defaults to the XML file's parent.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Images per GPU batch.")
    parser.add_argument("--device", type=str, default="cuda", help="PyTorch device string.")
    parser.add_argument(
        "--model",
        type=str,
        default="google/siglip2-base-patch16-384",
        help="HuggingFace model ID for SigLIP2.",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging.")
    return parser.parse_args()


def _serialize_xml(tree: ET.ElementTree, xml_path: Path) -> None:
    """Write an indented XML tree to disk.

    Args:
        tree: The ElementTree to serialize.
        xml_path: Destination file path.
    """
    ET.indent(tree.getroot(), space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def _compute_stats(
    results: dict[ET.Element, tuple[str, float | None]],
) -> tuple[int, int, dict[str, int], list[float]]:
    """Compute classification summary statistics from classify_figures results.

    Args:
        results: Dict mapping ET.Element to (label, confidence | None).

    Returns:
        Tuple of (n_classified, n_unknown, label_counts, confidences).
    """
    label_counts: dict[str, int] = {}
    n_classified = 0
    n_unknown = 0
    confidences: list[float] = []
    for label, conf in results.values():
        if label == "unknown":
            n_unknown += 1
        else:
            n_classified += 1
            label_counts[label] = label_counts.get(label, 0) + 1
            if conf is not None:
                confidences.append(conf)
    return n_classified, n_unknown, label_counts, confidences


def main() -> None:
    """Run the figure classification post-processing pass."""
    args = parse_args()

    xml_path: Path = args.xml.resolve()
    if not xml_path.exists():
        sys.exit(f"XML file not found: {xml_path}")

    img_root: Path = (args.img_root or xml_path.parent).resolve()

    # Parse XML — raise immediately on malformed input
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        sys.exit(f"Malformed XML in {xml_path}: {exc}")
    root = tree.getroot()
    figures = root.findall(".//figure")

    print(f"Found {len(figures)} <figure> elements in {xml_path.name}")

    if not figures:
        print("Nothing to classify.")
        return

    clf = FigureClassifier(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )

    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        results = clf.classify_figures(figures, img_root=img_root)
    elapsed = time.perf_counter() - t0

    for w in caught_warnings:
        print(f"  WARNING: {w.message}", file=sys.stderr)

    # Write category and category_confidence attributes onto ET elements
    for fig, (label, confidence) in results.items():
        fig.set("category", label)
        if confidence is not None:
            fig.set("category_confidence", f"{confidence:.3f}")

    # Serialize back to same path with consistent pretty-printing
    _serialize_xml(tree, xml_path)

    # Compute summary stats
    n_classified, n_unknown, label_counts, confidences = _compute_stats(results)

    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    throughput = len(figures) / elapsed if elapsed > 0 else 0.0

    print(f"Classified {n_classified} figures, {n_unknown} unknown in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} img/s | Mean confidence: {mean_conf:.3f}")
    print(f"Label distribution: {label_counts}")
    print(f"Enriched XML written to {xml_path}")

    if not args.no_wandb:
        wandb.init(
            project="mineru-xml-pipeline",
            job_type="figure-classification",
            config={
                "model": args.model,
                "batch_size": args.batch_size,
                "device": args.device,
                "xml_path": str(xml_path),
            },
        )
        wandb.log(
            {
                "figures/total": len(figures),
                "figures/classified": n_classified,
                "figures/unknown": n_unknown,
                "figures/mean_confidence": mean_conf,
                "figures/inference_time_s": elapsed,
                "figures/throughput_img_per_s": throughput,
                "figures/label_distribution": wandb.plot.bar(
                    wandb.Table(
                        data=[[label, count] for label, count in label_counts.items()],
                        columns=["label", "count"],
                    ),
                    "label",
                    "count",
                    title="Figure Label Distribution",
                ),
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
