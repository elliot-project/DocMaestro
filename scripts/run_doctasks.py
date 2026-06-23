"""Run all registered DocTask composers over every XML in the output/ folder.

Scans ``output/`` for ``*_*.xml`` files (one per document), parses each with
xml.etree.ElementTree, runs every composer registered in
``src.task_composer.registry``, and writes the collected samples as a single
``<doc_name>_doctasks.json`` file next to the source XML.

Usage:
    uv run scripts/run_doctasks.py
    uv run scripts/run_doctasks.py --output-dir output --tasks basic_reading ocr layout
    uv run scripts/run_doctasks.py --xml output/adidas/adidas_hybrid-auto-engine.xml
    uv run scripts/run_doctasks.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.task_composer import registry  # noqa: E402

# ── Helpers ──────────────────────────────────────────────────────────────────


def find_xmls(output_dir: Path) -> list[Path]:
    """Discover all parsed document XML files under *output_dir*.

    Returns one XML per document: when multiple backends exist for the same
    doc, prefers ``hybrid-auto-engine`` then ``pipeline``, then the first found.

    Args:
        output_dir: Root output directory.

    Returns:
        Sorted list of XML paths, one per document folder.
    """
    result: list[Path] = []
    for doc_dir in sorted(output_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        xmls = sorted(doc_dir.rglob("*.xml"))
        if not xmls:
            continue
        # Prefer hybrid > pipeline > first
        preferred = (
            next((x for x in xmls if "hybrid-auto-engine" in x.name), None)
            or next((x for x in xmls if "pipeline" in x.name), None)
            or xmls[0]
        )
        result.append(preferred)
    return result


def process_xml(xml_path: Path, task_names: list[str], seed: int = 42) -> dict:
    """Parse one document XML and run all requested composers.

    Args:
        xml_path: Path to the ``<doc_name>_<backend>.xml`` file.
        task_names: Which task names from the registry to run.
        seed: RNG seed forwarded to seeded composers.

    Returns:
        Dict with keys ``doc_name``, ``xml_path``, ``tasks`` (dict of
        task_name → list of sample payloads).
    """
    root = ET.parse(xml_path).getroot()
    img_root = xml_path.parent

    tasks_out: dict[str, list[dict]] = {}
    for task_name in task_names:
        try:
            spec = registry.REGISTRY[task_name]
            # Merge seed into default config where supported
            cfg = {**spec.default_config}
            if "seed" in cfg:
                cfg["seed"] = seed
            composer = spec.composer_factory(**cfg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                samples = composer.compose_document(root, img_root)
            tasks_out[task_name] = [s.payload for s in samples]
        except Exception as exc:  # noqa: BLE001
            tasks_out[task_name] = [{"__error__": str(exc)}]

    return {
        "doc_name": root.attrib.get("name", xml_path.stem),
        "xml_path": str(xml_path),
        "page_count": int(root.attrib.get("page_count", 0)),
        "tasks": tasks_out,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Run DocTask composers over all parsed XMLs in output/."
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Root output directory to scan. Default: output/",
    )
    p.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="Process a single XML file instead of scanning --output-dir.",
    )
    p.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        metavar="TASK",
        help="Task names to run. Default: all registered tasks.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only process the first N documents.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for seeded composers. Default: 42.",
    )
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    all_tasks = [s.task_name for s in registry.list_tasks()]
    task_names = args.tasks if args.tasks else all_tasks

    # Validate task names
    unknown = set(task_names) - set(all_tasks)
    if unknown:
        print(f"ERROR: unknown tasks: {', '.join(sorted(unknown))}")
        print(f"Available: {', '.join(all_tasks)}")
        sys.exit(1)

    if args.xml:
        xmls = [args.xml]
    else:
        xmls = find_xmls(args.output_dir)

    if args.limit:
        xmls = xmls[: args.limit]

    print(f"DocTask Runner — {len(xmls)} document(s), {len(task_names)} task(s)")
    print(f"Tasks: {', '.join(task_names)}\n")

    manifest: list[dict] = []  # index written to output/doctasks_index.json

    for i, xml_path in enumerate(xmls, 1):
        print(
            f"[{i:3d}/{len(xmls)}] {xml_path.parent.name}/{xml_path.name} ...", end=" ", flush=True
        )
        try:
            result = process_xml(xml_path, task_names, seed=args.seed)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}")
            continue

        total_samples = sum(len(v) for v in result["tasks"].values())
        print(f"→ {total_samples} samples across {result['page_count']} pages")

        # Write per-doc JSON next to the XML
        out_path = xml_path.parent / f"{result['doc_name']}_doctasks.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

        manifest.append(
            {
                "doc_name": result["doc_name"],
                "page_count": result["page_count"],
                "json_path": str(out_path),
                "sample_counts": {t: len(v) for t, v in result["tasks"].items()},
            }
        )

    # Write global index consumed by the web viewer
    index_path = args.output_dir / "doctasks_index.json"
    index_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    total = sum(sum(e["sample_counts"].values()) for e in manifest)
    print(f"\n✓ Done. {total} total samples. Index written to {index_path}")
    print("  Open the viewer: output/doctasks_viewer/index.html")


if __name__ == "__main__":
    main()
