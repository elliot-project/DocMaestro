"""CLI: generate layout-bbox, reading-order, and document-tree visualizations.

Usage:
    uv run scripts/visualize.py --pdf data/pdfs/macdonals.pdf
    uv run scripts/visualize.py --pdf data/pdfs/macdonals.pdf --backend pipeline --pages 0 1 2 5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visualizer import generate_all_visualizations  # noqa: E402


def load_content_list(data_dir: Path, doc_name: str) -> list[dict]:
    """Load content_list.json from the extracted data directory.

    Args:
        data_dir: Root data directory (e.g. data/).
        doc_name: Stem of the PDF filename.

    Returns:
        Parsed list from content_list.json.

    Raises:
        FileNotFoundError: If the JSON file cannot be located.
    """
    # Try the canonical location written by run_extraction.py
    candidate = data_dir / doc_name / "hybrid-auto-engine" / f"{doc_name}_content_list.json"
    if not candidate.exists():
        # Fall back to any backend subdirectory
        matches = list((data_dir / doc_name).rglob(f"{doc_name}_content_list.json"))
        if not matches:
            raise FileNotFoundError(
                f"No content_list.json found for '{doc_name}' under {data_dir}. "
                f"Run scripts/run_extraction.py first."
            )
        candidate = matches[0]

    print(f"  loading: {candidate}")
    return json.loads(candidate.read_text(encoding="utf-8"))


def main() -> None:
    """Entry point for the visualization CLI."""
    parser = argparse.ArgumentParser(
        description="Generate layout-bbox, reading-order, and document-tree visualizations."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Path to the source PDF (must have been extracted already).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Backend subdirectory to load data from. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--pages",
        type=int,
        nargs="*",
        default=None,
        help="Page indices to visualize (0-based). Default: first 3 pages with content.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root directory for extracted JSON data. Default: data/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Root directory for visualization outputs. Default: output/.",
    )
    parser.add_argument(
        "--dom",
        action="store_true",
        default=False,
        help="Also generate interactive DOM graph HTML files (full-doc + per page).",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"ERROR: PDF not found: {args.pdf}")
        sys.exit(1)

    doc_name = args.pdf.stem
    print(f"Visualizing: {doc_name}")

    if args.backend:
        cl_path = args.data_dir / doc_name / args.backend / f"{doc_name}_content_list.json"
        content_list = json.loads(cl_path.read_text(encoding="utf-8"))
        print(f"  loading: {cl_path}")
    else:
        content_list = load_content_list(args.data_dir, doc_name)

    print(
        f"  {len(content_list)} content blocks across "
        f"{len(set(x.get('page_idx', 0) for x in content_list))} pages"
    )

    saved = generate_all_visualizations(
        pdf_path=args.pdf,
        content_list=content_list,
        doc_name=doc_name,
        output_dir=args.output_dir,
        pages=args.pages,
        dom=args.dom,
    )

    print(f"\nDone. {len(saved)} files saved to {args.output_dir / doc_name / 'plots'}/")


if __name__ == "__main__":
    main()
