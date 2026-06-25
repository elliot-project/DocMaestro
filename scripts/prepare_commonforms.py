"""Materialize the CommonForms dataset into per-document PDFs + GT sidecars.

The CommonForms HuggingFace dataset stores one page image per row. Pages of the
same document share a root file name (``0005676-1.jpeg``, ``0005676-2.jpeg`` ...).
This script groups pages by document, assembles each document into a single
multi-page PDF (consumable by the existing MinerU pipeline), and writes the
ground-truth form-field annotations to an additive sidecar JSON.

The output PDF directory is the ``--pdf`` input for ``scripts/run_sharded.py``.

Example:
    uv run scripts/prepare_commonforms.py
    uv run scripts/prepare_commonforms.py --split valid --limit 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure the project root is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.commonforms import (  # noqa: E402
    build_fields_sidecar,
    build_pdf,
    group_pages,
    parse_file_name,
)

_REPO_ID = "jbarrow/CommonForms"
_DEFAULT_ROOT = Path("/data/151-1/datasets/CommonForms")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to ``path`` atomically (temp file in the same dir, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, obj: object) -> None:
    _atomic_write_bytes(path, json.dumps(obj, indent=2).encode("utf-8"))


def _load_dataset(split: str, cache_dir: Path):
    """Load the CommonForms split as a memory-mapped Dataset with raw image bytes.

    ``Image(decode=False)`` keeps the original JPEG bytes so they can be embedded
    into the PDF losslessly instead of being decoded to a PIL image.
    """
    from datasets import Image, load_dataset

    ds = load_dataset(_REPO_ID, split=split, cache_dir=str(cache_dir))
    return ds.cast_column("image", Image(decode=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize CommonForms into per-document PDFs + GT sidecars."
    )
    parser.add_argument(
        "--root", type=Path, default=_DEFAULT_ROOT, help=f"Output root (default: {_DEFAULT_ROOT})."
    )
    parser.add_argument(
        "--split", default="valid", help="Dataset split to materialize (default: valid)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N documents (smoke testing).",
    )
    args = parser.parse_args()

    raw_dir = args.root / "raw"
    pdf_dir = args.root / "pdfs"
    fields_dir = args.root / "fields"
    for d in (raw_dir, pdf_dir, fields_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading {_REPO_ID} split={args.split} (cache: {raw_dir}) ...")
    ds = _load_dataset(args.split, raw_dir)

    # Build a lightweight index (file_name + row position), then group by document.
    file_names = ds["file_name"]
    records = [{"file_name": fn, "_idx": i} for i, fn in enumerate(file_names)]
    grouped = group_pages(records)
    doc_ids = sorted(grouped)
    if args.limit is not None:
        doc_ids = doc_ids[: args.limit]

    print(f"{len(file_names)} pages → {len(grouped)} documents (processing {len(doc_ids)}).")

    built = skipped = 0
    for n, doc_id in enumerate(doc_ids, start=1):
        pdf_path = pdf_dir / f"{doc_id}.pdf"
        sidecar_path = fields_dir / f"{doc_id}.fields.json"
        if pdf_path.exists() and sidecar_path.exists():
            skipped += 1
            continue

        images: list[tuple[bytes, int, int]] = []
        pages: list[dict] = []
        for record in grouped[doc_id]:
            row = ds[record["_idx"]]
            _, page_num = parse_file_name(record["file_name"])
            width, height = int(row["width"]), int(row["height"])
            images.append((row["image"]["bytes"], width, height))
            pages.append(
                {
                    "page": page_num,
                    "width": width,
                    "height": height,
                    "objects": row["objects"],
                }
            )

        _atomic_write_bytes(pdf_path, build_pdf(images))
        _atomic_write_json(sidecar_path, build_fields_sidecar(doc_id, pages))
        built += 1
        if n % 100 == 0 or n == len(doc_ids):
            print(f"  [{n}/{len(doc_ids)}] built={built} skipped={skipped}")

    print(f"Done. built={built} skipped={skipped}")
    print(f"PDFs:   {pdf_dir}")
    print(f"Fields: {fields_dir}")
    print("Next: run extraction with")
    print(
        f"  uv run scripts/run_sharded.py --gpus 6,7 --pdf {pdf_dir} "
        f"--config configs/extractor/commonforms.yaml"
    )


if __name__ == "__main__":
    main()
