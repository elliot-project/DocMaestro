"""DocVQA-2026 extraction sweep.

Downloads the val + test splits of VLR-CVC/DocVQA-2026 from HuggingFace,
converts each multi-page image document to a PDF, then runs MinerU pipeline
extraction to produce structured XML output.

Note: DocVQA-2026 stores documents as image sequences — there are no native
PDFs in the dataset. Images are converted to lossless multi-page PDFs before
extraction so the existing pipeline can process them unchanged.

Batching strategy
-----------------
Documents are extracted in chunks of --batch-size (default 4).  Within each
chunk, a single do_parse call is issued so MinerU's streaming pipeline can
fill its page window (default 64, override via --window-size) from multiple
documents simultaneously, keeping the GPU busier than one-doc-at-a-time calls.

GPU note: batch_ratio (internal multiplier for layout/OCR batch sizes) is
auto-set by MinerU from available VRAM — nothing to configure here.
--gpu-memory-utilization is only relevant for VLM/hybrid backends (no-op for
pipeline, which uses CV models, not vLLM).

Outputs per document:
  data/docvqa2026/<doc_id>/<doc_id>.pdf           — converted source PDF
  data/docvqa2026/<doc_id>/auto/                  — raw MinerU JSON outputs
  data/docvqa2026/<doc_id>/<backend>/             — explicit JSON copies
  data/docvqa2026/<doc_id>/qa.json                — questions and answers
  output/docvqa2026/<doc_id>/<doc_id>_<backend>.xml

Usage:
    uv run scripts/run_docvqa2026.py
    uv run scripts/run_docvqa2026.py --limit 5 --no-wandb
    uv run scripts/run_docvqa2026.py --batch-size 8 --window-size 128
    uv run scripts/run_docvqa2026.py --server-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from PIL import Image

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extractor import SUPPORTED_BACKENDS, ExtractionResult, batch_extract  # noqa: E402
from src.utils.config import chunked  # noqa: E402
from src.xml_formatter import build_document_xml, save_document_xml  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────

_DATA_ROOT = Path("data/docvqa2026")
_OUTPUT_ROOT = Path("output/docvqa2026")
_DATASET_ID = "VLR-CVC/DocVQA-2026"
_SPLITS = ["val", "test"]

Image.MAX_IMAGE_PIXELS = None  # Disable PIL decompression bomb protection (DocVQA-2026 has some very large pages)
# ── Image → PDF conversion ────────────────────────────────────────────────────


def images_to_pdf_bytes(pages: list) -> bytes:
    """Convert a list of PIL images to a single multi-page PDF in memory.

    PIL's PDF writer requires RGB or L mode. RGBA and palette (P) images are
    converted to RGB before saving. Grayscale (L/LA) images are kept as L.

    Args:
        pages: List of PIL Image objects representing document pages.

    Returns:
        Raw PDF bytes of the assembled document.

    Raises:
        ValueError: If the pages list is empty.
    """
    if not pages:
        raise ValueError("Cannot create a PDF from an empty page list.")

    def _normalize(img):
        if img.mode in ("RGBA", "P", "LA"):
            return img.convert("RGB")
        if img.mode not in ("RGB", "L"):
            return img.convert("RGB")
        return img

    normalized = [_normalize(p) for p in pages]
    buf = io.BytesIO()
    normalized[0].save(
        buf,
        format="PDF",
        save_all=True,
        append_images=normalized[1:],
    )
    return buf.getvalue()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ensure_pdf(record: dict) -> Path:
    """Convert a DocVQA record's image pages to PDF, or reuse an existing one.

    Args:
        record: Dataset row with 'doc_id' and 'document' fields.

    Returns:
        Path to the (possibly newly created) PDF file.
    """
    doc_id: str = record["doc_id"]
    pdf_path = _DATA_ROOT / doc_id / f"{doc_id}.pdf"
    if pdf_path.exists():
        return pdf_path
    pages = record["document"]
    if not isinstance(pages, list):
        pages = [pages]
    (_DATA_ROOT / doc_id).mkdir(parents=True, exist_ok=True)
    print(f"  [convert] {doc_id}: {len(pages)} page(s) → PDF")
    pdf_path.write_bytes(images_to_pdf_bytes(pages))
    return pdf_path


def _finalize_document(
    record: dict,
    result: ExtractionResult,
    backend: str,
) -> dict:
    """Save raw JSON, XML, and QA data for one successfully extracted document.

    Args:
        record: Original dataset row (provides doc_id, questions, answers).
        result: ExtractionResult from batch_extract.
        backend: Backend name used (for file naming).

    Returns:
        Outcome dict with status, pages, duration_s, doc_category.
    """
    doc_id: str = record["doc_id"]
    doc_dir = _DATA_ROOT / doc_id
    out_dir = _OUTPUT_ROOT / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Raw JSON copies under data/<doc_id>/<backend>/
    raw_dir = doc_dir / backend
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{doc_id}_middle.json").write_text(
        json.dumps(result.middle_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (raw_dir / f"{doc_id}_content_list.json").write_text(
        json.dumps(result.content_list, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # XML synthesis
    xml_path = out_dir / f"{doc_id}_{backend}.xml"
    xml_str = build_document_xml(result.content_list, result.middle_json, doc_id)
    save_document_xml(xml_str, xml_path)
    print(f"  [xml]     {doc_id} → {xml_path}")

    # QA data
    (doc_dir / "qa.json").write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "doc_category": record.get("doc_category", ""),
                "questions": record.get("questions", {}),
                "answers": record.get("answers", {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "doc_id": doc_id,
        "status": "ok",
        "pages": result.page_count,
        "duration_s": result.duration_seconds,
        "doc_category": record.get("doc_category", ""),
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run MinerU pipeline extraction on the DocVQA-2026 dataset."
    )
    parser.add_argument(
        "--backend",
        default="pipeline",
        choices=sorted(SUPPORTED_BACKENDS),
        help="MinerU backend (default: pipeline).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        metavar="N",
        help="Documents per do_parse call (default 4). Higher values fill the "
             "GPU page window from more documents simultaneously. Increase "
             "--window-size if batch_size × avg_pages > 64.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        metavar="N",
        help="MinerU processing window in pages (MINERU_PROCESSING_WINDOW_SIZE). "
             "Default is 64. Set larger when batch_size × avg_pages exceeds 64.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        metavar="F",
        help="vLLM KV-cache fraction (e.g. 0.85). Only affects VLM/hybrid "
             "backends — no-op for pipeline.",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        metavar="URL",
        help="VLM server URL for http-client backends (e.g. http://localhost:8000).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N documents (useful for smoke-testing).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-process documents even if their XML output already exists.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging.",
    )
    return parser.parse_args()




def main() -> None:
    """Download DocVQA-2026, combine splits, and run batched extraction."""
    args = parse_args()

    # Set window size before MinerU initialises its pipeline
    if args.window_size:
        os.environ["MINERU_PROCESSING_WINDOW_SIZE"] = str(args.window_size)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading {_DATASET_ID} (splits: {', '.join(_SPLITS)}) …")
    splits = [load_dataset(_DATASET_ID, split=s) for s in _SPLITS]
    dataset = concatenate_datasets(splits)
    print(f"Combined dataset: {len(dataset)} documents")

    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"Limiting to {len(dataset)} document(s).")

    _DATA_ROOT.mkdir(parents=True, exist_ok=True)
    _OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ── WandB init ────────────────────────────────────────────────────────────
    if not args.no_wandb:
        wandb.init(
            project="mineru-xml-pipeline",
            name=f"docvqa2026-{args.backend}",
            config={
                "dataset": _DATASET_ID,
                "splits": _SPLITS,
                "backend": args.backend,
                "batch_size": args.batch_size,
                "window_size": args.window_size,
                "total_docs": len(dataset),
                "gpu_memory_utilization": args.gpu_memory_utilization,
            },
        )

    # ── Phase 1: ensure all PDFs exist ───────────────────────────────────────
    # Accessing dataset["doc_id"] returns the column as a plain list with zero
    # image decoding. We check PDF existence from that alone, then only load
    # full rows (which decode PIL images) for docs that actually need conversion.
    print("\n[Phase 1] Converting image documents to PDF …")
    doc_ids: list[str] = dataset["doc_id"]
    pdf_paths: dict[str, Path] = {}
    missing_indices: list[int] = []
    for i, doc_id in enumerate(doc_ids):
        pdf_path = _DATA_ROOT / doc_id / f"{doc_id}.pdf"
        pdf_paths[doc_id] = pdf_path
        if not pdf_path.exists():
            missing_indices.append(i)

    if missing_indices:
        print(f"  {len(missing_indices)} document(s) need PDF conversion …")
        for record in dataset.select(missing_indices):
            pdf_paths[record["doc_id"]] = _ensure_pdf(record)
    else:
        print("  All PDFs already exist, skipping conversion.")

    # Strip the image column before materializing for Phase 2: only metadata
    # fields (doc_id, questions, answers, doc_category) are needed from here on.
    records = list(dataset.remove_columns(["document"]))

    # ── Phase 2: batch extraction ─────────────────────────────────────────────
    resume = not args.no_resume
    results: list[dict] = []
    sweep_t0 = time.perf_counter()
    global_idx = 0

    print(f"\n[Phase 2] Extracting in chunks of {args.batch_size} …")
    for chunk in chunked(records, args.batch_size):
        # Apply resume filter: skip docs whose XML already exists
        pending = [
            r for r in chunk
            if not resume
            or not (_OUTPUT_ROOT / r["doc_id"] / f"{r['doc_id']}_{args.backend}.xml").exists()
        ]
        skipped_in_chunk = [r for r in chunk if r not in pending]

        for r in skipped_in_chunk:
            print(f"  [skip] {r['doc_id']} — XML already exists")
            results.append({"doc_id": r["doc_id"], "status": "skipped", "pages": 0, "duration_s": 0.0})
            global_idx += 1

        if not pending:
            continue

        chunk_pdfs = [pdf_paths[r["doc_id"]] for r in pending]
        doc_ids_str = ", ".join(r["doc_id"] for r in pending)
        print(f"\n[{global_idx + 1}–{global_idx + len(pending)}/{len(records)}] "
              f"batch_extract: {doc_ids_str}")

        chunk_results = batch_extract(
            pdf_paths=chunk_pdfs,
            output_dir=_DATA_ROOT,
            backend=args.backend,
            server_url=args.server_url,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        for record, extraction in zip(pending, chunk_results):
            doc_id = record["doc_id"]
            if isinstance(extraction, Exception):
                print(f"  [error] {doc_id}: {extraction}")
                outcome: dict = {
                    "doc_id": doc_id,
                    "status": "error",
                    "pages": 0,
                    "duration_s": 0.0,
                    "error": str(extraction),
                }
            else:
                spp = extraction.duration_seconds / max(extraction.page_count, 1)
                print(f"  [ok]    {doc_id}: {extraction.page_count}p, {spp:.3f} s/page")
                outcome = _finalize_document(record, extraction, args.backend)

            results.append(outcome)

            if not args.no_wandb and outcome["status"] != "skipped":
                log_payload: dict = {
                    "docvqa/doc_id": doc_id,
                    "docvqa/status": outcome["status"],
                    "docvqa/pages": outcome.get("pages", 0),
                    "docvqa/duration_s": outcome.get("duration_s", 0.0),
                }
                if outcome["status"] == "ok":
                    log_payload["docvqa/secs_per_page"] = (
                        outcome["duration_s"] / max(outcome["pages"], 1)
                    )
                    log_payload["docvqa/doc_category"] = outcome.get("doc_category", "")
                elif outcome["status"] == "error":
                    log_payload["docvqa/error"] = outcome.get("error", "")
                wandb.log(log_payload, step=global_idx)

            global_idx += 1

    sweep_duration = time.perf_counter() - sweep_t0

    # ── Summary ───────────────────────────────────────────────────────────────
    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "error"]
    total_pages = sum(r["pages"] for r in ok)

    print("\n── DocVQA-2026 Sweep Summary ──────────────────────────────────")
    print(f"  Total documents  : {len(results)}")
    print(f"  Succeeded        : {len(ok)}")
    print(f"  Skipped (resume) : {len(skipped)}")
    print(f"  Failed           : {len(errors)}")
    print(f"  Total pages      : {total_pages}")
    print(f"  Wall-clock total : {sweep_duration:.1f}s")
    if ok:
        mean_spp = sum(r["duration_s"] / max(r["pages"], 1) for r in ok) / len(ok)
        print(f"  Mean s/page      : {mean_spp:.4f}")
    if errors:
        print("\n  Failed documents:")
        for r in errors:
            print(f"    {r['doc_id']}: {r.get('error', '')}")

    summary_path = _OUTPUT_ROOT / f"sweep_summary_{args.backend}.json"
    summary_path.write_text(
        json.dumps(
            {
                "backend": args.backend,
                "batch_size": args.batch_size,
                "window_size": args.window_size,
                "total": len(results),
                "ok": len(ok),
                "skipped": len(skipped),
                "errors": len(errors),
                "total_pages": total_pages,
                "sweep_duration_s": sweep_duration,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  Summary saved: {summary_path}")

    if not args.no_wandb:
        wandb.log({
            "docvqa/sweep/total_docs": len(results),
            "docvqa/sweep/ok": len(ok),
            "docvqa/sweep/skipped": len(skipped),
            "docvqa/sweep/errors": len(errors),
            "docvqa/sweep/total_pages": total_pages,
            "docvqa/sweep/wall_clock_s": sweep_duration,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
