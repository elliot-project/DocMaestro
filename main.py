"""OmniDocParser — full pipeline orchestrator.

Runs MinerU extraction, XML synthesis, and optional post-processing stages
(title-aided hierarchy detection, SigLIP2 figure classification) for one PDF
or a directory of PDFs, driven by a YAML config file.

Batching (pipeline backend only)
---------------------------------
When backend=pipeline and batch_size > 1, all PDFs in a chunk are passed to
a single do_parse call so MinerU's streaming window batches pages across docs
simultaneously — improving GPU utilisation vs. one-at-a-time calls.
Batching has no effect on VLM/hybrid backends (they loop docs internally).
Title-aided detection requires batch_size=1 (hybrid backend only).

Set window_size (or MINERU_PROCESSING_WINDOW_SIZE) when
batch_size × avg_pages_per_doc exceeds the default window of 64.

Usage:
    uv run main.py --pdf data/pdfs/report.pdf
    uv run main.py --pdf data/pdfs/ --config configs/extractor/fast.yaml
    uv run main.py --pdf data/pdfs/ --backend pipeline --batch-size 4
    uv run main.py --pdf data/pdfs/report.pdf --no-wandb
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.extractor import ExtractionResult, batch_extract, extract  # noqa: E402
from src.figure_classifier import FigureClassifier  # noqa: E402
from src.utils.benchmark import (  # noqa: E402
    init_run,
    log_extraction_metrics,
    log_gpu_memory,
    log_xml_stats,
)
from src.utils.config import (  # noqa: E402
    FigureClassificationConfig,
    PipelineConfig,
    chunked,
    collect_pdfs,
    load_config,
)
from src.xml_formatter import build_document_xml, save_document_xml  # noqa: E402

_DEFAULT_CONFIG = Path("configs/extractor/default.yaml")


# ── Stage helpers ─────────────────────────────────────────────────────────────


def _save_raw_json(result, data_dir: Path, doc_name: str, backend: str) -> None:
    """Persist MinerU's middle.json and content_list.json to data/."""
    raw_dir = data_dir / doc_name / backend
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{doc_name}_middle.json").write_text(
        json.dumps(result.middle_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (raw_dir / f"{doc_name}_content_list.json").write_text(
        json.dumps(result.content_list, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _find_img_root(extraction_output_dir: Path) -> Path:
    """Locate the directory that contains MinerU's images/ subfolder.

    MinerU nests its image output inside a backend-specific subdirectory
    (e.g. data/<doc>/<backend>/<doc>/hybrid_auto/images/). img_path values
    in content_list.json are relative to that parent, so we search for the
    deepest images/ directory rather than assuming a fixed structure.

    Args:
        extraction_output_dir: The output_dir passed to extract().

    Returns:
        Parent of the images/ folder, or extraction_output_dir as fallback.
    """
    candidates = sorted(
        extraction_output_dir.rglob("images"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].parent if candidates else extraction_output_dir


def _enrich_figures(xml_path: Path, img_root: Path, cfg: FigureClassificationConfig) -> None:
    """Run SigLIP2 zero-shot classification and write category attributes in-place."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    figures = root.findall(".//figure")
    if not figures:
        return

    clf = FigureClassifier(model_name=cfg.model, device=cfg.device, batch_size=cfg.batch_size)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = clf.classify_figures(figures, img_root=img_root)

    for w in caught:
        print(f"  [figure-clf] {w.message}", file=sys.stderr)

    for fig, (label, confidence) in results.items():
        fig.set("category", label)
        if confidence is not None:
            fig.set("category_confidence", f"{confidence:.3f}")

    ET.indent(root, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    print(f"  [figure-clf] Classified {len(figures)} figures → {xml_path.name}")


# ── Per-doc finalization ──────────────────────────────────────────────────────


def _finalize_document(
    pdf_path: Path,
    result: ExtractionResult,
    config: PipelineConfig,
    step: int = 0,
) -> None:
    """Save JSON, build XML, run figure-clf, and log WandB for one extracted doc.

    Args:
        pdf_path: Source PDF path.
        result: ExtractionResult from extract() or batch_extract().
        config: Full pipeline configuration.
        step: WandB step index for the current run.
    """
    ext = config.extraction
    doc_name = pdf_path.stem
    xml_path = config.paths.output_dir / doc_name / f"{doc_name}_{ext.backend}.xml"

    print(f"[{doc_name}] Extracted {result.page_count} pages in {result.duration_seconds:.1f}s")

    log_extraction_metrics(
        duration_seconds=result.duration_seconds,
        page_count=result.page_count,
        backend=ext.backend,
        pdf_path=pdf_path,
        content_item_count=len(result.content_list),
    )
    log_gpu_memory()
    _save_raw_json(result, config.paths.data_dir, doc_name, ext.backend)

    xml_str = build_document_xml(result.content_list, result.middle_json, doc_name)
    save_document_xml(xml_str, xml_path)
    log_xml_stats(xml_str, xml_path)
    print(f"[{doc_name}] XML saved: {xml_path}")

    run = wandb.run
    if run is not None:
        artifact = wandb.Artifact(
            name=f"{doc_name}-{ext.backend}",
            type="document-xml",
            description=f"Synthesized XML for {doc_name} using {ext.backend}",
        )
        artifact.add_file(str(xml_path))
        run.log_artifact(artifact)

    if config.figure_classification.enabled:
        print(f"[{doc_name}] Running figure classification")
        img_root = _find_img_root(result.output_dir)
        _enrich_figures(xml_path, img_root, config.figure_classification)


# ── Entry point ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with pdf, config, no_wandb, batch_size, and window_size fields.
    """
    parser = argparse.ArgumentParser(
        description="OmniDocParser: full MinerU → XML → enrichment pipeline."
    )
    parser.add_argument(
        "--pdf", type=Path, required=True, help="PDF file or directory of PDFs to process."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"YAML config file. Default: {_DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Documents per do_parse call (pipeline backend only). "
             "Overrides the value in the config file.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        metavar="N",
        help="MinerU page-window size (MINERU_PROCESSING_WINDOW_SIZE). "
             "Overrides the value in the config file.",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable WandB logging for this run."
    )
    return parser.parse_args()


def main() -> None:
    """Orchestrate the full pipeline for all input PDFs."""
    args = parse_args()
    config = load_config(args.config)

    if args.no_wandb:
        config.wandb_enabled = False

    # CLI overrides for batch tuning
    if args.batch_size is not None:
        config.extraction.batch_size = args.batch_size
    if args.window_size is not None:
        config.extraction.window_size = args.window_size

    ext = config.extraction
    use_batching = ext.backend == "pipeline" and ext.batch_size > 1

    if ext.window_size:
        os.environ["MINERU_PROCESSING_WINDOW_SIZE"] = str(ext.window_size)

    pdfs = collect_pdfs(args.pdf)
    print(
        f"Found {len(pdfs)} PDF(s) | backend={ext.backend} | "
        f"batch_size={ext.batch_size if use_batching else 1} | "
        f"figure-clf={'on' if config.figure_classification.enabled else 'off'} | "
        f"title-aided={'on' if config.title_aided.enabled else 'off'}"
    )
    if ext.batch_size > 1 and ext.backend != "pipeline":
        print(
            f"  [info] batch_size={ext.batch_size} ignored — "
            f"batching only improves throughput for the pipeline backend."
        )

    init_run(
        pdf_name=args.pdf.name,
        backend=ext.backend,
        extra_config={
            "language": ext.language,
            "batch_size": ext.batch_size if use_batching else 1,
            "window_size": ext.window_size,
            "title_aided": config.title_aided.enabled,
            "figure_classification": config.figure_classification.enabled,
            "total_pdfs": len(pdfs),
        },
    )

    try:
        if use_batching:
            # ── Batched pipeline path ─────────────────────────────────────────
            # batch_extract passes N PDFs to one do_parse call so the streaming
            # window can batch pages across docs simultaneously.
            for chunk in chunked(pdfs, ext.batch_size):
                print(f"\nBatch extracting: {[p.name for p in chunk]}")
                log_gpu_memory()
                chunk_results = batch_extract(
                    pdf_paths=chunk,
                    output_dir=config.paths.data_dir,
                    backend=ext.backend,
                    language=ext.language,
                )
                for pdf_path, result in zip(chunk, chunk_results):
                    if isinstance(result, Exception):
                        print(f"[{pdf_path.stem}] FAILED: {result}")
                        wandb.log({"error": str(result), "doc": pdf_path.name})
                        continue
                    _finalize_document(pdf_path, result, config)
        else:
            # ── Per-document path (hybrid / VLM / single-doc pipeline) ────────
            for i, pdf_path in enumerate(pdfs):
                doc_name = pdf_path.stem
                print(f"\n[{doc_name}] Extracting with backend={ext.backend}")
                log_gpu_memory()

                result = extract(
                    pdf_path=pdf_path,
                    output_dir=config.paths.data_dir / doc_name / ext.backend,
                    backend=ext.backend,
                    language=ext.language,
                    start_page=ext.start_page,
                    end_page=ext.end_page,
                    server_url=ext.server_url,
                    title_aided_config=config.title_aided.as_mineru_config(),
                )
                _finalize_document(pdf_path, result, config, step=i)

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
