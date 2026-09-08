"""DocMaestro — full pipeline orchestrator.

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

Worker mode
-----------
When invoked with ``--worker``, main.py becomes a long-lived per-GPU worker:
it reads absolute PDF paths from stdin (one per line), extracts each, and
emits a single ``@@OMNIDOC@@`` protocol line per document on stdout. All
human-readable output (logs, progress) goes to stderr so stdout carries only
the protocol. Worker mode never touches the manifest — only the parent
process (run_sharded.py) or standalone mode write it.

Usage:
    uv run main.py --pdf data/pdfs/report.pdf
    uv run main.py --pdf data/pdfs/ --config configs/extractor/fast.yaml
    uv run main.py --pdf data/pdfs/ --backend pipeline --batch-size 4
    uv run main.py --pdf data/pdfs/report.pdf --no-wandb
    uv run main.py --worker --pdf data/pdfs/ --input-root data/pdfs/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.extractor import ExtractionResult, batch_extract, extract  # noqa: E402
from src.figure_classifier import FigureClassifier  # noqa: E402
from src.pipeline.worker import worker_loop  # noqa: E402
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
from src.utils.manifest import Manifest, compute_run_id, manifest_path  # noqa: E402
from src.utils.paths import doc_rel_dir, output_xml_path  # noqa: E402
from src.xml_formatter import build_document_xml, save_document_xml  # noqa: E402

_DEFAULT_CONFIG = Path("configs/extractor/default.yaml")


# ── Logging noise suppression ────────────────────────────────────────────────


def _suppress_noisy_logging() -> None:
    """Suppress MinerU/vLLM internal logging clutter.

    MinerU uses loguru for DEBUG messages and tqdm for internal progress bars
    (Layout Predict, MFR Predict, OCR-det, Processing pages, etc.). These
    flood the terminal when running large jobs. We suppress them here so only
    our Rich progress bar is visible.
    """
    # Suppress loguru (MinerU's logger)
    try:
        from loguru import logger as loguru_logger
        loguru_logger.remove()  # remove default stderr sink
        # Re-add with WARNING level only so critical errors still surface
        loguru_logger.add(sys.stderr, level="WARNING")
    except ImportError:
        pass

    # Suppress standard library loggers from vLLM and MinerU
    for name in [
        "vllm",
        "vllm.engine",
        "vllm.worker",
        "vllm.core",
        "vllm.model_executor",
        "mineru",
        "mineru.backend",
        "mineru.cli",
        "mineru.utils",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Disable tqdm progress bars globally via environment variable.
    # MinerU's internal code checks this.
    os.environ["TQDM_DISABLE"] = "1"

    # Suppress vLLM verbose progress bars
    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"


# ── Stage helpers ─────────────────────────────────────────────────────────────


def save_raw_json(result: ExtractionResult, out_dir: Path, doc_name: str) -> None:
    """Persist MinerU's raw middle.json and content_list.json next to the XML.

    Args:
        result: ExtractionResult from extract() or batch_extract().
        out_dir: Directory to write the two JSON files into (typically the
            same directory as the synthesized XML for this document).
        doc_name: Document stem, used as the filename prefix.
    """
    (out_dir / f"{doc_name}_middle.json").write_text(
        json.dumps(result.middle_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / f"{doc_name}_content_list.json").write_text(
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


# ── Per-doc finalization ──────────────────────────────────────────────────────


def _finalize_document(
    pdf_path: Path,
    result: ExtractionResult,
    config: PipelineConfig,
    root_dir: Path,
    console,
    step: int = 0,
) -> None:
    """Save JSON, build XML, run figure-clf, and log WandB for one extracted doc.

    Never writes human-readable text to stdout directly — all such output
    goes through ``console.print`` so worker mode (which redirects this
    console to stderr) keeps stdout reserved for the protocol stream.

    Args:
        pdf_path: Source PDF path.
        result: ExtractionResult from extract() or batch_extract().
        config: Full pipeline configuration.
        root_dir: Base input directory to calculate relative path.
        console: Rich Console for human-readable output.
        step: WandB step index for the current run.
    """
    ext = config.extraction
    doc_name = pdf_path.stem

    # XML sits alongside MinerU's native output: <conference>/<year>/<doc>/<doc>_backend.xml
    rel_dir = doc_rel_dir(pdf_path, root_dir)
    xml_path = output_xml_path(config.paths.output_dir, rel_dir, doc_name, ext.backend)

    console.print(
        f"  [green]✓[/green] {doc_name}: {result.page_count} pages in {result.duration_seconds:.1f}s"
    )

    log_extraction_metrics(
        duration_seconds=result.duration_seconds,
        page_count=result.page_count,
        backend=ext.backend,
        pdf_path=pdf_path,
        content_item_count=len(result.content_list),
    )
    log_gpu_memory()

    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_str = build_document_xml(result.content_list, result.middle_json, doc_name)
    save_document_xml(xml_str, xml_path)
    log_xml_stats(xml_str, xml_path)
    save_raw_json(result, xml_path.parent, doc_name)

    run = wandb.run
    if run is not None:
        safe_name = f"{doc_name}-{ext.backend}"[:128]
        artifact = wandb.Artifact(
            name=safe_name,
            type="document-xml",
            description=f"Synthesized XML for {doc_name[:100]}... using {ext.backend}",
        )
        artifact.add_file(str(xml_path))
        run.log_artifact(artifact)

    if config.figure_classification.enabled:
        console.print(f"  [dim]Running figure classification for {doc_name}...[/dim]")
        img_root = _find_img_root(result.output_dir)
        _enrich_figures(xml_path, img_root, config.figure_classification)


# ── Entry point ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with pdf, config, no_wandb, batch_size, window_size, worker,
        input_root, and retry_failed fields.
    """
    parser = argparse.ArgumentParser(
        description="DocMaestro: full MinerU → XML → enrichment pipeline."
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
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Run as a long-lived per-GPU worker: read PDF paths from stdin, "
             "emit @@OMNIDOC@@ protocol lines on stdout. Used by "
             "scripts/run_sharded.py; not intended for standalone use.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Original input directory (or single PDF), used by --worker to "
             "compute each document's output hierarchy via doc_rel_dir().",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-queue documents previously marked 'failed' in the manifest.",
    )
    return parser.parse_args()


def run_worker(args: argparse.Namespace, config: PipelineConfig) -> None:
    """Run as a long-lived per-GPU worker: stdin paths in, protocol lines out.

    Reads absolute PDF path lines from stdin until EOF, extracts each one on
    this process's GPU (pinned via CUDA_VISIBLE_DEVICES by the parent), and
    writes the synthesized XML and raw JSON next to it in the output tree.
    Exactly one ``@@OMNIDOC@@`` protocol line is written to stdout per
    document; everything else (MinerU/vLLM logs, the ``✓``/``✗`` summary
    lines from ``_finalize_document``) goes to stderr. The worker never
    touches the manifest — only the parent process writes it.

    Args:
        args: Parsed CLI arguments. ``args.input_root`` must be set; it is
            the original ``--pdf`` root the parent dispatcher collected PDFs
            from, needed to reconstruct each document's output hierarchy.
        config: Loaded pipeline configuration.
    """
    from rich.console import Console

    if args.input_root is None:
        raise SystemExit("--worker requires --input-root (the original --pdf input directory).")

    err_console = Console(file=sys.stderr)
    ext = config.extraction
    input_root = args.input_root

    if ext.window_size:
        os.environ["MINERU_PROCESSING_WINDOW_SIZE"] = str(ext.window_size)

    if not args.no_wandb and config.wandb_enabled:
        init_run(
            pdf_name=input_root.name if input_root else args.pdf.name,
            backend=ext.backend,
            extra_config={
                "language": ext.language,
                "window_size": ext.window_size,
                "title_aided": config.title_aided.enabled,
                "figure_classification": config.figure_classification.enabled,
                "worker": True,
            },
        )

    def process(path_str: str) -> dict:
        pdf_path = Path(path_str)
        rel_dir = doc_rel_dir(pdf_path, input_root)
        result = extract(
            pdf_path=pdf_path,
            output_dir=config.paths.data_dir / rel_dir.parent,
            backend=ext.backend,
            language=ext.language,
            start_page=ext.start_page,
            end_page=ext.end_page,
            server_url=ext.server_url,
            title_aided_config=config.title_aided.as_mineru_config(),
            gpu_memory_utilization=ext.gpu_memory_utilization,
        )
        _finalize_document(pdf_path, result, config, input_root, console=err_console)
        return {"pages": result.page_count, "seconds": result.duration_seconds}

    try:
        worker_loop(
            iter(sys.stdin.readline, ""),
            process,
            lambda line: print(line, flush=True),
        )
    finally:
        wandb.finish()


def main() -> None:
    """Orchestrate the full pipeline for all input PDFs."""
    # Suppress noise BEFORE importing/loading MinerU models
    _suppress_noisy_logging()

    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    args = parse_args()
    config = load_config(args.config)

    if args.no_wandb:
        config.wandb_enabled = False

    # CLI overrides for batch tuning
    if args.batch_size is not None:
        config.extraction.batch_size = args.batch_size
    if args.window_size is not None:
        config.extraction.window_size = args.window_size

    if args.worker:
        run_worker(args, config)
        return

    ext = config.extraction
    use_batching = ext.backend == "pipeline" and ext.batch_size > 1

    if ext.window_size:
        os.environ["MINERU_PROCESSING_WINDOW_SIZE"] = str(ext.window_size)

    console = Console()

    all_pdfs = collect_pdfs(args.pdf)

    # ── Resume logic via the manifest: skip 'done', optionally retry 'failed' ──
    run_id = compute_run_id(args.pdf, ext.backend, config.paths.output_dir)
    manifest = Manifest(manifest_path(config.paths.state_dir, run_id))

    doc_ids = [doc_rel_dir(p, args.pdf).as_posix() for p in all_pdfs]
    pending_ids = set(manifest.pending(doc_ids, retry_failed=args.retry_failed))
    pdfs = [p for p, doc_id in zip(all_pdfs, doc_ids) if doc_id in pending_ids]

    total_found = len(all_pdfs)
    skipped = total_found - len(pdfs)

    console.rule("[bold blue]DocMaestro[/bold blue]")
    console.print(f"  Total PDFs found:   [bold]{total_found}[/bold]")
    if skipped:
        console.print(f"  Already processed:  [yellow]{skipped} (skipped)[/yellow]")
    console.print(f"  Remaining to do:    [bold green]{len(pdfs)}[/bold green]")
    console.print(f"  Backend:            {ext.backend}")
    console.print(f"  GPU mem utilization: {ext.gpu_memory_utilization or 'default'}")
    console.print(f"  Window size:        {ext.window_size or 'default'}")
    console.print()

    if not pdfs:
        console.print("[bold green]✓ All documents have been processed. Nothing to do.[/bold green]")
        return

    if ext.batch_size > 1 and ext.backend != "pipeline":
        console.print(
            f"  [yellow]⚠ batch_size={ext.batch_size} ignored — "
            f"batching only improves throughput for the pipeline backend.[/yellow]\n"
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

    failed_docs: list[tuple[str, str]] = []

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            if use_batching:
                # ── Batched pipeline path ─────────────────────────────────────
                task = progress.add_task(
                    "[cyan]Extracting (batch)...", total=len(pdfs)
                )
                for chunk in chunked(pdfs, ext.batch_size):
                    try:
                        chunk_results = batch_extract(
                            pdf_paths=chunk,
                            output_dir=config.paths.data_dir,
                            backend=ext.backend,
                            language=ext.language,
                            gpu_memory_utilization=ext.gpu_memory_utilization,
                        )
                    except Exception as exc:
                        for p in chunk:
                            failed_docs.append((p.stem, str(exc)))
                            manifest.mark_failed(doc_rel_dir(p, args.pdf).as_posix(), str(exc))
                            progress.console.print(
                                f"  [red]✗[/red] {p.stem}: {exc}"
                            )
                        progress.advance(task, advance=len(chunk))
                        continue

                    for pdf_path, result in zip(chunk, chunk_results):
                        doc_id = doc_rel_dir(pdf_path, args.pdf).as_posix()
                        if isinstance(result, Exception):
                            failed_docs.append((pdf_path.stem, str(result)))
                            manifest.mark_failed(doc_id, str(result))
                            progress.console.print(
                                f"  [red]✗[/red] {pdf_path.stem}: {result}"
                            )
                            wandb.log({"error": str(result), "doc": pdf_path.name})
                        else:
                            _finalize_document(
                                pdf_path, result, config, args.pdf,
                                console=progress.console,
                            )
                            manifest.mark_done(
                                doc_id, pages=result.page_count, seconds=result.duration_seconds
                            )
                    progress.advance(task, advance=len(chunk))
            else:
                # ── Per-document path (hybrid / VLM / single-doc pipeline) ───
                task = progress.add_task(
                    "[cyan]Extracting...", total=len(pdfs)
                )
                for i, pdf_path in enumerate(pdfs):
                    doc_name = pdf_path.stem
                    rel_dir = doc_rel_dir(pdf_path, args.pdf)
                    doc_id = rel_dir.as_posix()

                    progress.update(task, description=f"[cyan]{doc_name[:60]}...")

                    try:
                        # Pass the parent dir so MinerU creates <doc>/hybrid_auto/ inside
                        result = extract(
                            pdf_path=pdf_path,
                            output_dir=config.paths.data_dir / rel_dir.parent,
                            backend=ext.backend,
                            language=ext.language,
                            start_page=ext.start_page,
                            end_page=ext.end_page,
                            server_url=ext.server_url,
                            title_aided_config=config.title_aided.as_mineru_config(),
                            gpu_memory_utilization=ext.gpu_memory_utilization,
                        )
                    except Exception as exc:
                        failed_docs.append((doc_name, str(exc)))
                        manifest.mark_failed(doc_id, str(exc))
                        progress.console.print(
                            f"  [red]✗[/red] {doc_name}: {exc}"
                        )
                        progress.advance(task)
                        continue

                    _finalize_document(
                        pdf_path, result, config, args.pdf,
                        console=progress.console, step=i,
                    )
                    manifest.mark_done(doc_id, pages=result.page_count, seconds=result.duration_seconds)
                    progress.advance(task)

    finally:
        wandb.finish()

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold blue]Summary[/bold blue]")
    console.print(f"  Processed: [green]{len(pdfs) - len(failed_docs)}[/green]")
    if failed_docs:
        console.print(f"  Failed:    [red]{len(failed_docs)}[/red]")
        for name, reason in failed_docs:
            console.print(f"    [red]✗[/red] {name}: {reason[:120]}")
    else:
        console.print("  [bold green]All documents processed successfully![/bold green]")


if __name__ == "__main__":
    main()
