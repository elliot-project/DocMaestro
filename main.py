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
    console=None,
    step: int = 0,
) -> None:
    """Save JSON, build XML, run figure-clf, and log WandB for one extracted doc.

    Args:
        pdf_path: Source PDF path.
        result: ExtractionResult from extract() or batch_extract().
        config: Full pipeline configuration.
        root_dir: Base input directory to calculate relative path.
        console: Rich Console for output (falls back to print if None).
        step: WandB step index for the current run.
    """
    ext = config.extraction
    doc_name = pdf_path.stem

    if root_dir.is_file():
        rel_dir = Path(doc_name)
    else:
        try:
            rel_dir = pdf_path.relative_to(root_dir.resolve()).parent / doc_name
        except ValueError:
            rel_dir = Path(doc_name)

    # XML sits alongside MinerU's native output: <conference>/<year>/<doc>/<doc>_backend.xml
    xml_path = config.paths.output_dir / rel_dir / f"{doc_name}_{ext.backend}.xml"

    _log = console.print if console else print

    _log(f"  [green]✓[/green] {doc_name}: {result.page_count} pages in {result.duration_seconds:.1f}s")

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
        _log(f"  [dim]Running figure classification for {doc_name}...[/dim]")
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

    ext = config.extraction
    use_batching = ext.backend == "pipeline" and ext.batch_size > 1

    if ext.window_size:
        os.environ["MINERU_PROCESSING_WINDOW_SIZE"] = str(ext.window_size)

    console = Console()

    pdfs = collect_pdfs(args.pdf)

    # ── Resume logic: skip already-processed docs ──
    total_found = len(pdfs)
    skipped = 0
    unprocessed_pdfs = []
    for pdf_path in pdfs:
        doc_name = pdf_path.stem
        if args.pdf.is_file():
            rel_dir = Path(doc_name)
        else:
            try:
                rel_dir = pdf_path.relative_to(args.pdf.resolve()).parent / doc_name
            except ValueError:
                rel_dir = Path(doc_name)
        xml_path = config.paths.output_dir / rel_dir / f"{doc_name}_{ext.backend}.xml"
        if xml_path.exists():
            skipped += 1
        else:
            unprocessed_pdfs.append(pdf_path)

    pdfs = unprocessed_pdfs

    console.rule("[bold blue]OmniDocParser[/bold blue]")
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
                            progress.console.print(
                                f"  [red]✗[/red] {p.stem}: {exc}"
                            )
                        progress.advance(task, advance=len(chunk))
                        continue

                    for pdf_path, result in zip(chunk, chunk_results):
                        if isinstance(result, Exception):
                            failed_docs.append((pdf_path.stem, str(result)))
                            progress.console.print(
                                f"  [red]✗[/red] {pdf_path.stem}: {result}"
                            )
                            wandb.log({"error": str(result), "doc": pdf_path.name})
                        else:
                            _finalize_document(
                                pdf_path, result, config, args.pdf,
                                console=progress.console,
                            )
                    progress.advance(task, advance=len(chunk))
            else:
                # ── Per-document path (hybrid / VLM / single-doc pipeline) ───
                task = progress.add_task(
                    "[cyan]Extracting...", total=len(pdfs)
                )
                for i, pdf_path in enumerate(pdfs):
                    doc_name = pdf_path.stem
                    if args.pdf.is_file():
                        rel_dir = Path(doc_name)
                    else:
                        try:
                            rel_dir = pdf_path.relative_to(args.pdf.resolve()).parent / doc_name
                        except ValueError:
                            rel_dir = Path(doc_name)

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
                        progress.console.print(
                            f"  [red]✗[/red] {doc_name}: {exc}"
                        )
                        progress.advance(task)
                        continue

                    _finalize_document(
                        pdf_path, result, config, args.pdf,
                        console=progress.console, step=i,
                    )
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
