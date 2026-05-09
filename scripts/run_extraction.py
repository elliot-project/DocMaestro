"""Main orchestration script: PDF → MinerU extraction → XML synthesis → WandB logging.

Usage:
    uv run scripts/run_extraction.py --pdf data/pdfs/0001.pdf
    uv run scripts/run_extraction.py --pdf data/pdfs/0001.pdf --backend pipeline
    uv run scripts/run_extraction.py --pdf data/pdfs/ --backend hybrid-auto-engine
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

import wandb

# Reproducibility seeds (CPU-only; CUDA seeds are deferred to avoid initializing
# the CUDA context before vLLM's engine process can do so cleanly)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark import (  # noqa: E402
    init_run,
    log_extraction_metrics,
    log_gpu_memory,
    log_xml_stats,
)
from src.extractor import SUPPORTED_BACKENDS, extract  # noqa: E402
from src.xml_formatter import build_document_xml, save_document_xml  # noqa: E402


def collect_pdfs(input_path: Path) -> list[Path]:
    """Collect PDF paths from a file or directory.

    Args:
        input_path: Path to a single PDF or a directory of PDFs.

    Returns:
        Sorted list of PDF Paths.

    Raises:
        ValueError: If no PDFs are found.
    """
    if input_path.is_file():
        return [input_path]
    pdfs = sorted(input_path.glob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No PDF files found in {input_path}")
    return pdfs


def process_pdf(
    pdf_path: Path,
    backend: str,
    data_dir: Path,
    output_dir: Path,
    language: str,
    start_page: int,
    end_page: int | None,
    wandb_run: wandb.sdk.wandb_run.Run,
    server_url: str | None = None,
    title_aided_config: dict | None = None,
) -> Path:
    """Extract a single PDF, synthesize XML, and log metrics.

    Args:
        pdf_path: Path to the source PDF.
        backend: MinerU backend identifier.
        data_dir: Root directory for raw MinerU JSON output.
        output_dir: Root directory for synthesized XML output.
        language: OCR language hint.
        start_page: Zero-based first page index.
        end_page: Zero-based last page index (inclusive), or None for all.
        wandb_run: Active WandB run for artifact logging.
        server_url: VLM server URL for http-client backends.
        title_aided_config: Optional LLM config for H1–H4 hierarchy detection.

    Returns:
        Path to the saved XML file.
    """
    doc_name = pdf_path.stem
    mineru_output_dir = data_dir / doc_name / backend

    print(f"[{backend}] Extracting: {pdf_path.name}")
    log_gpu_memory()

    result = extract(
        pdf_path=pdf_path,
        output_dir=mineru_output_dir,
        backend=backend,
        language=language,
        start_page=start_page,
        end_page=end_page,
        server_url=server_url,
        title_aided_config=title_aided_config,
    )

    print(
        f"[{backend}] Done in {result.duration_seconds:.1f}s — "
        f"{result.page_count} pages, {len(result.content_list)} content blocks"
    )

    log_extraction_metrics(
        duration_seconds=result.duration_seconds,
        page_count=result.page_count,
        backend=backend,
        pdf_path=pdf_path,
        content_item_count=len(result.content_list),
    )
    log_gpu_memory()

    # Save raw JSON intermediates to data/
    raw_dir = mineru_output_dir
    middle_save_path = raw_dir / f"{doc_name}_middle.json"
    content_list_save_path = raw_dir / f"{doc_name}_content_list.json"
    raw_dir.mkdir(parents=True, exist_ok=True)

    middle_save_path.write_text(
        json.dumps(result.middle_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    content_list_save_path.write_text(
        json.dumps(result.content_list, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Synthesize XML
    xml_str = build_document_xml(
        content_list=result.content_list,
        middle_json=result.middle_json,
        doc_name=doc_name,
    )

    xml_output_path = output_dir / doc_name / f"{doc_name}_{backend}.xml"
    save_document_xml(xml_str, xml_output_path)
    print(f"[{backend}] XML saved: {xml_output_path}")

    log_xml_stats(xml_str, xml_output_path)

    # Log XML as WandB artifact
    artifact = wandb.Artifact(
        name=f"{doc_name}-{backend}",
        type="document-xml",
        description=f"Synthesized XML for {doc_name} using {backend}",
    )
    artifact.add_file(str(xml_output_path))
    wandb_run.log_artifact(artifact)

    return xml_output_path


def main() -> None:
    """Entry point for the extraction pipeline CLI."""
    parser = argparse.ArgumentParser(
        description="MinerU2.5-Pro extraction pipeline with XML synthesis and WandB logging."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Path to a PDF file or directory of PDFs.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="hybrid-http-client",
        choices=sorted(SUPPORTED_BACKENDS),
        help="MinerU inference backend. Default: hybrid-http-client (uses running VLM server).",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="VLM server base URL for http-client backends, e.g. http://localhost:8000. Required with hybrid-http-client.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="OCR language hint (e.g. 'en', 'ch'). Default: en.",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=0,
        help="Zero-based first page index. Default: 0.",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="Zero-based last page index (inclusive). Default: all pages.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory for raw MinerU JSON outputs. Default: data/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for synthesized XML. Default: output/.",
    )

    title_group = parser.add_argument_group(
        "title-aided",
        "LLM-aided title hierarchy (H1–H4). All three flags are required together.",
    )
    title_group.add_argument(
        "--title-aided-api-key",
        type=str,
        default=None,
        help="API key for the LLM endpoint (use 'dummy' for local servers).",
    )
    title_group.add_argument(
        "--title-aided-base-url",
        type=str,
        default=None,
        help="Base URL of the OpenAI-compatible LLM endpoint, e.g. http://localhost:8000/v1.",
    )
    title_group.add_argument(
        "--title-aided-model",
        type=str,
        default=None,
        help="Model name to use for title hierarchy inference.",
    )
    title_group.add_argument(
        "--title-aided-enable-thinking",
        action="store_true",
        default=False,
        help=(
            "Pass enable_thinking=True to the LLM endpoint. Required for "
            "Qwen3-*-Thinking models so the model wraps reasoning in <think> "
            "tags that MinerU can strip before parsing the dict."
        ),
    )

    args = parser.parse_args()

    title_aided_args = (args.title_aided_api_key, args.title_aided_base_url, args.title_aided_model)
    if any(title_aided_args) and not all(title_aided_args):
        parser.error(
            "--title-aided-api-key, --title-aided-base-url, and --title-aided-model must all be set together."
        )
    title_aided_config: dict | None = None
    if all(title_aided_args):
        title_aided_config = {
            "api_key": args.title_aided_api_key,
            "base_url": args.title_aided_base_url,
            "model": args.title_aided_model,
        }
        if args.title_aided_enable_thinking:
            title_aided_config["enable_thinking"] = True

    if args.backend == "hybrid-http-client" and not args.server_url:
        parser.error("--server-url is required when --backend is hybrid-http-client.")

    pdfs = collect_pdfs(args.pdf)
    print(f"Found {len(pdfs)} PDF(s) to process with backend={args.backend}")

    for pdf_path in pdfs:
        run = init_run(
            pdf_name=pdf_path.name,
            backend=args.backend,
            extra_config={
                "language": args.lang,
                "start_page": args.start_page,
                "end_page": args.end_page,
                "server_url": args.server_url,
                "title_aided": title_aided_config is not None,
            },
        )
        try:
            process_pdf(
                pdf_path=pdf_path,
                backend=args.backend,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                language=args.lang,
                start_page=args.start_page,
                end_page=args.end_page,
                wandb_run=run,
                server_url=args.server_url,
                title_aided_config=title_aided_config,
            )
        finally:
            wandb.finish()


if __name__ == "__main__":
    main()
