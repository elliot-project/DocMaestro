"""MinerU extraction wrapper for pipeline, VLM-auto, and hybrid-auto engines."""

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_BACKENDS = frozenset(["pipeline", "vlm-auto-engine", "hybrid-auto-engine"])


@dataclass
class ExtractionResult:
    """Parsed outputs from a single MinerU extraction run.

    Args:
        content_list: Flat reading-order list from content_list.json.
        middle_json: Full topology dict from middle.json.
        backend: The MinerU backend used (e.g. 'hybrid-auto-engine').
        pdf_path: Absolute path to the source PDF.
        duration_seconds: Wall-clock time for the extraction subprocess.
        page_count: Number of pages detected in the document.
        output_dir: Directory where MinerU wrote its raw outputs.
    """

    content_list: list[dict]
    middle_json: dict
    backend: str
    pdf_path: Path
    duration_seconds: float
    page_count: int
    output_dir: Path


def _find_json_file(directory: Path, suffix: str) -> Path | None:
    """Recursively find a JSON file ending with *suffix* under directory.

    Args:
        directory: Root directory to search.
        suffix: Filename suffix to match (e.g. '_middle.json').

    Returns:
        Path to the matching file, or None if not found.
    """
    matches = list(directory.rglob(f"*{suffix}"))
    if not matches:
        return None
    # Prefer the shallowest match
    return sorted(matches, key=lambda p: len(p.parts))[0]


def extract(
    pdf_path: Path,
    output_dir: Path,
    backend: str = "hybrid-auto-engine",
    language: str = "en",
    start_page: int = 0,
    end_page: int | None = None,
) -> ExtractionResult:
    """Run MinerU extraction on a PDF and return parsed outputs.

    Invokes the `mineru` CLI as a subprocess, then reads the resulting
    middle.json and content_list.json files from disk.

    Args:
        pdf_path: Path to the source PDF file.
        output_dir: Directory where MinerU should write its outputs.
        backend: Inference backend — one of 'pipeline', 'vlm-auto-engine',
            or 'hybrid-auto-engine'. Defaults to 'hybrid-auto-engine'.
        language: OCR language hint (e.g. 'en', 'ch'). Defaults to 'en'.
        start_page: Zero-based index of the first page to parse.
        end_page: Zero-based index of the last page (inclusive), or None
            for all pages.

    Returns:
        ExtractionResult containing parsed JSON data and timing metadata.

    Raises:
        ValueError: If backend is not one of the supported backends.
        FileNotFoundError: If the PDF does not exist.
        RuntimeError: If MinerU subprocess fails or output files are missing.
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{backend}'. Choose from: {sorted(SUPPORTED_BACKENDS)}"
        )
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mineru",
        "--path",
        str(pdf_path),
        "--output",
        str(output_dir),
        "--backend",
        backend,
        "--lang",
        language,
        "--start",
        str(start_page),
    ]
    if end_page is not None:
        cmd += ["--end", str(end_page)]

    print(f"  cmd: {' '.join(cmd)}")
    print("  (MinerU may download model weights on first run — this can take several minutes)")

    start_time = time.perf_counter()
    # Stream stdout to terminal so progress is visible; capture stderr for error messages.
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    duration = time.perf_counter() - start_time

    if result.returncode != 0:
        raise RuntimeError(
            f"mineru subprocess failed (exit {result.returncode}).\n"
            f"STDERR: {result.stderr[-3000:]}"
        )

    middle_path = _find_json_file(output_dir, "_middle.json")
    content_list_path = _find_json_file(output_dir, "_content_list.json")

    if middle_path is None:
        raise RuntimeError(f"middle.json not found under {output_dir}.")
    if content_list_path is None:
        raise RuntimeError(f"content_list.json not found under {output_dir}.")

    middle_json: dict = json.loads(middle_path.read_text(encoding="utf-8"))
    content_list: list[dict] = json.loads(content_list_path.read_text(encoding="utf-8"))

    page_count = len(middle_json.get("pdf_info", []))

    return ExtractionResult(
        content_list=content_list,
        middle_json=middle_json,
        backend=backend,
        pdf_path=pdf_path.resolve(),
        duration_seconds=duration,
        page_count=page_count,
        output_dir=output_dir,
    )
