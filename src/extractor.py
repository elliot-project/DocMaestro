"""MinerU extraction wrapper for pipeline, VLM-auto, and hybrid-auto engines."""

import ast
import importlib
import json
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_BACKENDS = frozenset(
    ["pipeline", "vlm-auto-engine", "hybrid-auto-engine", "hybrid-http-client"]
)

# Only the hybrid backend has title_aided support; patching pipeline/vlm modules is not needed.
_HYBRID_MIDDLE_JSON_MOD = "mineru.backend.hybrid.hybrid_model_output_to_middle_json"


@dataclass
class ExtractionResult:
    """Parsed outputs from a single MinerU extraction run.

    Args:
        content_list: Flat reading-order list from content_list.json.
        middle_json: Full topology dict from middle.json.
        backend: The MinerU backend used (e.g. 'hybrid-auto-engine').
        pdf_path: Absolute path to the source PDF.
        duration_seconds: Wall-clock time for the extraction.
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

    Returns the most recently modified match. do_parse() writes to a nested
    subdirectory; process_pdf() may leave an older top-level copy from a
    prior run. Sorting by mtime ensures we always read the freshest file.

    Args:
        directory: Root directory to search.
        suffix: Filename suffix to match (e.g. '_middle.json').

    Returns:
        Path to the matching file, or None if not found.
    """
    matches = list(directory.rglob(f"*{suffix}"))
    if not matches:
        return None
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _validate_title_aided_config(config: dict) -> None:
    """Validate that title_aided_config contains the required keys.

    Args:
        config: Dict to validate.

    Raises:
        ValueError: If required keys are missing or have wrong types.
    """
    for key in ("api_key", "base_url", "model"):
        if key not in config:
            raise ValueError(f"title_aided_config is missing required key: '{key}'")
        if not isinstance(config[key], str):
            raise ValueError(f"title_aided_config['{key}'] must be a string")


def _robust_parse_llm_dict(content: str) -> dict:
    """Parse an LLM-generated dict string tolerating Python and JSON syntax.

    LLMs sometimes return compact Python dict syntax like {0:1,1:2,...} with
    unquoted integer keys. json_repair misparses this compact form (it treats
    "1,1" as a value), so we fall back to ast.literal_eval which handles it
    correctly. json_repair is tried first because it tolerates prose wrappers
    and code fences that literal_eval would reject.

    Args:
        content: Raw string from the LLM response.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If neither parser succeeds.
    """
    import json_repair as _jr

    # Attempt 1: json_repair + validate int coercion (catches json_repair misparses)
    try:
        result = _jr.loads(content)
        {int(k): int(v) for k, v in result.items()}  # validation only
        return result
    except (ValueError, TypeError, AttributeError):
        pass

    # Attempt 2: ast.literal_eval handles {0:1,1:2,...} with unquoted int keys
    result = ast.literal_eval(content)
    if not isinstance(result, dict):
        raise ValueError(f"Expected dict, got {type(result).__name__}")
    return result


def _make_patched_request_title_levels(original_fn):
    """Wrap _request_title_levels to use robust JSON parsing instead of json_repair alone."""
    import functools

    @functools.wraps(original_fn)
    def patched(title_aided_config, title_dict, prompt_builder=None):
        import json_repair as _jr

        # Capture the real json_repair.loads BEFORE replacing it so the inner
        # closure calls the original, not itself (which would cause recursion).
        _original_loads = _jr.loads

        def _robust_loads(content):
            # Attempt 1: real json_repair (handles JSON and most prose-wrapped dicts)
            try:
                result = _original_loads(content)
                {int(k): int(v) for k, v in result.items()}
                return result
            except (ValueError, TypeError, AttributeError):
                pass

            # Attempt 2: ast.literal_eval (handles {0:1,1:2,...} unquoted int keys)
            try:
                result = ast.literal_eval(content)
                if isinstance(result, dict):
                    {int(k): int(v) for k, v in result.items()}
                    return result
            except (ValueError, SyntaxError):
                pass

            # Attempt 3: extract the last {...} block from prose (handles thinking
            # models that output reasoning text followed by the dict without <think>
            # tags, e.g. Qwen3-VL-8B-Thinking without enable_thinking=True)
            m = re.search(r"(\{[^{}]+\})\s*$", content.strip())
            if m:
                try:
                    result = ast.literal_eval(m.group(1))
                    if isinstance(result, dict):
                        {int(k): int(v) for k, v in result.items()}
                        return result
                except (ValueError, SyntaxError):
                    pass

            raise ValueError(f"Could not parse int-keyed dict from LLM content (len={len(content)})")

        _jr.loads = _robust_loads
        try:
            return original_fn(title_aided_config, title_dict, prompt_builder)
        finally:
            _jr.loads = _original_loads

    return patched


@contextmanager
def _title_aided_patch(title_aided_config: dict | None):
    """Temporarily patch the hybrid module's title_aided globals for one extraction.

    Sets title_aided_enable=True, injects the LLM config, and replaces
    json_repair.loads inside _request_title_levels with a robust parser that
    falls back to ast.literal_eval for Python-syntax dicts ({0:1,1:2,...}).
    All patches are restored on exit.

    Args:
        title_aided_config: LLM config dict (api_key, base_url, model), or None.
    """
    if title_aided_config is None:
        yield
        return

    # Patch 1: hybrid module globals
    mid_mod = importlib.import_module(_HYBRID_MIDDLE_JSON_MOD)
    old_enable = getattr(mid_mod, "title_aided_enable", False)
    old_config = getattr(mid_mod, "title_aided_config", None)
    mid_mod.title_aided_enable = True
    mid_mod.title_aided_config = title_aided_config

    # Patch 2: robust JSON parsing in _request_title_levels
    llm_mod = importlib.import_module("mineru.utils.llm_aided")
    original_fn = llm_mod._request_title_levels
    llm_mod._request_title_levels = _make_patched_request_title_levels(original_fn)

    try:
        yield
    finally:
        mid_mod.title_aided_enable = old_enable
        mid_mod.title_aided_config = old_config
        llm_mod._request_title_levels = original_fn


def extract(
    pdf_path: Path,
    output_dir: Path,
    backend: str = "hybrid-http-client",
    language: str = "en",
    start_page: int = 0,
    end_page: int | None = None,
    server_url: str | None = None,
    title_aided_config: dict | None = None,
    gpu_memory_utilization: float | None = None,
) -> ExtractionResult:
    """Run MinerU extraction on a PDF and return parsed outputs.

    Calls MinerU's Python API (`do_parse`) directly instead of a subprocess,
    then reads the resulting middle.json and content_list.json from disk.

    Args:
        pdf_path: Path to the source PDF file.
        output_dir: Directory where MinerU should write its outputs.
        backend: Inference backend. Use 'hybrid-http-client' (default) when a
            VLM server is already running, or 'hybrid-auto-engine' to load the
            model in-process. Other options: 'pipeline', 'vlm-auto-engine'.
        language: OCR language hint (e.g. 'en', 'ch'). Defaults to 'en'.
        start_page: Zero-based index of the first page to parse.
        end_page: Zero-based index of the last page (inclusive), or None
            for all pages.
        server_url: VLM server base URL for http-client backends, e.g.
            'http://localhost:8000'. Required when backend is 'hybrid-http-client'.
        title_aided_config: Optional LLM config for H1–H4 title hierarchy
            detection. Required keys: 'api_key' (str), 'base_url' (str),
            'model' (str). Optional key: 'enable_thinking' (bool).
            Example: {"api_key": "dummy", "base_url": "http://localhost:8000/v1",
                      "model": "Qwen/Qwen3-VL-8B-Thinking"}
            Only active for hybrid backends.
        gpu_memory_utilization: Fraction of GPU VRAM vLLM may use for KV-cache
            (e.g. 0.85). Passed directly to vllm.LLM(). None → MinerU default
            (0.5). Higher values increase batch capacity and throughput at the
            cost of VRAM headroom. Only used by in-process vLLM backends.
            No-op for the 'pipeline' backend (pipeline uses CV models, not vLLM).

    Returns:
        ExtractionResult containing parsed JSON data and timing metadata.

    Raises:
        ValueError: If backend is not supported, or title_aided_config is malformed.
        FileNotFoundError: If the PDF does not exist.
        RuntimeError: If output files are not produced after extraction.
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{backend}'. Choose from: {sorted(SUPPORTED_BACKENDS)}"
        )
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if title_aided_config is not None:
        _validate_title_aided_config(title_aided_config)

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_name = pdf_path.stem
    pdf_bytes = pdf_path.read_bytes()

    print(f"  backend={backend}, language={language}")
    if server_url:
        print(f"  server_url={server_url}")
    if gpu_memory_utilization is not None:
        print(f"  gpu_memory_utilization={gpu_memory_utilization}")
    if title_aided_config is not None:
        print(f"  title_aided enabled (model={title_aided_config['model']})")

    from mineru.cli.common import do_parse  # lazy — avoids heavy deps at import time

    # Extra vLLM kwargs forwarded through do_parse → vllm.LLM()
    vlm_kwargs: dict = {}
    if gpu_memory_utilization is not None:
        vlm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization

    start_time = time.perf_counter()
    with _title_aided_patch(title_aided_config):
        do_parse(
            output_dir=str(output_dir),
            pdf_file_names=[doc_name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=[language],
            backend=backend,
            parse_method="auto",
            server_url=server_url,
            f_dump_middle_json=True,
            f_dump_content_list=True,
            f_dump_md=False,
            f_dump_orig_pdf=False,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_model_output=False,
            start_page_id=start_page,
            end_page_id=end_page,
            **vlm_kwargs,
        )
    duration = time.perf_counter() - start_time

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


def batch_extract(
    pdf_paths: list[Path],
    output_dir: Path,
    backend: str = "pipeline",
    language: str = "en",
    server_url: str | None = None,
    gpu_memory_utilization: float | None = None,
) -> list[ExtractionResult | Exception]:
    """Run MinerU extraction on a batch of PDFs in a single do_parse call.

    Passing multiple PDFs at once lets MinerU's streaming pipeline batch pages
    across all documents in one GPU forward pass. The sliding window (default
    64 pages, set via MINERU_PROCESSING_WINDOW_SIZE) fills from all documents
    simultaneously, improving GPU utilisation compared to calling extract()
    once per document.

    Output structure: do_parse writes each doc to
      <output_dir>/<doc_stem>/<parse_method>/  (e.g. .../doc1/auto/)
    Use _find_json_file(output_dir / doc_stem, suffix) to locate results.

    Throughput notes:
      - 'pipeline': gains come from filling the page window. batch_ratio
        (internal GPU batch multiplier) is auto-set from VRAM by MinerU.
        Set MINERU_PROCESSING_WINDOW_SIZE if batch_size × avg_pages > 64.
      - VLM/hybrid: page-batching gains are limited here; use aio_do_parse
        for true async concurrency on those backends.
      - gpu_memory_utilization is silently ignored for 'pipeline' by MinerU.

    Args:
        pdf_paths: PDFs to process. All must exist. Stems must be unique
            within the batch (MinerU uses stem as the output sub-directory).
        output_dir: Shared root directory. MinerU writes each doc to
            output_dir/<stem>/<parse_method>/.
        backend: MinerU backend. Defaults to 'pipeline'.
        language: OCR language hint applied to all documents.
        server_url: VLM server URL for http-client backends.
        gpu_memory_utilization: vLLM KV-cache fraction. No-op for pipeline.

    Returns:
        List aligned with pdf_paths: ExtractionResult on success, Exception
        on per-doc output failure. If do_parse itself raises, all entries
        in the batch receive the same Exception.

    Raises:
        ValueError: If backend is unsupported.
        FileNotFoundError: If any pdf_path does not exist.
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{backend}'. Choose from: {sorted(SUPPORTED_BACKENDS)}"
        )
    for p in pdf_paths:
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {p}")

    output_dir.mkdir(parents=True, exist_ok=True)

    doc_names = [p.stem for p in pdf_paths]
    pdf_bytes_list = [p.read_bytes() for p in pdf_paths]
    lang_list = [language] * len(pdf_paths)

    vlm_kwargs: dict = {}
    if gpu_memory_utilization is not None:
        vlm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization

    from mineru.cli.common import do_parse

    start_time = time.perf_counter()
    batch_exc: Exception | None = None
    try:
        do_parse(
            output_dir=str(output_dir),
            pdf_file_names=list(doc_names),
            pdf_bytes_list=list(pdf_bytes_list),
            p_lang_list=lang_list,
            backend=backend,
            parse_method="auto",
            server_url=server_url,
            f_dump_middle_json=True,
            f_dump_content_list=True,
            f_dump_md=False,
            f_dump_orig_pdf=False,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_model_output=False,
            **vlm_kwargs,
        )
    except Exception as exc:
        batch_exc = exc

    duration = time.perf_counter() - start_time

    results: list[ExtractionResult | Exception] = []
    for doc_name, pdf_path in zip(doc_names, pdf_paths):
        if batch_exc is not None:
            results.append(batch_exc)
            continue
        doc_out = output_dir / doc_name
        try:
            middle_path = _find_json_file(doc_out, "_middle.json")
            content_list_path = _find_json_file(doc_out, "_content_list.json")
            if middle_path is None:
                raise RuntimeError(f"middle.json not found under {doc_out}.")
            if content_list_path is None:
                raise RuntimeError(f"content_list.json not found under {doc_out}.")
            middle_json: dict = json.loads(middle_path.read_text(encoding="utf-8"))
            content_list: list[dict] = json.loads(content_list_path.read_text(encoding="utf-8"))
            results.append(ExtractionResult(
                content_list=content_list,
                middle_json=middle_json,
                backend=backend,
                pdf_path=pdf_path.resolve(),
                duration_seconds=duration,
                page_count=len(middle_json.get("pdf_info", [])),
                output_dir=doc_out,
            ))
        except Exception as exc:
            results.append(exc)

    return results
