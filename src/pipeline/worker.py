"""Per-GPU worker loop.

A worker is a long-lived subprocess pinned to one GPU. It reads PDF paths from
stdin (one per line), processes each, and emits one status line on stdout. The
loop ends when stdin is closed (EOF), at which point the worker exits.

The actual extraction work is injected as ``process`` so the loop is unit
testable without a GPU or MinerU: tests pass a fake that returns page/time
metadata or raises.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from src.pipeline.protocol import encode_status


def worker_loop(
    input_lines: Iterable[str],
    process: Callable[[str], dict],
    emit: Callable[[str], None],
) -> None:
    """Process each PDF path from ``input_lines``, emitting one status per doc.

    For every non-empty input line the loop calls ``process(pdf_path)``. On
    success it emits a ``done`` event carrying any ``pages``/``seconds`` the
    processor reported; on any exception it emits a ``failed`` event with the
    error string. The same ``doc`` value (the input path) is echoed back so the
    dispatcher can map the result to its manifest entry.

    Args:
        input_lines: Iterable of input lines (typically ``sys.stdin``).
        process: Callable that processes one PDF path and returns a dict of
            metadata (e.g. ``{"pages": 10, "seconds": 12.5}``); raises on
            failure.
        emit: Callable that writes one protocol line (e.g. prints to stdout).
    """
    for raw in input_lines:
        path = raw.strip()
        if not path:
            continue
        try:
            info = process(path)
        except Exception as exc:  # noqa: BLE001 — report any failure, keep serving
            emit(encode_status("failed", doc=path, error=str(exc)))
        else:
            emit(
                encode_status(
                    "done",
                    doc=path,
                    pages=info.get("pages"),
                    seconds=info.get("seconds"),
                )
            )
