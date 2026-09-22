"""Append-only JSONL manifest for resumable, retriable batch processing.

Each processed document appends one JSON record to a line-delimited file:

    {"doc": "icml/2024/paper", "status": "done", "pages": 10, "seconds": 12.5}
    {"doc": "icml/2024/other", "status": "failed", "error": "CUDA OOM"}

The manifest lives outside the output tree so the output directory holds only
real artifacts. It enables three things:

- **Resume** — on restart, documents already marked ``done`` are skipped.
- **Retry** — ``--retry-failed`` re-queues documents marked ``failed``.
- **Audit** — a durable record of what ran, how long it took, and why it failed.

Concurrency model: a manifest has a **single writer**. In multi-GPU runs the
parent dispatcher is the only writer; in single-GPU runs ``main.py`` is. Worker
subprocesses never touch it, so no locking is required. Reads on startup
tolerate a truncated or corrupt final line (the common shape of a crash during
a write).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def compute_run_id(input_root: Path, backend: str, output_dir: Path) -> str:
    """Return a stable short identifier for a (input, backend, output) run.

    Reruns against the same input directory, backend, and output directory
    resolve to the same id, so they share a manifest and resume each other.

    Args:
        input_root: The ``--pdf`` input path.
        backend: Extraction backend name.
        output_dir: Configured output root.

    Returns:
        A 12-character hex digest.
    """
    key = f"{input_root.resolve()}|{backend}|{output_dir.resolve()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def manifest_path(state_dir: Path, run_id: str) -> Path:
    """Return the manifest file path for a run id under ``state_dir``."""
    return state_dir / run_id / "manifest.jsonl"


class Manifest:
    """Single-writer JSONL store mapping document ids to their latest status."""

    def __init__(self, path: Path) -> None:
        """Open (and create) the manifest at ``path``, loading any prior state.

        Args:
            path: Manifest file path. Parent directories are created.
        """
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._status: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # Truncated/corrupt line (e.g. crash mid-write) — skip it.
                    continue
                doc = record.get("doc")
                status = record.get("status")
                if doc is not None and status is not None:
                    self._status[doc] = status  # last record wins

    def _append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._status[record["doc"]] = record["status"]

    def mark_done(
        self, doc: str, *, pages: int | None = None, seconds: float | None = None
    ) -> None:
        """Record ``doc`` as successfully processed.

        Args:
            doc: Document id (typically the relative output path, posix form).
            pages: Page count, for the audit trail.
            seconds: Wall-clock processing time, for the audit trail.
        """
        record = {"doc": doc, "status": "done"}
        if pages is not None:
            record["pages"] = pages
        if seconds is not None:
            record["seconds"] = round(seconds, 3)
        self._append(record)

    def mark_failed(self, doc: str, error: str) -> None:
        """Record ``doc`` as failed with a short error string.

        Args:
            doc: Document id.
            error: Failure reason (truncated for storage).
        """
        self._append({"doc": doc, "status": "failed", "error": error[:500]})

    def status(self, doc: str) -> str | None:
        """Return the latest status for ``doc`` (``"done"``/``"failed"``) or ``None``."""
        return self._status.get(doc)

    def pending(self, docs: list[str], *, retry_failed: bool = False) -> list[str]:
        """Filter ``docs`` to those still needing work, preserving order.

        A document is pending when it has no ``done`` record. Documents marked
        ``failed`` are excluded unless ``retry_failed`` is set.

        Args:
            docs: Candidate document ids.
            retry_failed: Re-include documents previously marked ``failed``.

        Returns:
            The subset of ``docs`` that should be processed.
        """
        result = []
        for doc in docs:
            st = self._status.get(doc)
            if st == "done":
                continue
            if st == "failed" and not retry_failed:
                continue
            result.append(doc)
        return result
