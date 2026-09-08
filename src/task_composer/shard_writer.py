"""WebDataset-compatible .tar shard writing."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any

from src.task_composer.base import Sample


class ShardWriter:
    """Writes ``Sample`` objects into WebDataset-compatible ``.tar`` shards.

    Each shard is a plain ``.tar`` (or ``.tar.gz``) archive. Entries are
    named ``{sample.key}.{ext}`` for every ``(ext, value)`` pair in
    ``sample.payload``, excluding ``"__key__"`` which is metadata-only.

    Args:
        out_dir: Directory where shard files are written. Created if absent.
        task_name: Prefix for shard filenames, e.g. ``"caption_prediction"``.
        shard_size: Maximum number of samples per shard archive.
        compress: Write ``.tar.gz`` instead of plain ``.tar`` when ``True``.
    """

    def __init__(
        self,
        out_dir: Path,
        task_name: str,
        shard_size: int = 1000,
        compress: bool = False,
    ) -> None:
        if shard_size < 1:
            raise ValueError(f"shard_size must be >= 1, got {shard_size}")
        self.out_dir = Path(out_dir)
        self.task_name = task_name
        self.shard_size = shard_size
        self._mode = "w:gz" if compress else "w:"
        self._ext = ".tar.gz" if compress else ".tar"

    def write(self, samples: list[Sample], start_index: int = 0) -> list[Path]:
        """Write samples into sharded ``.tar`` archives.

        Args:
            samples: Samples to write. Processed in the order provided.
            start_index: Shard number the first archive is named with. Callers
                writing in several passes pass the next unused index so a later
                pass does not overwrite the shards an earlier one produced.

        Returns:
            Ordered list of paths to written shard files.
        """
        if not samples:
            return []

        self.out_dir.mkdir(parents=True, exist_ok=True)
        shard_paths: list[Path] = []

        for offset, start in enumerate(range(0, len(samples), self.shard_size)):
            shard_idx = start_index + offset
            chunk = samples[start : start + self.shard_size]
            shard_path = self.out_dir / f"{self.task_name}-{shard_idx:05d}{self._ext}"

            with tarfile.open(shard_path, self._mode) as tar:
                for sample in chunk:
                    for ext, value in sample.payload.items():
                        if ext == "__key__":
                            continue
                        content = self._encode(value)
                        entry_name = f"{sample.key}.{ext}"
                        info = tarfile.TarInfo(name=entry_name)
                        info.size = len(content)
                        tar.addfile(info, io.BytesIO(content))

            shard_paths.append(shard_path)

        return shard_paths

    def _encode(self, value: Any) -> bytes:
        """Encode a payload value to bytes for tar entry storage.

        Priority: bytes → verbatim, dict/list → UTF-8 JSON, anything else → str().encode().
        """
        if isinstance(value, bytes):
            return value
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False).encode()
        return str(value).encode()


class ShardBuffer:
    """Accumulates samples per task across documents, flushing full shards.

    ``ShardWriter`` shards whatever list it is handed, so writing one document
    at a time would produce a shard per document — thousands of tiny archives
    instead of the uniform ones WebDataset expects. ``ShardBuffer`` sits in
    front of it and holds samples until ``shard_size`` of them have arrived,
    so shards stay full regardless of how the samples are fed in. Anything
    left over is written by ``close``.

    Buffering is per task: each task gets its own writer, its own shard
    numbering, and its own partial shard.

    Args:
        out_dir: Directory shard files are written to. Created if absent.
        shard_size: Samples per shard archive.
        compress: Write ``.tar.gz`` instead of plain ``.tar`` when ``True``.
    """

    def __init__(self, out_dir: Path, shard_size: int = 1000, compress: bool = False) -> None:
        self.out_dir = Path(out_dir)
        self.shard_size = shard_size
        self.compress = compress
        self._writers: dict[str, ShardWriter] = {}
        self._pending: dict[str, list[Sample]] = {}
        self._next_index: dict[str, int] = {}
        self._written: dict[str, list[Path]] = {}

    def add(self, task_name: str, samples: list[Sample]) -> None:
        """Buffer samples for a task, writing out any now-complete shards.

        Args:
            task_name: Task the samples belong to; names its shard files.
            samples: Samples to buffer. An empty list is a no-op.
        """
        if not samples:
            return
        pending = self._pending.setdefault(task_name, [])
        pending.extend(samples)

        full_count = len(pending) - (len(pending) % self.shard_size)
        if full_count:
            self._flush(task_name, pending[:full_count])
            del pending[:full_count]

    def close(self) -> dict[str, list[Path]]:
        """Write every partial shard still buffered.

        Safe to call more than once; buffers are cleared as they are flushed.

        Returns:
            Mapping of task name to the shard paths written over this buffer's
            lifetime, in write order.
        """
        for task_name, pending in self._pending.items():
            if pending:
                self._flush(task_name, list(pending))
                pending.clear()
        return self._written

    def _flush(self, task_name: str, samples: list[Sample]) -> None:
        """Write *samples* as shards for *task_name*, continuing its numbering."""
        writer = self._writers.get(task_name)
        if writer is None:
            writer = ShardWriter(self.out_dir, task_name, self.shard_size, self.compress)
            self._writers[task_name] = writer

        start = self._next_index.get(task_name, 0)
        paths = writer.write(samples, start_index=start)
        self._next_index[task_name] = start + len(paths)
        self._written.setdefault(task_name, []).extend(paths)
