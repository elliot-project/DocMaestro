"""WebDataset-compatible .tar shard writer."""

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

    def write(self, samples: list[Sample]) -> list[Path]:
        """Write samples into sharded ``.tar`` archives.

        Args:
            samples: Samples to write. Processed in the order provided.

        Returns:
            Ordered list of paths to written shard files.
        """
        if not samples:
            return []

        self.out_dir.mkdir(parents=True, exist_ok=True)
        shard_paths: list[Path] = []

        for shard_idx, start in enumerate(range(0, len(samples), self.shard_size)):
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
