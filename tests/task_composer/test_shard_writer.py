"""Tests for ShardWriter."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

from doc_maestro.task_composer.base import Sample
from doc_maestro.task_composer.shard_writer import ShardBuffer, ShardWriter


def _make_sample(key: str, text: str = "hello") -> Sample:
    return Sample(
        key=key, payload={"__key__": key, "target.txt": text, "json": {"task_type": "test"}}
    )


def _read_tar_entries(path: Path) -> dict[str, bytes]:
    """Return {member_name: content_bytes} for all entries in a tar file."""
    entries = {}
    with tarfile.open(path, "r:*") as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f:
                entries[member.name] = f.read()
    return entries


class TestShardWriterBasic:
    def test_creates_output_dir(self, tmp_path):
        out = tmp_path / "shards"
        writer = ShardWriter(out_dir=out, task_name="test_task")
        writer.write([_make_sample("k0")])
        assert out.exists()

    def test_returns_shard_paths(self, tmp_path):
        writer = ShardWriter(out_dir=tmp_path, task_name="test_task")
        paths = writer.write([_make_sample("k0"), _make_sample("k1")])
        assert len(paths) == 1
        assert paths[0].suffix == ".tar"

    def test_shard_filename_uses_task_name(self, tmp_path):
        writer = ShardWriter(out_dir=tmp_path, task_name="caption_prediction")
        paths = writer.write([_make_sample("k0")])
        assert paths[0].name.startswith("caption_prediction-")

    def test_shard_filename_is_zero_padded(self, tmp_path):
        writer = ShardWriter(out_dir=tmp_path, task_name="t")
        paths = writer.write([_make_sample("k0")])
        assert "-00000.tar" in paths[0].name

    def test_empty_samples_writes_no_shards(self, tmp_path):
        writer = ShardWriter(out_dir=tmp_path, task_name="t")
        paths = writer.write([])
        assert paths == []

    def test_tar_entries_named_correctly(self, tmp_path):
        writer = ShardWriter(out_dir=tmp_path, task_name="t")
        paths = writer.write([_make_sample("mykey")])
        entries = _read_tar_entries(paths[0])
        assert "mykey.target.txt" in entries
        assert "mykey.json" in entries
        assert "mykey.__key__" not in entries  # __key__ is NOT written as a file

    def test_text_value_encoded_as_utf8(self, tmp_path):
        writer = ShardWriter(out_dir=tmp_path, task_name="t")
        paths = writer.write([_make_sample("k0", text="café")])
        entries = _read_tar_entries(paths[0])
        assert entries["k0.target.txt"] == "café".encode()

    def test_dict_value_encoded_as_json(self, tmp_path):
        writer = ShardWriter(out_dir=tmp_path, task_name="t")
        paths = writer.write([_make_sample("k0")])
        entries = _read_tar_entries(paths[0])
        parsed = json.loads(entries["k0.json"])
        assert parsed["task_type"] == "test"

    def test_bytes_value_written_verbatim(self, tmp_path):
        raw = b"\xff\xd8\xff\xe0"
        s = Sample(key="k", payload={"__key__": "k", "image.jpg": raw})
        writer = ShardWriter(out_dir=tmp_path, task_name="t")
        paths = writer.write([s])
        entries = _read_tar_entries(paths[0])
        assert entries["k.image.jpg"] == raw


class TestShardRotation:
    def test_multiple_shards_when_exceeds_shard_size(self, tmp_path):
        writer = ShardWriter(out_dir=tmp_path, task_name="t", shard_size=3)
        samples = [_make_sample(f"k{i}") for i in range(7)]
        paths = writer.write(samples)
        assert len(paths) == 3  # ceil(7 / 3) = 3

    def test_shard_indices_are_sequential(self, tmp_path):
        writer = ShardWriter(out_dir=tmp_path, task_name="t", shard_size=2)
        paths = writer.write([_make_sample(f"k{i}") for i in range(4)])
        names = [p.name for p in paths]
        assert "t-00000.tar" in names
        assert "t-00001.tar" in names

    def test_each_shard_contains_correct_samples(self, tmp_path):
        writer = ShardWriter(out_dir=tmp_path, task_name="t", shard_size=2)
        paths = writer.write([_make_sample(f"k{i}") for i in range(4)])
        shard0 = _read_tar_entries(paths[0])
        shard1 = _read_tar_entries(paths[1])
        assert "k0.target.txt" in shard0
        assert "k1.target.txt" in shard0
        assert "k2.target.txt" in shard1
        assert "k3.target.txt" in shard1


class TestShardWriterCompress:
    def test_compress_produces_tar_gz(self, tmp_path):
        writer = ShardWriter(out_dir=tmp_path, task_name="t", compress=True)
        paths = writer.write([_make_sample("k0")])
        assert paths[0].name.endswith(".tar.gz")

    def test_compressed_tar_is_readable(self, tmp_path):
        writer = ShardWriter(out_dir=tmp_path, task_name="t", compress=True)
        paths = writer.write([_make_sample("k0", text="compressed")])
        entries = _read_tar_entries(paths[0])
        assert entries["k0.target.txt"] == b"compressed"


class TestShardWriterStartIndex:
    """start_index lets a caller append shards without clobbering earlier ones."""

    def test_shard_numbering_starts_at_given_index(self, tmp_path):
        w = ShardWriter(tmp_path, "t", shard_size=2)
        paths = w.write([_make_sample(f"k{i}") for i in range(3)], start_index=5)
        assert [p.name for p in paths] == ["t-00005.tar", "t-00006.tar"]

    def test_default_start_index_is_zero(self, tmp_path):
        w = ShardWriter(tmp_path, "t", shard_size=2)
        paths = w.write([_make_sample("k0")])
        assert paths[0].name == "t-00000.tar"

    def test_successive_writes_do_not_overwrite(self, tmp_path):
        w = ShardWriter(tmp_path, "t", shard_size=1)
        w.write([_make_sample("a")], start_index=0)
        w.write([_make_sample("b")], start_index=1)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["t-00000.tar", "t-00001.tar"]
        assert "a.target.txt" in _read_tar_entries(tmp_path / "t-00000.tar")
        assert "b.target.txt" in _read_tar_entries(tmp_path / "t-00001.tar")


class TestShardBuffer:
    """ShardBuffer accumulates samples across documents and flushes full shards."""

    def test_flushes_when_buffer_reaches_shard_size(self, tmp_path):
        buf = ShardBuffer(tmp_path, shard_size=2)
        buf.add("ocr", [_make_sample("a"), _make_sample("b")])
        assert (tmp_path / "ocr-00000.tar").exists()

    def test_holds_partial_shard_until_close(self, tmp_path):
        buf = ShardBuffer(tmp_path, shard_size=5)
        buf.add("ocr", [_make_sample("a")])
        assert list(tmp_path.iterdir()) == []
        buf.close()
        assert (tmp_path / "ocr-00000.tar").exists()

    def test_samples_from_several_documents_share_a_shard(self, tmp_path):
        buf = ShardBuffer(tmp_path, shard_size=4)
        buf.add("ocr", [_make_sample("doc1_a"), _make_sample("doc1_b")])
        buf.add("ocr", [_make_sample("doc2_a"), _make_sample("doc2_b")])
        buf.close()
        entries = _read_tar_entries(tmp_path / "ocr-00000.tar")
        assert "doc1_a.target.txt" in entries
        assert "doc2_b.target.txt" in entries

    def test_shard_indices_continue_across_flushes(self, tmp_path):
        buf = ShardBuffer(tmp_path, shard_size=1)
        buf.add("ocr", [_make_sample("a"), _make_sample("b")])
        buf.add("ocr", [_make_sample("c")])
        buf.close()
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "ocr-00000.tar",
            "ocr-00001.tar",
            "ocr-00002.tar",
        ]

    def test_tasks_are_sharded_independently(self, tmp_path):
        buf = ShardBuffer(tmp_path, shard_size=2)
        buf.add("ocr", [_make_sample("a")])
        buf.add("layout", [_make_sample("b")])
        buf.close()
        assert (tmp_path / "ocr-00000.tar").exists()
        assert (tmp_path / "layout-00000.tar").exists()

    def test_close_returns_written_paths_per_task(self, tmp_path):
        buf = ShardBuffer(tmp_path, shard_size=1)
        buf.add("ocr", [_make_sample("a"), _make_sample("b")])
        buf.add("layout", [_make_sample("c")])
        written = buf.close()
        assert [p.name for p in written["ocr"]] == ["ocr-00000.tar", "ocr-00001.tar"]
        assert [p.name for p in written["layout"]] == ["layout-00000.tar"]

    def test_close_is_idempotent(self, tmp_path):
        buf = ShardBuffer(tmp_path, shard_size=5)
        buf.add("ocr", [_make_sample("a")])
        buf.close()
        buf.close()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["ocr-00000.tar"]

    def test_adding_no_samples_writes_nothing(self, tmp_path):
        buf = ShardBuffer(tmp_path, shard_size=2)
        buf.add("ocr", [])
        buf.close()
        assert list(tmp_path.iterdir()) == []

    def test_compress_produces_tar_gz(self, tmp_path):
        buf = ShardBuffer(tmp_path, shard_size=1, compress=True)
        buf.add("ocr", [_make_sample("a")])
        buf.close()
        assert (tmp_path / "ocr-00000.tar.gz").exists()
