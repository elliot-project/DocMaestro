"""Tests for src/utils/manifest.py — JSONL resume/retry state store."""

from __future__ import annotations

from pathlib import Path

from doc_maestro.utils.manifest import Manifest, compute_run_id, manifest_path


class TestManifest:
    def test_mark_done_then_status_is_done(self, tmp_path: Path) -> None:
        m = Manifest(tmp_path / "manifest.jsonl")
        m.mark_done("icml/2024/paper", pages=10, seconds=12.5)
        assert m.status("icml/2024/paper") == "done"

    def test_unknown_doc_has_no_status(self, tmp_path: Path) -> None:
        m = Manifest(tmp_path / "manifest.jsonl")
        assert m.status("nope") is None

    def test_pending_excludes_done(self, tmp_path: Path) -> None:
        m = Manifest(tmp_path / "manifest.jsonl")
        m.mark_done("a")
        assert m.pending(["a", "b", "c"]) == ["b", "c"]

    def test_pending_excludes_failed_by_default(self, tmp_path: Path) -> None:
        m = Manifest(tmp_path / "manifest.jsonl")
        m.mark_failed("a", "OOM")
        assert m.pending(["a", "b"]) == ["b"]

    def test_pending_includes_failed_when_retry_requested(self, tmp_path: Path) -> None:
        m = Manifest(tmp_path / "manifest.jsonl")
        m.mark_failed("a", "OOM")
        assert m.pending(["a", "b"], retry_failed=True) == ["a", "b"]

    def test_state_persists_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        Manifest(path).mark_done("a")
        assert Manifest(path).status("a") == "done"

    def test_last_record_wins_failed_then_done(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        m = Manifest(path)
        m.mark_failed("a", "OOM")
        m.mark_done("a")
        assert Manifest(path).status("a") == "done"
        assert Manifest(path).pending(["a"]) == []

    def test_corrupt_final_line_is_tolerated(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        m = Manifest(path)
        m.mark_done("a")
        # Simulate a crash mid-write: a truncated trailing JSON line.
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"doc": "b", "status": "do')

        reopened = Manifest(path)
        assert reopened.status("a") == "done"
        assert reopened.status("b") is None


class TestRunId:
    def test_run_id_is_stable_for_same_inputs(self, tmp_path: Path) -> None:
        a = compute_run_id(tmp_path, "hybrid-auto-engine", tmp_path / "out")
        b = compute_run_id(tmp_path, "hybrid-auto-engine", tmp_path / "out")
        assert a == b

    def test_run_id_differs_by_backend(self, tmp_path: Path) -> None:
        a = compute_run_id(tmp_path, "hybrid-auto-engine", tmp_path / "out")
        b = compute_run_id(tmp_path, "pipeline", tmp_path / "out")
        assert a != b

    def test_manifest_path_layout(self, tmp_path: Path) -> None:
        assert manifest_path(tmp_path, "abc123") == tmp_path / "abc123" / "manifest.jsonl"
