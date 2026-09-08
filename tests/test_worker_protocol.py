"""Tests for the parent<->worker status protocol and the worker loop."""

from __future__ import annotations

from src.pipeline.protocol import decode_status, encode_status
from src.pipeline.worker import worker_loop


class TestProtocol:
    def test_encode_then_decode_round_trip(self) -> None:
        line = encode_status("done", doc="a.pdf", pages=3, seconds=1.2)
        assert decode_status(line) == {
            "event": "done",
            "doc": "a.pdf",
            "pages": 3,
            "seconds": 1.2,
        }

    def test_decode_ignores_non_protocol_lines(self) -> None:
        assert decode_status("Loading model... 47%|####  | it/s") is None
        assert decode_status("") is None
        assert decode_status("✓ a.pdf: 3 pages in 1.2s") is None

    def test_decode_ignores_corrupt_protocol_json(self) -> None:
        assert decode_status("@@OMNIDOC@@ {not json") is None


class TestWorkerLoop:
    def _collect(self, lines, process):
        emitted: list[str] = []
        worker_loop(lines, process, emitted.append)
        return [decode_status(line) for line in emitted]

    def test_emits_done_per_successful_doc(self) -> None:
        results = self._collect(
            ["a.pdf\n", "b.pdf\n"],
            lambda p: {"pages": 5, "seconds": 2.0},
        )
        assert [r["event"] for r in results] == ["done", "done"]
        assert [r["doc"] for r in results] == ["a.pdf", "b.pdf"]
        assert results[0]["pages"] == 5

    def test_emits_failed_and_keeps_serving(self) -> None:
        def process(path: str) -> dict:
            if path == "bad.pdf":
                raise RuntimeError("CUDA OOM")
            return {"pages": 1, "seconds": 0.5}

        results = self._collect(["bad.pdf\n", "good.pdf\n"], process)
        assert results[0]["event"] == "failed"
        assert "CUDA OOM" in results[0]["error"]
        assert results[1]["event"] == "done"

    def test_blank_lines_are_skipped(self) -> None:
        calls: list[str] = []

        def process(path: str) -> dict:
            calls.append(path)
            return {}

        self._collect(["\n", "  \n", "a.pdf\n"], process)
        assert calls == ["a.pdf"]
