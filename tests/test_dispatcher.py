"""Tests for src/pipeline/dispatcher.py — the parent work-queue coordinator.

These exercise the scheduling logic with fake in-memory workers, so no
subprocesses or GPUs are involved. The central guarantee is that every task is
dispatched exactly once regardless of the task/worker ratio — the structural
fix for the dropped-shard bug.
"""

from __future__ import annotations

import queue

from src.pipeline.dispatcher import run_dispatch


class FakeWorker:
    """A worker that completes every task it receives instantly."""

    def __init__(self, worker_id: str, results: queue.Queue) -> None:
        self.id = worker_id
        self._results = results
        self.sent: list[str] = []
        self.closed = False

    def send(self, task: str) -> None:
        self.sent.append(task)
        self._results.put((self.id, {"event": "done", "doc": task, "pages": 1, "seconds": 0.1}))

    def close(self) -> None:
        self.closed = True


class DyingWorker:
    """A worker that completes ``die_after`` tasks, then dies on the next send."""

    def __init__(self, worker_id: str, results: queue.Queue, die_after: int = 0) -> None:
        self.id = worker_id
        self._results = results
        self.sent: list[str] = []
        self.closed = False
        self._completed = 0
        self._die_after = die_after

    def send(self, task: str) -> None:
        self.sent.append(task)
        if self._completed >= self._die_after:
            self._results.put((self.id, {"event": "died"}))
        else:
            self._completed += 1
            self._results.put((self.id, {"event": "done", "doc": task, "pages": 1}))

    def close(self) -> None:
        self.closed = True


def _run(tasks, workers, results):
    recorded: list[dict] = []
    run_dispatch(tasks, workers, results, recorded.append)
    return recorded


def _drain(tasks, n_workers):
    results: queue.Queue = queue.Queue()
    workers = [FakeWorker(f"g{i}", results) for i in range(n_workers)]
    return workers, _run(tasks, workers, results)


class TestRunDispatch:
    def test_every_task_dispatched_exactly_once(self) -> None:
        tasks = [f"doc{i}.pdf" for i in range(7)]
        workers, recorded = _drain(tasks, n_workers=2)

        all_sent = [t for w in workers for t in w.sent]
        assert sorted(all_sent) == sorted(tasks)
        assert len(all_sent) == len(tasks)  # exactly once, none dropped
        assert {r["doc"] for r in recorded} == set(tasks)
        assert all(r["event"] == "done" for r in recorded)

    def test_more_workers_than_tasks_closes_idle_workers(self) -> None:
        tasks = ["a.pdf", "b.pdf"]
        workers, recorded = _drain(tasks, n_workers=4)

        assert len([r for r in recorded if r["event"] == "done"]) == 2
        assert all(w.closed for w in workers)
        assert sum(len(w.sent) for w in workers) == 2

    def test_single_worker_processes_all(self) -> None:
        tasks = [f"d{i}.pdf" for i in range(5)]
        workers, recorded = _drain(tasks, n_workers=1)

        assert workers[0].sent == tasks
        assert len(recorded) == 5

    def test_worker_death_fails_its_inflight_doc_survivors_finish(self) -> None:
        results: queue.Queue = queue.Queue()
        alive = FakeWorker("g0", results)
        dying = DyingWorker("g1", results, die_after=0)  # dies on first task
        tasks = [f"d{i}.pdf" for i in range(5)]

        recorded = _run(tasks, [alive, dying], results)

        # Every task accounted for exactly once.
        assert {r["doc"] for r in recorded} == set(tasks)
        failed = [r for r in recorded if r["event"] == "failed"]
        done = [r for r in recorded if r["event"] == "done"]
        assert len(failed) == 1
        assert failed[0]["error"] == "worker died"
        assert len(done) == 4
        assert alive.closed

    def test_all_workers_die_remaining_tasks_marked_failed(self) -> None:
        results: queue.Queue = queue.Queue()
        w = DyingWorker("g0", results, die_after=0)
        tasks = ["a.pdf", "b.pdf", "c.pdf"]

        recorded = _run(tasks, [w], results)

        assert {r["doc"] for r in recorded} == set(tasks)
        assert all(r["event"] == "failed" for r in recorded)
