"""Work-queue dispatcher: hand tasks to GPU workers, one at a time.

The dispatcher owns a single pool of tasks and feeds the next one to whichever
worker just finished. This is self-balancing (a worker that draws heavy
documents simply pulls fewer of them) and structurally immune to the
"more shards than GPUs → dropped PDFs" bug, because there are no shards: every
task is dispatched exactly once, to whichever worker is free.

Workers communicate results back through a shared queue of
``(worker_id, payload)`` tuples. ``payload`` is either a decoded status dict
(``{"event": "done"|"failed", "doc": ...}``) or the sentinel
``{"event": "died"}`` pushed when a worker's stream ends unexpectedly.

The coordination logic here is pure (no subprocess or GPU code) so it can be
tested with in-memory fake workers; the real subprocess wiring lives in
``scripts/run_sharded.py``.
"""

from __future__ import annotations

import queue
from collections import deque
from collections.abc import Callable
from typing import Protocol


class Worker(Protocol):
    """A handle the dispatcher can feed tasks to and shut down."""

    id: str

    def send(self, task: str) -> None:
        """Hand one task to the worker (e.g. write a PDF path to its stdin)."""

    def close(self) -> None:
        """Signal no more tasks (e.g. close the worker's stdin)."""


def run_dispatch(
    tasks: list[str],
    workers: list[Worker],
    results: "queue.Queue[tuple[str, dict]]",
    on_result: Callable[[dict], None],
) -> None:
    """Dispatch ``tasks`` across ``workers``, calling ``on_result`` per outcome.

    Each worker holds at most one in-flight task. When a worker reports a
    result, ``on_result`` is invoked with the payload and the worker is fed the
    next task (or closed if the pool is empty). If a worker dies, its in-flight
    task is reported as failed and the remaining work continues on the survivors;
    if every worker dies, the still-pending tasks are reported as failed so no
    task is silently lost.

    Args:
        tasks: Task tokens (PDF paths) to process, each dispatched exactly once.
        workers: Worker handles, each processing one task at a time.
        results: Shared queue the workers push ``(worker_id, payload)`` onto.
        on_result: Callback invoked once per task with its outcome payload
            (``{"event": "done"|"failed", "doc": ...}``).
    """
    pending: deque[str] = deque(tasks)
    by_id = {w.id: w for w in workers}
    in_flight: dict[str, str | None] = {w.id: None for w in workers}
    live: set[str] = set(by_id)
    closed: set[str] = set()

    def feed_idle() -> None:
        for wid in by_id:
            if wid in live and in_flight[wid] is None and pending:
                task = pending.popleft()
                by_id[wid].send(task)
                in_flight[wid] = task
        # Nothing left for an idle live worker → close it once.
        for wid in by_id:
            if wid in live and in_flight[wid] is None and not pending and wid not in closed:
                by_id[wid].close()
                closed.add(wid)

    feed_idle()

    while any(t is not None for t in in_flight.values()):
        wid, payload = results.get()
        if payload.get("event") == "died":
            live.discard(wid)
            stalled = in_flight.get(wid)
            if stalled is not None:
                on_result({"event": "failed", "doc": stalled, "error": "worker died"})
                in_flight[wid] = None
        else:
            on_result(payload)
            in_flight[wid] = None
        feed_idle()

    # Every worker died with work still queued — fail the remainder explicitly.
    while pending:
        on_result({"event": "failed", "doc": pending.popleft(), "error": "no live workers"})

    for wid in by_id:
        if wid not in closed:
            by_id[wid].close()
            closed.add(wid)
