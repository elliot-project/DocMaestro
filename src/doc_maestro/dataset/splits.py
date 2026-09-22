"""Document-level train/val/test assignment.

Splits are assigned **per document, never per page**. Pages of one paper share
layout, typography, figure style and subject matter, so a page-level split puts
near-duplicates of the training data into validation and every task's metrics
become optimistic. Because every sample key carries its document id, holding
the document fixed holds all of its samples on one side of the split.

Assignment is a pure function of ``(doc_id, seed)`` computed from a stable
digest, so it needs no global shuffle, no ordering, and no shared state: a
streaming packer, a second machine, and a later incremental run over new
documents all agree without coordinating. Adding documents never reshuffles the
ones already assigned.

Venue stratification comes free from that uniformity — the digest is
independent of the venue prefix, so each venue lands on the target proportions
in expectation rather than needing separate per-venue bookkeeping.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

SPLIT_NAMES = ("train", "val", "test")
_RESOLUTION = 1_000_000  # digest buckets; caps quantisation error at 1e-6


@dataclass(frozen=True)
class SplitRatios:
    """Target proportions for the three splits.

    Args:
        train: Share of documents assigned to training.
        val: Share assigned to validation.
        test: Share assigned to test. May be ``0.0``.

    Raises:
        ValueError: If any share is negative or the three do not sum to 1.
    """

    train: float = 0.9
    val: float = 0.05
    test: float = 0.05

    def __post_init__(self) -> None:
        if min(self.train, self.val, self.test) < 0:
            raise ValueError(
                f"Split ratios must be non-negative, got "
                f"train={self.train}, val={self.val}, test={self.test}"
            )
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total} "
                f"(train={self.train}, val={self.val}, test={self.test})"
            )


DEFAULT_RATIOS = SplitRatios()


def venue_of(doc_id: str) -> str:
    """Return the venue prefix of a document id.

    Args:
        doc_id: Posix-form document id, e.g. ``"CVPR/2020/some_paper"``.

    Returns:
        The leading path component, or ``""`` for a flat id.
    """
    head, sep, _ = doc_id.partition("/")
    return head if sep else ""


def assign_split(
    doc_id: str,
    *,
    ratios: SplitRatios = DEFAULT_RATIOS,
    seed: int = 42,
) -> str:
    """Return the split ``doc_id`` belongs to.

    Args:
        doc_id: Document identifier; all of its samples follow it.
        ratios: Target proportions.
        seed: Changes the partition without changing its statistics.

    Returns:
        One of ``"train"``, ``"val"``, ``"test"``.
    """
    digest = hashlib.sha1(f"{seed}|{doc_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % _RESOLUTION

    train_cut = round(ratios.train * _RESOLUTION)
    val_cut = train_cut + round(ratios.val * _RESOLUTION)
    if bucket < train_cut:
        return "train"
    if bucket < val_cut:
        return "val"
    return "test"


def split_documents(
    doc_ids: list[str],
    *,
    ratios: SplitRatios = DEFAULT_RATIOS,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Partition documents into the three splits, preserving input order.

    Args:
        doc_ids: Document identifiers. Must be unique — a repeated id would
            place the same document in a split twice and, since sample keys are
            derived from it, silently duplicate its samples.
        ratios: Target proportions.
        seed: Changes the partition without changing its statistics.

    Returns:
        Mapping of split name to its document ids, in input order. Every split
        name is present even when its list is empty.

    Raises:
        ValueError: If ``doc_ids`` contains duplicates.
    """
    seen: set[str] = set()
    duplicates = sorted({d for d in doc_ids if d in seen or seen.add(d)})  # type: ignore[func-returns-value]
    if duplicates:
        raise ValueError(
            f"split_documents received duplicate doc ids: {duplicates[:5]}"
            f"{' ...' if len(duplicates) > 5 else ''}"
        )

    out: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for doc_id in doc_ids:
        out[assign_split(doc_id, ratios=ratios, seed=seed)].append(doc_id)
    return out
