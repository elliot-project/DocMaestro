"""Tests for src/dataset/splits.py — document-level train/val/test assignment."""

from __future__ import annotations

import pytest

from src.dataset.splits import SplitRatios, assign_split, split_documents, venue_of


class TestVenueOf:
    def test_extracts_leading_component(self) -> None:
        assert venue_of("CVPR/2020/some_paper") == "CVPR"

    def test_flat_doc_id_has_no_venue(self) -> None:
        assert venue_of("some_paper") == ""


class TestRatioValidation:
    def test_rejects_ratios_that_do_not_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum"):
            SplitRatios(0.5, 0.2, 0.1)

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            SplitRatios(1.2, -0.2, 0.0)

    def test_accepts_float_noise(self) -> None:
        SplitRatios(0.98, 0.01, 0.01)

    def test_allows_empty_test_split(self) -> None:
        assert SplitRatios(0.9, 0.1, 0.0).test == 0.0


class TestDeterminism:
    def test_same_doc_same_seed_is_stable(self) -> None:
        a = assign_split("CVPR/2020/paper", seed=42)
        b = assign_split("CVPR/2020/paper", seed=42)
        assert a == b

    def test_stable_across_calls_without_shared_state(self) -> None:
        """Independent workers must agree; no process-local RNG allowed."""
        ids = [f"CVPR/2020/p{i}" for i in range(200)]
        first = [assign_split(d, seed=7) for d in ids]
        second = [assign_split(d, seed=7) for d in reversed(ids)][::-1]
        assert first == second

    def test_seed_changes_assignment(self) -> None:
        ids = [f"CVPR/2020/p{i}" for i in range(300)]
        a = [assign_split(d, seed=1) for d in ids]
        b = [assign_split(d, seed=2) for d in ids]
        assert a != b

    def test_only_valid_split_names_returned(self) -> None:
        got = {assign_split(f"nips/2024/p{i}", seed=0) for i in range(300)}
        assert got <= {"train", "val", "test"}


class TestProportions:
    def test_respects_ratios_within_tolerance(self) -> None:
        ratios = SplitRatios(0.8, 0.1, 0.1)
        ids = [f"CVPR/2020/p{i}" for i in range(20_000)]
        assigned = [assign_split(d, ratios=ratios, seed=42) for d in ids]
        train = assigned.count("train") / len(ids)
        assert 0.78 < train < 0.82

    def test_a_zero_ratio_split_is_never_assigned(self) -> None:
        ratios = SplitRatios(0.9, 0.1, 0.0)
        ids = [f"CVPR/2020/p{i}" for i in range(5_000)]
        assert "test" not in {assign_split(d, ratios=ratios, seed=42) for d in ids}


class TestVenueStratification:
    def test_each_venue_holds_the_ratio(self) -> None:
        """A split skewed by venue would train on CVPR and validate on NIPS."""
        ratios = SplitRatios(0.8, 0.1, 0.1)
        for venue in ("CVPR", "ICCV", "nips", "WACV"):
            ids = [f"{venue}/2020/p{i}" for i in range(8_000)]
            assigned = [assign_split(d, ratios=ratios, seed=42) for d in ids]
            share = assigned.count("train") / len(ids)
            assert 0.77 < share < 0.83, f"{venue} train share {share:.3f}"


class TestSplitDocuments:
    def test_partitions_exhaustively_and_disjointly(self) -> None:
        ids = [f"CVPR/2020/p{i}" for i in range(500)]
        out = split_documents(ids, seed=42)
        assert set(out) == {"train", "val", "test"}
        merged = out["train"] + out["val"] + out["test"]
        assert sorted(merged) == sorted(ids)
        assert len(set(merged)) == len(ids)

    def test_preserves_input_order_within_each_split(self) -> None:
        ids = [f"CVPR/2020/p{i}" for i in range(200)]
        out = split_documents(ids, seed=42)
        assert out["train"] == [d for d in ids if assign_split(d, seed=42) == "train"]

    def test_empty_input_yields_empty_splits(self) -> None:
        assert split_documents([], seed=42) == {"train": [], "val": [], "test": []}

    def test_duplicate_doc_ids_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            split_documents(["a", "a"], seed=42)


class TestNoPageLevelLeak:
    def test_all_pages_of_a_document_share_a_split(self) -> None:
        """Pages of one paper share layout and content; splitting them leaks."""
        doc = "CVPR/2020/paper"
        assert len({assign_split(doc, seed=42) for _ in range(50)}) == 1
