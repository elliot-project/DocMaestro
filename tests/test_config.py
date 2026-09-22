"""Tests for config loading and PDF collection."""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_maestro.utils.config import collect_pdfs, load_config


def _make_tree(root: Path) -> None:
    for rel in [
        "CVPR/2020/a.pdf",
        "CVPR/2020/b.pdf",
        "nips/2024/c.pdf",
        "tools/raw/d.pdf",
        "tools/raw/nested/e.pdf",
    ]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"%PDF-1.4\n")


class TestCollectPdfs:
    def test_collects_all_without_exclude(self, tmp_path: Path):
        _make_tree(tmp_path)
        assert len(collect_pdfs(tmp_path)) == 5

    def test_single_file_returned_as_is(self, tmp_path: Path):
        _make_tree(tmp_path)
        pdf = tmp_path / "CVPR/2020/a.pdf"
        assert collect_pdfs(pdf) == [pdf]

    def test_raises_when_empty(self, tmp_path: Path):
        with pytest.raises(ValueError):
            collect_pdfs(tmp_path)

    def test_directory_prefix_excludes_whole_subtree(self, tmp_path: Path):
        _make_tree(tmp_path)
        found = collect_pdfs(tmp_path, exclude=["tools/raw"])
        rels = sorted(str(p.relative_to(tmp_path)) for p in found)
        assert rels == ["CVPR/2020/a.pdf", "CVPR/2020/b.pdf", "nips/2024/c.pdf"]

    def test_top_level_directory_name_excludes_subtree(self, tmp_path: Path):
        _make_tree(tmp_path)
        found = collect_pdfs(tmp_path, exclude=["tools"])
        assert all("tools" not in str(p) for p in found)

    def test_glob_pattern_supported(self, tmp_path: Path):
        _make_tree(tmp_path)
        found = collect_pdfs(tmp_path, exclude=["*/raw/*"])
        assert len(found) == 3

    def test_empty_exclude_is_a_noop(self, tmp_path: Path):
        _make_tree(tmp_path)
        assert len(collect_pdfs(tmp_path, exclude=[])) == 5

    def test_excluding_everything_raises(self, tmp_path: Path):
        _make_tree(tmp_path)
        with pytest.raises(ValueError):
            collect_pdfs(tmp_path, exclude=["*"])

    def test_exclude_ignored_for_single_file_input(self, tmp_path: Path):
        """An explicitly named PDF is honoured even if a pattern would match it."""
        _make_tree(tmp_path)
        pdf = tmp_path / "tools/raw/d.pdf"
        assert collect_pdfs(pdf, exclude=["tools"]) == [pdf]


class TestPathsExcludeConfig:
    def test_exclude_defaults_to_empty(self, tmp_path: Path):
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text("paths:\n  data_dir: d\n", encoding="utf-8")
        assert load_config(cfg_path).paths.exclude == []

    def test_exclude_loaded_from_yaml(self, tmp_path: Path):
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text("paths:\n  exclude:\n    - tools/raw\n", encoding="utf-8")
        assert load_config(cfg_path).paths.exclude == ["tools/raw"]
