"""Tests for main.py — pure helpers only (no GPU/MinerU/subprocess)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from main import save_raw_json


class TestSaveRawJson:
    def test_writes_middle_and_content_list_json(self, tmp_path: Path) -> None:
        result = SimpleNamespace(
            middle_json={"pdf_info": [{"page": 1}]},
            content_list=[{"type": "text", "text": "hello"}],
        )

        save_raw_json(result, tmp_path, "paper")

        middle_path = tmp_path / "paper_middle.json"
        content_path = tmp_path / "paper_content_list.json"
        assert middle_path.exists()
        assert content_path.exists()
        assert json.loads(middle_path.read_text(encoding="utf-8")) == result.middle_json
        assert json.loads(content_path.read_text(encoding="utf-8")) == result.content_list

    def test_preserves_unicode_without_escaping(self, tmp_path: Path) -> None:
        result = SimpleNamespace(
            middle_json={"title": "café"},
            content_list=[{"text": "日本語"}],
        )

        save_raw_json(result, tmp_path, "doc")

        middle_text = (tmp_path / "doc_middle.json").read_text(encoding="utf-8")
        content_text = (tmp_path / "doc_content_list.json").read_text(encoding="utf-8")
        assert "café" in middle_text
        assert "日本語" in content_text
        # ensure_ascii=False means no \uXXXX escapes
        assert "\\u" not in middle_text
        assert "\\u" not in content_text

    def test_creates_parent_directory_if_missing(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "nested" / "doc"
        result = SimpleNamespace(middle_json={}, content_list=[])

        # out_dir does not exist yet; caller is expected to mkdir first per
        # the _finalize_document contract, but save_raw_json itself should
        # not crash if the directory already exists.
        out_dir.mkdir(parents=True)
        save_raw_json(result, out_dir, "doc")

        assert (out_dir / "doc_middle.json").exists()
        assert (out_dir / "doc_content_list.json").exists()
