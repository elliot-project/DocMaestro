"""Tests for src/utils/paths.py — output path derivation."""

from __future__ import annotations

from pathlib import Path

from src.utils.paths import doc_rel_dir, output_xml_path


class TestDocRelDir:
    def test_nested_pdf_under_dir_root_preserves_hierarchy(self, tmp_path: Path) -> None:
        root = tmp_path / "pdfs"
        pdf = root / "icml" / "2024" / "paper.pdf"
        pdf.parent.mkdir(parents=True)
        pdf.touch()

        assert doc_rel_dir(pdf, root) == Path("icml/2024/paper")

    def test_pdf_directly_in_root_is_flat(self, tmp_path: Path) -> None:
        root = tmp_path / "pdfs"
        pdf = root / "paper.pdf"
        root.mkdir()
        pdf.touch()

        assert doc_rel_dir(pdf, root) == Path("paper")

    def test_relative_input_root_preserves_hierarchy(self, tmp_path: Path, monkeypatch) -> None:
        # Bug A regression: a relative --pdf used to raise ValueError on
        # relative_to() and silently flatten the hierarchy.
        root = tmp_path / "pdfs"
        pdf = root / "icml" / "2024" / "paper.pdf"
        pdf.parent.mkdir(parents=True)
        pdf.touch()

        monkeypatch.chdir(tmp_path)
        rel_root = Path("pdfs")
        rel_pdf = Path("pdfs/icml/2024/paper.pdf")

        assert doc_rel_dir(rel_pdf, rel_root) == Path("icml/2024/paper")

    def test_single_file_root_is_flat(self, tmp_path: Path) -> None:
        pdf = tmp_path / "report.pdf"
        pdf.touch()

        assert doc_rel_dir(pdf, pdf) == Path("report")

    def test_pdf_outside_root_falls_back_to_flat(self, tmp_path: Path) -> None:
        root = tmp_path / "pdfs"
        root.mkdir()
        outside = tmp_path / "elsewhere" / "paper.pdf"
        outside.parent.mkdir(parents=True)
        outside.touch()

        assert doc_rel_dir(outside, root) == Path("paper")


class TestOutputXmlPath:
    def test_joins_output_root_rel_dir_and_backend(self) -> None:
        path = output_xml_path(
            Path("/out"), Path("icml/2024/paper"), "paper", "hybrid-auto-engine"
        )
        assert path == Path("/out/icml/2024/paper/paper_hybrid-auto-engine.xml")
