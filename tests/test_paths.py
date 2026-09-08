"""Tests for src/utils/paths.py — output path derivation."""

from __future__ import annotations

from pathlib import Path

from src.utils.paths import build_pdf_index, doc_rel_dir, find_img_root, output_xml_path


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
        path = output_xml_path(Path("/out"), Path("icml/2024/paper"), "paper", "hybrid-auto-engine")
        assert path == Path("/out/icml/2024/paper/paper_hybrid-auto-engine.xml")


def _images_dir(base: Path, n_files: int) -> Path:
    """Create an ``images/`` dir under *base* holding *n_files* JPEGs."""
    d = base / "images"
    d.mkdir(parents=True)
    for i in range(n_files):
        (d / f"{i}.jpg").touch()
    return d


class TestFindImgRoot:
    """XML <img src> values are 'images/<file>', so the resolved root is the
    PARENT of the images directory."""

    def test_returns_parent_of_images_dir(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        holder = data / "ASML" / "hybrid-auto-engine" / "ASML" / "hybrid_auto"
        _images_dir(holder, 3)
        assert find_img_root(data, "ASML") == holder

    def test_src_resolves_against_returned_root(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        holder = data / "doc" / "backend"
        _images_dir(holder, 1)
        root = find_img_root(data, "doc")
        assert (root / "images" / "0.jpg").exists()

    def test_picks_most_populated_images_dir(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        sparse = data / "doc" / "bench_r0"
        rich = data / "doc" / "full_run"
        _images_dir(sparse, 1)
        _images_dir(rich, 9)
        assert find_img_root(data, "doc") == rich

    def test_returns_none_when_no_images_anywhere(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        (data / "doc").mkdir(parents=True)
        assert find_img_root(data, "doc") is None

    def test_returns_none_when_data_dir_missing(self, tmp_path: Path) -> None:
        assert find_img_root(tmp_path / "nope", "doc") is None

    def test_falls_back_to_leaf_name_search(self, tmp_path: Path) -> None:
        """The XML may sit at output/a/b/doc while data/ nests it elsewhere."""
        data = tmp_path / "data"
        holder = data / "elsewhere" / "doc" / "backend"
        _images_dir(holder, 2)
        assert find_img_root(data, "unrelated/doc") == holder

    def test_nested_rel_folder_resolves_directly(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        holder = data / "conf" / "2024" / "paper" / "backend"
        _images_dir(holder, 2)
        assert find_img_root(data, "conf/2024/paper") == holder

    def test_direct_hit_wins_over_unrelated_doc(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        mine = data / "doc_a" / "backend"
        other = data / "doc_b" / "backend"
        _images_dir(mine, 2)
        _images_dir(other, 50)
        assert find_img_root(data, "doc_a") == mine


class TestBuildPdfIndex:
    def _pdf(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4\n")
        return path

    def test_maps_stem_to_path(self, tmp_path: Path) -> None:
        a = self._pdf(tmp_path / "CVPR" / "2020" / "a.pdf")
        b = self._pdf(tmp_path / "nips" / "2024" / "b.pdf")
        assert build_pdf_index(tmp_path) == {"a": a, "b": b}

    def test_missing_root_returns_empty(self, tmp_path: Path) -> None:
        assert build_pdf_index(tmp_path / "nope") == {}

    def test_empty_root_returns_empty(self, tmp_path: Path) -> None:
        assert build_pdf_index(tmp_path) == {}

    def test_duplicate_stems_resolve_deterministically(self, tmp_path: Path) -> None:
        """Two venues can hold the same title; the sorted-first path wins."""
        first = self._pdf(tmp_path / "CVPR" / "dup.pdf")
        self._pdf(tmp_path / "nips" / "dup.pdf")
        assert build_pdf_index(tmp_path)["dup"] == first

    def test_ignores_non_pdf_files(self, tmp_path: Path) -> None:
        self._pdf(tmp_path / "a.pdf")
        (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
        assert set(build_pdf_index(tmp_path)) == {"a"}
