"""Tests for FigureSelectionComposer."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from src.task_composer.base import DocumentContext
from src.task_composer.tasks.figure_selection import FigureSelectionComposer


def _make_figure(img_src: str = "images/fig.jpg") -> ET.Element:
    fig = ET.Element("figure", {"bbox": "[0,0,100,100]"})
    img = ET.SubElement(fig, "img")
    img.set("src", img_src)
    return fig


def _make_page(idx: int, figures: list[ET.Element] | None = None, text: str = "") -> ET.Element:
    page = ET.Element("page", {"idx": str(idx), "width": "600", "height": "800"})
    if text:
        tb = ET.SubElement(page, "text-block", {"bbox": "[0,0,600,20]"})
        tb.text = text
    for fig in (figures or []):
        page.append(fig)
    return page


def _make_doc(pages: list[ET.Element], tmp_path: Path) -> tuple[ET.Element, DocumentContext]:
    root = ET.Element("document", {"name": "testdoc", "page_count": str(len(pages))})
    for p in pages:
        root.append(p)
    ctx = DocumentContext(doc_name="testdoc", pages=pages, img_root=tmp_path)
    return root, ctx


class TestTaskName:
    def test_task_name(self):
        assert FigureSelectionComposer.task_name == "figure_selection"


class TestNoSamples:
    def test_page_with_no_figures_returns_empty(self, tmp_path):
        pages = [
            _make_page(0, text="hello"),
            _make_page(1, figures=[_make_figure("images/a.jpg")]),
        ]
        _, ctx = _make_doc(pages, tmp_path)
        composer = FigureSelectionComposer(n_distractors=1, seed=0)
        samples = composer.compose_page(pages[0], ctx)
        assert samples == []

    def test_not_enough_distractors_returns_empty(self, tmp_path):
        # Only 1 figure in the whole doc (the correct one) — can't get any distractors
        pages = [_make_page(0, figures=[_make_figure("images/a.jpg")])]
        _, ctx = _make_doc(pages, tmp_path)
        composer = FigureSelectionComposer(n_distractors=2, seed=0)
        samples = composer.compose_page(pages[0], ctx)
        assert samples == []


class TestSampleStructure:
    def _make_multi_page_doc(self, tmp_path: Path):
        pages = [
            _make_page(0, figures=[_make_figure("images/p0.jpg")], text="page zero context"),
            _make_page(1, figures=[_make_figure("images/p1.jpg")]),
            _make_page(2, figures=[_make_figure("images/p2.jpg")]),
        ]
        return pages, _make_doc(pages, tmp_path)[1]

    def test_one_sample_per_figure_on_page(self, tmp_path):
        pages = [
            _make_page(0, figures=[_make_figure("images/a.jpg"), _make_figure("images/b.jpg")]),
            _make_page(1, figures=[_make_figure("images/c.jpg"), _make_figure("images/d.jpg")]),
        ]
        _, ctx = _make_doc(pages, tmp_path)
        composer = FigureSelectionComposer(n_distractors=1, seed=0)
        samples = composer.compose_page(pages[0], ctx)
        assert len(samples) == 2

    def test_candidate_count_matches_n_distractors_plus_one(self, tmp_path):
        pages, ctx = self._make_multi_page_doc(tmp_path)
        composer = FigureSelectionComposer(n_distractors=2, seed=0)
        samples = composer.compose_page(pages[0], ctx)
        assert len(samples[0].payload["json"]["candidates"]) == 3  # 1 correct + 2 distractors

    def test_label_is_valid_candidate_index(self, tmp_path):
        pages, ctx = self._make_multi_page_doc(tmp_path)
        composer = FigureSelectionComposer(n_distractors=2, seed=0)
        samples = composer.compose_page(pages[0], ctx)
        label = samples[0].payload["target.json"]["label"]
        candidates = samples[0].payload["json"]["candidates"]
        assert 0 <= label < len(candidates)

    def test_correct_candidate_has_page_origin_matching_current_page(self, tmp_path):
        pages, ctx = self._make_multi_page_doc(tmp_path)
        composer = FigureSelectionComposer(n_distractors=2, seed=0)
        samples = composer.compose_page(pages[0], ctx)
        label = samples[0].payload["target.json"]["label"]
        candidates = samples[0].payload["json"]["candidates"]
        correct = candidates[label]
        assert correct["page_origin"] == 0
        assert correct["src"] == "images/p0.jpg"

    def test_distractors_come_from_other_pages(self, tmp_path):
        pages, ctx = self._make_multi_page_doc(tmp_path)
        composer = FigureSelectionComposer(n_distractors=2, seed=0)
        samples = composer.compose_page(pages[0], ctx)
        label = samples[0].payload["target.json"]["label"]
        candidates = samples[0].payload["json"]["candidates"]
        distractor_origins = [
            c["page_origin"]
            for i, c in enumerate(candidates)
            if i != label
        ]
        assert all(o != 0 for o in distractor_origins)

    def test_key_format(self, tmp_path):
        pages, ctx = self._make_multi_page_doc(tmp_path)
        composer = FigureSelectionComposer(n_distractors=2, seed=0)
        samples = composer.compose_page(pages[0], ctx)
        assert samples[0].key == "testdoc_p000_fig_sel_0"

    def test_input_text_contains_page_text(self, tmp_path):
        pages, ctx = self._make_multi_page_doc(tmp_path)
        composer = FigureSelectionComposer(n_distractors=2, seed=0)
        samples = composer.compose_page(pages[0], ctx)
        assert "page zero context" in samples[0].payload["input.txt"]

    def test_input_text_includes_title_elements(self, tmp_path):
        title = ET.Element("title", {"bbox": "[0,0,600,30]", "level": "1"})
        title.text = "Annual Report 2024"
        pages = [
            _make_page(0, figures=[_make_figure("images/p0.jpg")]),
            _make_page(1, figures=[_make_figure("images/p1.jpg")]),
            _make_page(2, figures=[_make_figure("images/p2.jpg")]),
        ]
        pages[0].insert(0, title)
        _, ctx = _make_doc(pages, tmp_path)
        samples = FigureSelectionComposer(n_distractors=2, seed=0).compose_page(pages[0], ctx)
        assert "Annual Report 2024" in samples[0].payload["input.txt"]

    def test_json_metadata_task_type(self, tmp_path):
        pages, ctx = self._make_multi_page_doc(tmp_path)
        composer = FigureSelectionComposer(n_distractors=2, seed=0)
        samples = composer.compose_page(pages[0], ctx)
        assert samples[0].payload["json"]["task_type"] == "figure_selection"

    def test_payload_key_matches_sample_key(self, tmp_path):
        pages, ctx = self._make_multi_page_doc(tmp_path)
        composer = FigureSelectionComposer(n_distractors=2, seed=0)
        samples = composer.compose_page(pages[0], ctx)
        assert samples[0].payload["__key__"] == samples[0].key


class TestDeterminism:
    def test_same_seed_produces_same_label(self, tmp_path):
        pages = [
            _make_page(0, figures=[_make_figure("images/a.jpg")]),
            _make_page(1, figures=[_make_figure("images/b.jpg")]),
            _make_page(2, figures=[_make_figure("images/c.jpg")]),
        ]
        _, ctx = _make_doc(pages, tmp_path)
        s1 = FigureSelectionComposer(n_distractors=2, seed=42).compose_page(pages[0], ctx)
        s2 = FigureSelectionComposer(n_distractors=2, seed=42).compose_page(pages[0], ctx)
        assert s1[0].payload["target.json"]["label"] == s2[0].payload["target.json"]["label"]
