"""Tests for src/dataset/prompts.py — versioned prompt template registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_maestro.task_composer import registry as task_registry
from doc_maestro.training.prompts import DEFAULT_PROMPTS_PATH, PromptRegistry


@pytest.fixture
def tiny(tmp_path: Path) -> Path:
    path = tmp_path / "prompts.yaml"
    path.write_text(
        "version: test-1\n"
        "tasks:\n"
        "  markdown:\n"
        "    - 'Convert to markdown.'\n"
        "    - 'Transcribe as markdown.'\n"
        "    - 'Render this page in markdown.'\n"
        "  reading_order:\n"
        "    - 'Order these: {elements}'\n",
        encoding="utf-8",
    )
    return path


class TestLoading:
    def test_reads_version(self, tiny: Path) -> None:
        assert PromptRegistry.load(tiny).version == "test-1"

    def test_templates_returned_in_file_order(self, tiny: Path) -> None:
        assert PromptRegistry.load(tiny).templates("markdown")[0] == "Convert to markdown."

    def test_unknown_task_raises(self, tiny: Path) -> None:
        with pytest.raises(KeyError):
            PromptRegistry.load(tiny).templates("nope")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            PromptRegistry.load(tmp_path / "absent.yaml")

    def test_task_with_no_templates_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "p.yaml"
        path.write_text("version: v\ntasks:\n  markdown: []\n", encoding="utf-8")
        with pytest.raises(ValueError, match="markdown"):
            PromptRegistry.load(path)

    def test_missing_version_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "p.yaml"
        path.write_text("tasks:\n  markdown:\n    - 'x'\n", encoding="utf-8")
        with pytest.raises(ValueError, match="version"):
            PromptRegistry.load(path)


class TestDeterministicSelection:
    def test_same_key_and_epoch_gives_same_template(self, tiny: Path) -> None:
        r = PromptRegistry.load(tiny)
        a = r.select("markdown", sample_key="doc_p001_md_0", epoch=0)
        b = r.select("markdown", sample_key="doc_p001_md_0", epoch=0)
        assert a == b

    def test_selection_is_stable_across_registry_instances(self, tiny: Path) -> None:
        """Two workers loading the same file must agree — no process-local RNG."""
        a = PromptRegistry.load(tiny).select("markdown", sample_key="k", epoch=3)
        b = PromptRegistry.load(tiny).select("markdown", sample_key="k", epoch=3)
        assert a == b

    def test_epoch_varies_the_template(self, tiny: Path) -> None:
        r = PromptRegistry.load(tiny)
        seen = {r.select("markdown", sample_key="doc_p001_md_0", epoch=e) for e in range(30)}
        assert len(seen) > 1, "template never varied across epochs"

    def test_all_templates_reachable_across_epochs(self, tiny: Path) -> None:
        r = PromptRegistry.load(tiny)
        seen = {r.select("markdown", sample_key="doc_p001_md_0", epoch=e) for e in range(200)}
        assert seen == set(r.templates("markdown"))

    def test_different_keys_diverge(self, tiny: Path) -> None:
        r = PromptRegistry.load(tiny)
        seen = {r.select("markdown", sample_key=f"doc_p{i:03d}_md_0", epoch=0) for i in range(50)}
        assert len(seen) > 1

    def test_single_template_task_always_returns_it(self, tiny: Path) -> None:
        r = PromptRegistry.load(tiny)
        only = r.templates("reading_order")[0]
        assert r.select("reading_order", sample_key="k", epoch=7) == only


class TestRendering:
    def test_fills_placeholders(self, tiny: Path) -> None:
        r = PromptRegistry.load(tiny)
        out = r.render("reading_order", sample_key="k", epoch=0, fields={"elements": "A, B"})
        assert out == "Order these: A, B"

    def test_no_placeholder_template_ignores_extra_fields(self, tiny: Path) -> None:
        r = PromptRegistry.load(tiny)
        out = r.render("markdown", sample_key="k", epoch=0, fields={"unused": 1})
        assert out in r.templates("markdown")

    def test_missing_field_raises_with_task_and_field_named(self, tiny: Path) -> None:
        r = PromptRegistry.load(tiny)
        with pytest.raises(KeyError) as exc:
            r.render("reading_order", sample_key="k", epoch=0, fields={})
        assert "reading_order" in str(exc.value)
        assert "elements" in str(exc.value)

    def test_render_is_deterministic(self, tiny: Path) -> None:
        r = PromptRegistry.load(tiny)
        kw = {"sample_key": "k", "epoch": 2, "fields": {"elements": "x"}}
        assert r.render("reading_order", **kw) == r.render("reading_order", **kw)


class TestShippedRegistry:
    """The registry that actually ships must cover every registered task."""

    def test_default_file_exists(self) -> None:
        assert DEFAULT_PROMPTS_PATH.exists()

    def test_covers_every_registered_task(self) -> None:
        r = PromptRegistry.load(DEFAULT_PROMPTS_PATH)
        registered = {s.task_name for s in task_registry.list_tasks()}
        assert registered == set(r.tasks)

    def test_every_task_has_at_least_three_templates(self) -> None:
        r = PromptRegistry.load(DEFAULT_PROMPTS_PATH)
        thin = {t: len(r.templates(t)) for t in r.tasks if len(r.templates(t)) < 3}
        assert not thin, f"tasks with too few templates: {thin}"

    def test_templates_are_unique_within_a_task(self) -> None:
        r = PromptRegistry.load(DEFAULT_PROMPTS_PATH)
        for task in r.tasks:
            t = r.templates(task)
            assert len(t) == len(set(t)), f"duplicate templates in {task}"

    def test_every_template_renders_with_declared_fields(self) -> None:
        """A placeholder no sample supplies would fail only at training time."""
        r = PromptRegistry.load(DEFAULT_PROMPTS_PATH)
        for task in r.tasks:
            fields = dict.fromkeys(r.placeholders(task), "X")
            for epoch in range(len(r.templates(task))):
                r.render(task, sample_key="k", epoch=epoch, fields=fields)
