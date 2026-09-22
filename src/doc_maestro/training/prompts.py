"""Versioned prompt template registry, applied at load time.

Samples on disk carry no instruction text. The dataloader picks a template here
and formats it, so a sample seen in three epochs is asked three different ways
instead of memorizing one phrasing, and changing the wording never requires
rebuilding ~2M samples.

Selection is a pure function of ``(task, sample_key, epoch)`` via a stable
digest rather than a stateful RNG. Two dataloader workers, or two runs on two
machines, therefore agree on which template a sample gets without sharing any
state — something ``random.Random`` seeded per process cannot guarantee, and
``hash()`` cannot either since it is salted per interpreter.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

import yaml

DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parents[3] / "configs" / "prompts" / "doctasks.yaml"


@dataclass(frozen=True)
class PromptRegistry:
    """Immutable set of prompt templates keyed by task name.

    Args:
        version: Registry version string, recorded into sample metadata so a
            trained checkpoint can be traced back to the wording it saw.
        tasks: Mapping of task name to its ordered templates.
    """

    version: str
    tasks: dict[str, tuple[str, ...]]

    @classmethod
    def load(cls, path: Path = DEFAULT_PROMPTS_PATH) -> PromptRegistry:
        """Load and validate a registry from YAML.

        Args:
            path: YAML file with top-level ``version`` and ``tasks`` keys.

        Returns:
            The parsed registry.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If ``version`` is absent, ``tasks`` is absent, or any
                task declares an empty template list.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt registry not found: {path}")

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        version = raw.get("version")
        if not version:
            raise ValueError(f"Prompt registry {path} is missing a 'version' key")

        raw_tasks = raw.get("tasks")
        if not raw_tasks:
            raise ValueError(f"Prompt registry {path} declares no tasks")

        tasks: dict[str, tuple[str, ...]] = {}
        for name, templates in raw_tasks.items():
            if not templates:
                raise ValueError(f"Task {name!r} in {path} has no prompt templates")
            tasks[name] = tuple(templates)

        return cls(version=str(version), tasks=tasks)

    def templates(self, task_name: str) -> tuple[str, ...]:
        """Return the templates declared for ``task_name``.

        Raises:
            KeyError: If the task is not in the registry.
        """
        try:
            return self.tasks[task_name]
        except KeyError:
            raise KeyError(
                f"No prompt templates for task {task_name!r}. "
                f"Known: {', '.join(sorted(self.tasks))}"
            ) from None

    def placeholders(self, task_name: str) -> set[str]:
        """Return every field name any of the task's templates interpolates.

        The union across templates, not the intersection: a field used by only
        one phrasing must still be supplied, since selection is per sample.

        Args:
            task_name: Key into the registry.

        Returns:
            Set of placeholder names, empty for instruction-only tasks.
        """
        names: set[str] = set()
        for template in self.templates(task_name):
            names.update(field for _, field, _, _ in Formatter().parse(template) if field)
        return names

    def select(self, task_name: str, *, sample_key: str, epoch: int) -> str:
        """Return the template this sample gets in this epoch.

        Args:
            task_name: Key into the registry.
            sample_key: The sample's ``__key__``; distinct samples decorrelate.
            epoch: Training epoch, so repeat views are phrased differently.

        Returns:
            One of the task's templates.

        Raises:
            KeyError: If the task is not in the registry.
        """
        templates = self.templates(task_name)
        digest = hashlib.sha1(f"{task_name}|{sample_key}|{epoch}".encode("utf-8")).digest()
        return templates[int.from_bytes(digest[:8], "big") % len(templates)]

    def render(
        self,
        task_name: str,
        *,
        sample_key: str,
        epoch: int,
        fields: dict[str, Any] | None = None,
    ) -> str:
        """Select a template and interpolate ``fields`` into it.

        Args:
            task_name: Key into the registry.
            sample_key: The sample's ``__key__``.
            epoch: Training epoch.
            fields: Values for the template's placeholders. Extra keys are
                ignored, so a caller may pass a superset.

        Returns:
            The formatted prompt.

        Raises:
            KeyError: If the task is unknown, or a placeholder the selected
                template needs is missing from ``fields``.
        """
        template = self.select(task_name, sample_key=sample_key, epoch=epoch)
        try:
            return template.format(**(fields or {}))
        except KeyError as exc:
            missing = exc.args[0]
            raise KeyError(
                f"Prompt for task {task_name!r} needs field {missing!r}, "
                f"which the sample did not supply. Template: {template!r}"
            ) from None
