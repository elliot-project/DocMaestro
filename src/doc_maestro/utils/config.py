"""Pipeline configuration: dataclasses and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

import yaml


@dataclass
class ExtractionConfig:
    """MinerU extraction settings."""

    backend: str = "hybrid-auto-engine"
    language: str = "en"
    start_page: int = 0
    end_page: int | None = None
    server_url: str | None = None
    batch_size: int = 1
    window_size: int | None = None
    gpu_memory_utilization: float | None = None


@dataclass
class TitleAidedConfig:
    """LLM-aided title hierarchy (H1–H4) settings."""

    enabled: bool = False
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    enable_thinking: bool = False

    def as_mineru_config(self) -> dict | None:
        """Return the dict expected by MinerU's title-aided API, or None if disabled.

        Returns:
            Dict with api_key, base_url, model (and optionally enable_thinking),
            or None when title-aided is disabled.
        """
        if not self.enabled:
            return None
        cfg = {"api_key": self.api_key, "base_url": self.base_url, "model": self.model}
        if self.enable_thinking:
            cfg["enable_thinking"] = True
        return cfg


@dataclass
class FigureClassificationConfig:
    """SigLIP2 zero-shot figure classification settings."""

    enabled: bool = False
    model: str = "google/siglip2-base-patch16-384"
    batch_size: int = 32
    device: str = "cuda"


@dataclass
class PathsConfig:
    """Directory paths for pipeline inputs and outputs."""

    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    state_dir: Path = field(default_factory=lambda: Path("state"))
    exclude: list[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Full pipeline configuration, mirroring the YAML schema."""

    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    title_aided: TitleAidedConfig = field(default_factory=TitleAidedConfig)
    figure_classification: FigureClassificationConfig = field(
        default_factory=FigureClassificationConfig
    )
    paths: PathsConfig = field(default_factory=PathsConfig)
    wandb_enabled: bool = True


def load_config(config_path: Path) -> PipelineConfig:
    """Load a YAML config file and return a PipelineConfig.

    Missing keys fall back to dataclass defaults, so a minimal config file
    that only sets the fields it cares about is valid.

    Args:
        config_path: Path to a YAML configuration file.

    Returns:
        Populated PipelineConfig.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML contains unrecognised top-level keys.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw: dict = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    extraction_raw = raw.get("extraction", {})
    title_aided_raw = raw.get("title_aided", {})
    figure_raw = raw.get("figure_classification", {})
    paths_raw = raw.get("paths", {})

    paths = PathsConfig(
        data_dir=Path(paths_raw.get("data_dir", "data")),
        output_dir=Path(paths_raw.get("output_dir", "output")),
        state_dir=Path(paths_raw.get("state_dir", "state")),
        exclude=list(paths_raw.get("exclude", []) or []),
    )

    return PipelineConfig(
        extraction=ExtractionConfig(
            backend=extraction_raw.get("backend", "hybrid-auto-engine"),
            language=extraction_raw.get("language", "en"),
            start_page=extraction_raw.get("start_page", 0),
            end_page=extraction_raw.get("end_page", None),
            server_url=extraction_raw.get("server_url", None),
            batch_size=extraction_raw.get("batch_size", 1),
            window_size=extraction_raw.get("window_size", None),
            gpu_memory_utilization=extraction_raw.get("gpu_memory_utilization", None),
        ),
        title_aided=TitleAidedConfig(
            enabled=title_aided_raw.get("enabled", False),
            api_key=title_aided_raw.get("api_key", None),
            base_url=title_aided_raw.get("base_url", None),
            model=title_aided_raw.get("model", None),
            enable_thinking=title_aided_raw.get("enable_thinking", False),
        ),
        figure_classification=FigureClassificationConfig(
            enabled=figure_raw.get("enabled", False),
            model=figure_raw.get("model", "google/siglip2-base-patch16-384"),
            batch_size=figure_raw.get("batch_size", 32),
            device=figure_raw.get("device", "cuda"),
        ),
        paths=paths,
        wandb_enabled=raw.get("wandb_enabled", True),
    )


def chunked(lst: list, n: int) -> list[list]:
    """Split lst into chunks of at most n items.

    Args:
        lst: Input list.
        n: Maximum chunk size (must be >= 1).

    Returns:
        List of sub-lists; the last chunk may be shorter than n.
    """
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def collect_pdfs(input_path: Path, exclude: list[str] | None = None) -> list[Path]:
    """Collect PDF paths from a single file or a directory.

    Args:
        input_path: Path to a PDF file or a directory containing PDFs.
        exclude: Patterns matched against each PDF's path *relative to*
            ``input_path``, in posix form. A pattern is a match when it
            fnmatch-matches the whole relative path (``"*/raw/*"``) or names a
            directory prefix of it (``"tools"``, ``"tools/raw"``). Matching
            files are dropped. Ignored when ``input_path`` is a single file,
            since naming a PDF explicitly is an unambiguous request for it.

    Returns:
        Sorted list of PDF paths.

    Raises:
        ValueError: If no PDF files are found, or all of them were excluded.
    """
    if input_path.is_file():
        return [input_path]
    pdfs = sorted(input_path.rglob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No PDF files found in {input_path}")
    if exclude:
        pdfs = [p for p in pdfs if not _is_excluded(p.relative_to(input_path), exclude)]
        if not pdfs:
            raise ValueError(f"All PDF files in {input_path} were excluded by {exclude}")
    return pdfs


def _is_excluded(rel: Path, patterns: list[str]) -> bool:
    """Return whether a relative PDF path matches any exclude pattern.

    Args:
        rel: PDF path relative to the input root.
        patterns: Exclude patterns; see :func:`collect_pdfs`.

    Returns:
        ``True`` when the path should be dropped.
    """
    rel_posix = rel.as_posix()
    for pattern in patterns:
        pattern = pattern.strip().rstrip("/")
        if not pattern:
            continue
        if fnmatch(rel_posix, pattern):
            return True
        if rel_posix.startswith(f"{pattern}/"):
            return True
    return False
