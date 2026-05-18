"""Pipeline configuration: dataclasses and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    )

    return PipelineConfig(
        extraction=ExtractionConfig(
            backend=extraction_raw.get("backend", "hybrid-auto-engine"),
            language=extraction_raw.get("language", "en"),
            start_page=extraction_raw.get("start_page", 0),
            end_page=extraction_raw.get("end_page", None),
            server_url=extraction_raw.get("server_url", None),
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


def collect_pdfs(input_path: Path) -> list[Path]:
    """Collect PDF paths from a single file or a directory.

    Args:
        input_path: Path to a PDF file or a directory containing PDFs.

    Returns:
        Sorted list of PDF paths.

    Raises:
        ValueError: If no PDF files are found.
    """
    if input_path.is_file():
        return [input_path]
    pdfs = sorted(input_path.glob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No PDF files found in {input_path}")
    return pdfs
