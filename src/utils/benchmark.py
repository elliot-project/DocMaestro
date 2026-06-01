"""WandB experiment tracking and benchmarking helpers for MinerU pipeline."""

import os
from pathlib import Path

import wandb


def init_run(
    pdf_name: str,
    backend: str,
    extra_config: dict | None = None,
) -> wandb.sdk.wandb_run.Run:
    """Initialize a WandB run for a single extraction job.

    Args:
        pdf_name: Source PDF filename, used to tag the run.
        backend: MinerU backend name (e.g. 'hybrid-auto-engine').
        extra_config: Optional additional config key-value pairs to log.

    Returns:
        Active WandB Run object.
    """
    config = {
        "pdf_name": pdf_name,
        "backend": backend,
        **(extra_config or {}),
    }
    run = wandb.init(
        project="mineru-xml-pipeline",
        config=config,
        reinit=True,
    )
    return run  # type: ignore[return-value]


def log_extraction_metrics(
    duration_seconds: float,
    page_count: int,
    backend: str,
    pdf_path: Path,
    content_item_count: int,
) -> None:
    """Log core extraction performance metrics to WandB.

    Records engine execution time, per-page latency, throughput, and
    document metadata so runs can be compared across engine configurations.

    Args:
        duration_seconds: Total wall-clock time for the MinerU subprocess.
        page_count: Number of pages in the source document.
        backend: MinerU backend that was used.
        pdf_path: Absolute path to the source PDF.
        content_item_count: Number of content blocks in content_list.json.
    """
    latency_per_page = duration_seconds / max(page_count, 1)
    pdf_size_mb = os.path.getsize(pdf_path) / (1024 * 1024) if pdf_path.exists() else 0.0

    wandb.log(
        {
            f"extraction/{backend}/total_duration_s": duration_seconds,
            f"extraction/{backend}/latency_per_page_s": latency_per_page,
            f"extraction/{backend}/pages": page_count,
            f"extraction/{backend}/content_items": content_item_count,
            f"extraction/{backend}/pdf_size_mb": pdf_size_mb,
            "extraction/backend": backend,
        }
    )


def log_gpu_memory() -> None:
    """Log current GPU VRAM usage for all visible CUDA devices.

    Only runs when PyTorch is available and a CUDA device is present.
    Silently skips if torch or CUDA is unavailable.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        for i in range(torch.cuda.device_count()):
            used = torch.cuda.memory_allocated(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            wandb.log(
                {
                    f"gpu/{i}/vram_used_gb": round(used, 3),
                    f"gpu/{i}/vram_total_gb": round(total, 3),
                    f"gpu/{i}/vram_utilization": round(used / max(total, 1e-6), 4),
                }
            )
    except ImportError:
        pass


def log_xml_stats(xml_str: str, output_path: Path) -> None:
    """Log statistics about the synthesized XML output.

    Args:
        xml_str: The full XML document string.
        output_path: Path where the XML was saved.
    """
    xml_size_kb = len(xml_str.encode("utf-8")) / 1024
    wandb.log(
        {
            "xml/size_kb": round(xml_size_kb, 2),
            "xml/output_path": str(output_path),
        }
    )
