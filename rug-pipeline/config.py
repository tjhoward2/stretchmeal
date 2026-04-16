"""Centralized configuration for the rug design pipeline.

All paths and thresholds are configurable here. In production, paths will
point to S3-backed mounts; for now they're local filesystem directories.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """Pipeline-wide configuration with sensible defaults."""

    # --- Directory layout ---
    base_dir: Path = field(default_factory=lambda: Path("./pipeline_data"))

    @property
    def masters_dir(self) -> Path:
        return self.base_dir / "masters"

    @property
    def archive_dir(self) -> Path:
        return self.base_dir / "archive"

    @property
    def cache_dir(self) -> Path:
        return self.base_dir / "cache"

    # --- Ingest thresholds ---
    validation_ssim_threshold: float = 0.95
    thumbnail_width_px: int = 1000

    # --- Border detection ---
    border_detection_width_px: int = 2000
    border_confidence_threshold: float = 0.7
    canny_low: int = 50
    canny_high: int = 150
    gradient_weight: float = 0.6
    edge_weight: float = 0.4

    # --- Scaling ---
    default_dpi: int = 150
    min_interior_px: int = 100
    seam_threshold: float = 5.0  # max avg color diff (0-255) along seam

    # --- Cache ---
    cache_max_age_days: int = 30

    # --- Real-ESRGAN ---
    realesrgan_binary: str = "realesrgan-ncnn-vulkan"
    realesrgan_scale: int = 4  # 4x upscale factor
    realesrgan_model: str = "realesrgan-x4plus"
    realesrgan_models_dir: str | None = None  # auto-detects from binary location

    # --- Storage backend ---
    storage_backend: str = "local"  # "local" or "s3"
    s3_masters_bucket: str = "rug-masters"
    s3_archive_bucket: str = "rug-archive"
    s3_region: str = "us-east-1"
    s3_glacier_transition_days: int = 1  # days before archive → Glacier Deep Archive

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        self.masters_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
