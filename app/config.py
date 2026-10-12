"""Application settings, read from environment variables (or a .env file).

Every setting can be overridden with an env var prefixed by ``DDA_``,
e.g. ``DDA_BACKEND=torch`` or ``DDA_MAX_BATCH_SIZE=16``.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DDA_", env_file=".env", extra="ignore")

    # --- model -------------------------------------------------------------
    # "onnx"  -> onnxruntime, no PyTorch needed (default; small Docker image)
    # "torch" -> Ultralytics/PyTorch, reads the original best.pt
    # "fake"  -> deterministic dummy detector, used by tests and CI
    backend: Literal["onnx", "torch", "fake"] = "onnx"
    model_path: str = "models/best.onnx"
    imgsz: int = 640  # network input size (square), must match the export
    conf_threshold: float = 0.4  # same defaults as detect.py in drone-detection
    iou_threshold: float = 0.45
    max_detections: int = 3

    # --- dynamic batching ----------------------------------------------------
    # Requests arriving within `max_wait_ms` of each other are run as one batch
    # of up to `max_batch_size` images.
    max_batch_size: int = 8
    max_wait_ms: float = 10.0

    # --- API -------------------------------------------------------------------
    max_upload_mb: float = 10.0

    # --- storage ---------------------------------------------------------------
    # SQLite for local dev; docker-compose overrides this with PostgreSQL.
    database_url: str = "sqlite+aiosqlite:///./predictions.db"


@lru_cache
def get_settings() -> Settings:
    """Settings are parsed once and cached for the lifetime of the process."""
    return Settings()
