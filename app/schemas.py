"""Pydantic models that define the API's JSON shapes (and the OpenAPI docs)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionOut(BaseModel):
    box: BoundingBox
    confidence: float
    class_id: int
    class_name: str


class DetectResponse(BaseModel):
    id: int
    width: int
    height: int
    backend: str
    latency_ms: float
    detections: list[DetectionOut]


class PredictionRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    filename: str | None
    width: int
    height: int
    backend: str
    latency_ms: float
    num_detections: int
    detections: list[DetectionOut]


class HealthResponse(BaseModel):
    status: str
    backend: str
    max_batch_size: int
