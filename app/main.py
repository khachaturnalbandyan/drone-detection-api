"""FastAPI application: HTTP layer over the detector, batcher and database.

Run locally:
    uvicorn app.main:app --reload
Interactive docs: http://localhost:8000/docs
"""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from sqlalchemy import select

from app.batching import MicroBatcher
from app.config import Settings, get_settings
from app.db import Prediction, init_db, make_engine, make_sessionmaker
from app.detectors import Detector, build_detector
from app.schemas import (
    BoundingBox,
    DetectionOut,
    DetectResponse,
    HealthResponse,
    PredictionRecord,
)


def create_app(settings: Settings | None = None, detector: Detector | None = None) -> FastAPI:
    """App factory. Tests pass their own settings and a fake detector."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup: load the model once, start the batch worker, prepare the DB.
        det = detector or build_detector(settings)
        batcher = MicroBatcher(det, settings.max_batch_size, settings.max_wait_ms)
        engine = make_engine(settings.database_url)
        await init_db(engine)
        await batcher.start()

        app.state.detector = det
        app.state.batcher = batcher
        app.state.sessionmaker = make_sessionmaker(engine)
        yield
        # Shutdown: stop the worker first so no request writes to a closed DB.
        await batcher.stop()
        await engine.dispose()

    app = FastAPI(
        title="Drone Detection API",
        version="1.0.0",
        description="YOLOv8 drone detector served with dynamic batching.",
        lifespan=lifespan,
    )
    max_upload_bytes = int(settings.max_upload_mb * 1024 * 1024)

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        return HealthResponse(
            status="ok",
            backend=request.app.state.detector.name,
            max_batch_size=settings.max_batch_size,
        )

    @app.post("/v1/detect", response_model=DetectResponse)
    async def detect(request: Request, file: Annotated[UploadFile, File()]) -> DetectResponse:
        # Read at most limit+1 bytes: enough to know the upload is too big
        # without loading an arbitrarily large body into memory.
        data = await file.read(max_upload_bytes + 1)
        if len(data) > max_upload_bytes:
            raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB")

        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(400, "Could not decode image (send a JPEG/PNG/BMP/WebP)")

        start = time.perf_counter()
        detections = await request.app.state.batcher.submit(image)
        latency_ms = (time.perf_counter() - start) * 1000  # includes batching wait

        out = [
            DetectionOut(
                box=BoundingBox(x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2),
                confidence=d.confidence,
                class_id=d.class_id,
                class_name=d.class_name,
            )
            for d in detections
        ]
        h, w = image.shape[:2]
        record = Prediction(
            filename=file.filename,
            width=w,
            height=h,
            backend=request.app.state.detector.name,
            latency_ms=latency_ms,
            num_detections=len(out),
            detections=[d.model_dump() for d in out],
        )
        async with request.app.state.sessionmaker() as session:
            session.add(record)
            await session.commit()

        return DetectResponse(
            id=record.id,
            width=w,
            height=h,
            backend=record.backend,
            latency_ms=round(latency_ms, 2),
            detections=out,
        )

    @app.get("/v1/predictions", response_model=list[PredictionRecord])
    async def list_predictions(
        request: Request,
        limit: int = Query(20, ge=1, le=200),
        only_with_detections: bool = False,
    ) -> list[Prediction]:
        """Most recent predictions first."""
        stmt = select(Prediction).order_by(Prediction.id.desc()).limit(limit)
        if only_with_detections:
            stmt = stmt.where(Prediction.num_detections > 0)
        async with request.app.state.sessionmaker() as session:
            return list((await session.scalars(stmt)).all())

    return app


app = create_app()
