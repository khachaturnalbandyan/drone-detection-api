import cv2
import httpx
import numpy as np
import pytest

from app.config import Settings
from app.detectors import FakeDetector
from app.main import create_app


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        backend="fake",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        max_batch_size=8,
        max_wait_ms=20,
        max_upload_mb=1,
    )


@pytest.fixture
def fake_detector() -> FakeDetector:
    return FakeDetector(delay_s=0.05)


@pytest.fixture
async def client(settings, fake_detector):
    app = create_app(settings, detector=fake_detector)
    # LifespanManager-free: run the lifespan context directly so startup/shutdown happen.
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


def jpeg_bytes(width: int = 320, height: int = 240) -> bytes:
    img = np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()
