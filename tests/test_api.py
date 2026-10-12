import asyncio

from tests.conftest import jpeg_bytes


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "backend": "fake", "max_batch_size": 8}


async def test_detect_returns_boxes_in_original_coordinates(client):
    r = await client.post(
        "/v1/detect", files={"file": ("a.jpg", jpeg_bytes(320, 240), "image/jpeg")}
    )
    assert r.status_code == 200
    body = r.json()
    assert (body["width"], body["height"]) == (320, 240)
    assert body["detections"][0]["box"] == {"x1": 80, "y1": 60, "x2": 240, "y2": 180}
    assert body["detections"][0]["class_name"] == "drone"


async def test_invalid_image_is_rejected(client):
    r = await client.post("/v1/detect", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


async def test_too_large_upload_is_rejected(client):
    big = b"\0" * (1024 * 1024 + 1)  # limit is 1 MB in the test settings
    r = await client.post("/v1/detect", files={"file": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413


async def test_predictions_are_logged_newest_first(client):
    for name in ("first.jpg", "second.jpg"):
        await client.post("/v1/detect", files={"file": (name, jpeg_bytes(), "image/jpeg")})
    r = await client.get("/v1/predictions", params={"limit": 10})
    assert r.status_code == 200
    records = r.json()
    assert [rec["filename"] for rec in records] == ["second.jpg", "first.jpg"]
    assert records[0]["num_detections"] == 1 and records[0]["backend"] == "fake"


async def test_parallel_http_requests_share_a_batch(client, fake_detector):
    files = [{"file": (f"{i}.jpg", jpeg_bytes(), "image/jpeg")} for i in range(6)]
    responses = await asyncio.gather(*(client.post("/v1/detect", files=f) for f in files))
    assert all(r.status_code == 200 for r in responses)
    assert max(fake_detector.batch_sizes) > 1  # requests were grouped
