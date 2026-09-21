from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from ocr_app.max_body_size_middleware import MaxBodySizeMiddleware


async def _echo(request):
    body = await request.body()
    return PlainTextResponse(f"received {len(body)} bytes")


def _build_app(max_body_size: int) -> Starlette:
    app = Starlette(routes=[Route("/echo", _echo, methods=["POST"])])
    app.add_middleware(MaxBodySizeMiddleware, max_body_size=max_body_size)
    return app


def test_request_within_limit_is_allowed():
    client = TestClient(_build_app(max_body_size=100))
    resp = client.post("/echo", content=b"x" * 50)
    assert resp.status_code == 200
    assert resp.text == "received 50 bytes"


def test_request_over_limit_rejected_via_content_length():
    client = TestClient(_build_app(max_body_size=100))
    resp = client.post("/echo", content=b"x" * 200)
    assert resp.status_code == 413


def test_request_over_limit_rejected_without_content_length():
    client = TestClient(_build_app(max_body_size=100))

    def body_stream():
        yield b"x" * 60
        yield b"x" * 60

    resp = client.post("/echo", content=body_stream())
    assert resp.status_code == 413
