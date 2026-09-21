"""ASGI-level defense against oversized uploads.

Starlette's multipart parser applies no size cap to file parts (only
non-file form fields get its own max_part_size check) — a file part is
written straight into a SpooledTemporaryFile with no upper bound, and
FastAPI parses the full body (via request.form()) before any Depends()-based
check or per-route rate limiter ever runs. Enforcing the size limit inside
the endpoint, after `await file.read()`, is therefore too late: by then the
server has already paid the memory/disk cost of receiving the whole body.
This matters even more here than on the api service, since this service has
no auth of its own at all.

This middleware rejects an oversized request before any of that happens —
via Content-Length up front when the client declares one honestly, and via
a running byte count over the raw ASGI receive channel otherwise (chunked
transfer or a missing/lying Content-Length).
"""

from __future__ import annotations

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class _MaxBodySizeExceeded(Exception):
    pass


class MaxBodySizeMiddleware:
    def __init__(self, app: ASGIApp, max_body_size: int) -> None:
        self._app = app
        self._max_body_size = max_body_size

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        content_length = Headers(scope=scope).get("content-length")
        if content_length is not None and content_length.isdigit() and int(content_length) > self._max_body_size:
            await self._reject(scope, send)
            return

        total_size = 0

        async def limited_receive() -> Message:
            nonlocal total_size
            message = await receive()
            if message["type"] == "http.request":
                total_size += len(message.get("body", b""))
                if total_size > self._max_body_size:
                    raise _MaxBodySizeExceeded()
            return message

        try:
            await self._app(scope, limited_receive, send)
        except _MaxBodySizeExceeded:
            await self._reject(scope, send)

    async def _reject(self, scope: Scope, send: Send) -> None:
        response = JSONResponse(
            status_code=413,
            content={"detail": f"Request body exceeds the maximum allowed size of {self._max_body_size} bytes"},
        )
        await response(scope, None, send)
