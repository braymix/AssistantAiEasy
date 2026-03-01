"""Request / response logging middleware with timing metrics and correlation ID."""

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Attaches a unique correlation ID, logs every request/response,
    and records wall-clock latency in an ``X-Process-Time`` header.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # ── Correlation ID ─────────────────────────────────────────────
        correlation_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            correlation_id=correlation_id,
            method=request.method,
            path=request.url.path,
        )

        logger = structlog.get_logger("gateway.request")
        logger.info("request_started")

        # ── Call downstream ────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error("request_failed", elapsed_ms=round(elapsed_ms, 2))
            raise

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # ── Response headers ───────────────────────────────────────────
        response.headers["X-Request-ID"] = correlation_id
        response.headers["X-Process-Time"] = f"{elapsed_ms:.2f}ms"

        logger.info(
            "request_completed",
            status=response.status_code,
            elapsed_ms=round(elapsed_ms, 2),
        )
        return response
