"""Admin API dependencies – authentication and common DI helpers."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.config import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str:
    """Validate the ``X-API-Key`` header against configured keys.

    Raises 401 if the key is missing or not in the allowed list.
    If ``settings.api_keys`` is empty the endpoint is **closed** by default
    (no open-door policy).
    """
    settings = get_settings()

    if not settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No API keys configured – admin access disabled",
        )

    if api_key is None or api_key not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    return api_key
