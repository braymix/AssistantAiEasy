"""Shared response envelope and pagination types for the Admin API."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginationMeta(BaseModel):
    """Pagination metadata included in list responses."""

    total: int
    limit: int
    offset: int


class ApiResponse(BaseModel, Generic[T]):
    """Standard API response envelope.

    Every admin endpoint returns ``{"data": ..., "meta": ...}``.
    """

    data: T
    meta: PaginationMeta | None = None


class ErrorDetail(BaseModel):
    detail: str
