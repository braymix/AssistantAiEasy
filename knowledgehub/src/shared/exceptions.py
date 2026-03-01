"""Application-level exceptions for KnowledgeHub."""

from fastapi import HTTPException, status


class KnowledgeHubError(Exception):
    """Base exception for all KnowledgeHub errors."""

    def __init__(self, message: str = "An error occurred"):
        self.message = message
        super().__init__(self.message)


class NotFoundError(KnowledgeHubError):
    """Raised when a requested resource is not found."""

    def __init__(self, resource: str = "Resource", identifier: str = ""):
        detail = f"{resource} not found"
        if identifier:
            detail = f"{resource} '{identifier}' not found"
        super().__init__(detail)


class VectorStoreError(KnowledgeHubError):
    """Raised when vector store operations fail."""


class LLMError(KnowledgeHubError):
    """Raised when LLM operations fail."""


class DetectionError(KnowledgeHubError):
    """Raised when context detection fails."""


class ConfigurationError(KnowledgeHubError):
    """Raised for invalid configuration states."""


# ---------------------------------------------------------------------------
# HTTP exception helpers
# ---------------------------------------------------------------------------


def not_found(detail: str = "Resource not found") -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


def bad_request(detail: str = "Bad request") -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def internal_error(detail: str = "Internal server error") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=detail,
    )
