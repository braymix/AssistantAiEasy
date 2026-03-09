"""
Enterprise multi-tenant support.

Provides tenant isolation for knowledge bases, detection rules, and
conversations.  When ``multitenancy_enabled`` is ``False`` (the default),
all data belongs to the ``"default"`` tenant and no filtering is applied.

Isolation model
───────────────
- Each tenant has its own knowledge base (vector store collection prefix)
- Detection rules can be tenant-specific or global (shared)
- Conversations are always tenant-scoped
- Global knowledge can be opted-in per tenant

Architecture
────────────
::

    ┌─────────────────────────────────────────┐
    │           Tenant Middleware              │
    │  (sets tenant_id on request.state)      │
    ├─────────────────────────────────────────┤
    │                                         │
    │  ┌─────────┐  ┌─────────┐  ┌────────┐  │
    │  │Tenant A │  │Tenant B │  │Global  │  │
    │  │Knowledge│  │Knowledge│  │Shared  │  │
    │  │  Base   │  │  Base   │  │  Base  │  │
    │  └─────────┘  └─────────┘  └────────┘  │
    │                                         │
    └─────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.enterprise import is_enterprise_enabled

logger = logging.getLogger(__name__)

# Default tenant used when multitenancy is disabled
DEFAULT_TENANT = "default"
GLOBAL_TENANT = "__global__"


# ---------------------------------------------------------------------------
# Tenant configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TenantConfig:
    """Per-tenant configuration and limits."""

    id: str
    name: str
    enabled: bool = True
    max_knowledge_items: int = 10_000
    max_rules: int = 100
    max_conversations: int = 50_000
    shared_global_knowledge: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def collection_prefix(self) -> str:
        """Vector store collection name for this tenant."""
        return f"kh_{self.id}"


# ---------------------------------------------------------------------------
# Tenant registry
# ---------------------------------------------------------------------------

class TenantRegistry:
    """In-memory tenant registry.

    In production this would be backed by the database.  The registry
    provides CRUD operations and validation for tenant configurations.
    """

    def __init__(self) -> None:
        self._tenants: dict[str, TenantConfig] = {
            DEFAULT_TENANT: TenantConfig(
                id=DEFAULT_TENANT,
                name="Default Tenant",
                shared_global_knowledge=True,
            ),
        }

    def get(self, tenant_id: str) -> TenantConfig | None:
        return self._tenants.get(tenant_id)

    def get_or_raise(self, tenant_id: str) -> TenantConfig:
        tenant = self.get(tenant_id)
        if tenant is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tenant '{tenant_id}' not found",
            )
        if not tenant.enabled:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Tenant '{tenant_id}' is disabled",
            )
        return tenant

    def create(
        self,
        tenant_id: str,
        name: str,
        **kwargs: Any,
    ) -> TenantConfig:
        if tenant_id in self._tenants:
            raise ValueError(f"Tenant '{tenant_id}' already exists")
        tenant = TenantConfig(id=tenant_id, name=name, **kwargs)
        self._tenants[tenant_id] = tenant
        logger.info("Tenant created: %s (%s)", tenant_id, name)
        return tenant

    def update(self, tenant_id: str, **kwargs: Any) -> TenantConfig:
        tenant = self.get(tenant_id)
        if tenant is None:
            raise ValueError(f"Tenant '{tenant_id}' not found")
        for key, value in kwargs.items():
            if hasattr(tenant, key):
                object.__setattr__(tenant, key, value)
        logger.info("Tenant updated: %s", tenant_id)
        return tenant

    def delete(self, tenant_id: str) -> bool:
        if tenant_id == DEFAULT_TENANT:
            raise ValueError("Cannot delete the default tenant")
        removed = self._tenants.pop(tenant_id, None)
        if removed:
            logger.info("Tenant deleted: %s", tenant_id)
        return removed is not None

    def list_all(self) -> list[TenantConfig]:
        return list(self._tenants.values())


# Singleton
_registry = TenantRegistry()


def get_tenant_registry() -> TenantRegistry:
    return _registry


# ---------------------------------------------------------------------------
# Tenant context helpers
# ---------------------------------------------------------------------------

def get_tenant_id(request: Request) -> str:
    """Extract the current tenant ID from a request.

    Resolution order:
    1. ``request.state.tenant_id`` (set by middleware)
    2. ``X-Tenant-ID`` header
    3. Fall back to ``DEFAULT_TENANT``
    """
    if not is_enterprise_enabled("multitenancy"):
        return DEFAULT_TENANT

    tenant_id = getattr(request.state, "tenant_id", None)
    if tenant_id:
        return tenant_id

    return request.headers.get("X-Tenant-ID", DEFAULT_TENANT)


def tenant_collection_name(tenant_id: str, base_collection: str) -> str:
    """Build a tenant-scoped vector store collection name.

    Examples::

        tenant_collection_name("acme", "knowledgehub")
        # → "kh_acme_knowledgehub"

        tenant_collection_name("default", "knowledgehub")
        # → "knowledgehub"  (no prefix for default)
    """
    if tenant_id == DEFAULT_TENANT:
        return base_collection
    return f"kh_{tenant_id}_{base_collection}"


# ---------------------------------------------------------------------------
# Tenant filter mixin (for SQLAlchemy queries)
# ---------------------------------------------------------------------------

class TenantFilterMixin:
    """Mixin providing tenant-aware query helpers.

    Models that support multitenancy should have a ``tenant_id`` column.
    Services inherit from this mixin to automatically scope queries.
    """

    def _apply_tenant_filter(self, query: Any, tenant_id: str) -> Any:
        """Apply tenant filtering to a SQLAlchemy query.

        When multitenancy is disabled, returns the query unchanged.
        When the tenant opts into global knowledge, includes both
        tenant-specific and global items.
        """
        if not is_enterprise_enabled("multitenancy"):
            return query

        tenant = _registry.get(tenant_id)
        if tenant and tenant.shared_global_knowledge:
            # Include both tenant-specific and global items
            return query.filter(
                query.column_descriptions[0]["entity"].tenant_id.in_(
                    [tenant_id, GLOBAL_TENANT]
                )
            )

        return query.filter(
            query.column_descriptions[0]["entity"].tenant_id == tenant_id
        )


# ---------------------------------------------------------------------------
# FastAPI middleware
# ---------------------------------------------------------------------------

class TenantMiddleware(BaseHTTPMiddleware):
    """Middleware that resolves and validates the tenant for each request.

    Sets ``request.state.tenant_id`` for downstream handlers.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        if not is_enterprise_enabled("multitenancy"):
            request.state.tenant_id = DEFAULT_TENANT
            return await call_next(request)

        # Resolve tenant from header or authenticated user
        tenant_id = request.headers.get("X-Tenant-ID", DEFAULT_TENANT)

        # Validate tenant exists and is enabled
        tenant = _registry.get(tenant_id)
        if tenant is None:
            return Response(
                content=f'{{"detail": "Unknown tenant: {tenant_id}"}}',
                status_code=status.HTTP_400_BAD_REQUEST,
                media_type="application/json",
            )
        if not tenant.enabled:
            return Response(
                content=f'{{"detail": "Tenant {tenant_id} is disabled"}}',
                status_code=status.HTTP_403_FORBIDDEN,
                media_type="application/json",
            )

        request.state.tenant_id = tenant_id
        logger.debug("Request scoped to tenant: %s", tenant_id)
        return await call_next(request)
