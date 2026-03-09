"""
Enterprise authentication and authorization.

Provides pluggable identity providers (OIDC, LDAP) and role-based
access control (RBAC).  When ``auth_provider`` is ``"none"`` (the
default / mini profile), all guards are transparent pass-throughs.

Roles
─────
  admin             Full access to every endpoint.
  knowledge_curator Knowledge CRUD, no rule management.
  analyst           Read-only analytics and reports.
  user              Chat only – no admin surface.

Usage
─────
::

    from src.enterprise.auth import require_role, get_current_user

    @router.get("/admin/rules")
    async def list_rules(
        user: AuthenticatedUser = Depends(get_current_user),
        _: None = Depends(require_role("admin", "knowledge_curator")),
    ):
        ...
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.enterprise import is_enterprise_enabled

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

class Role(str, enum.Enum):
    ADMIN = "admin"
    KNOWLEDGE_CURATOR = "knowledge_curator"
    ANALYST = "analyst"
    USER = "user"


# Permissions per role (cumulative – admin inherits everything)
_ROLE_PERMISSIONS: dict[Role, set[str]] = {
    Role.ADMIN: {
        "rules:read", "rules:write", "rules:delete",
        "contexts:read", "contexts:write", "contexts:delete",
        "knowledge:read", "knowledge:write", "knowledge:verify", "knowledge:delete",
        "analytics:read",
        "admin:manage_users", "admin:manage_keys",
        "chat:access",
    },
    Role.KNOWLEDGE_CURATOR: {
        "knowledge:read", "knowledge:write", "knowledge:verify", "knowledge:delete",
        "contexts:read", "contexts:write",
        "analytics:read",
        "chat:access",
    },
    Role.ANALYST: {
        "analytics:read",
        "contexts:read",
        "knowledge:read",
        "rules:read",
        "chat:access",
    },
    Role.USER: {
        "chat:access",
    },
}


# ---------------------------------------------------------------------------
# User model
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AuthenticatedUser:
    """Represents a verified identity with role information."""

    id: str
    username: str
    email: str = ""
    role: Role = Role.USER
    tenant_id: str = "default"
    provider: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_permission(self, permission: str) -> bool:
        return permission in _ROLE_PERMISSIONS.get(self.role, set())

    def has_any_permission(self, *permissions: str) -> bool:
        role_perms = _ROLE_PERMISSIONS.get(self.role, set())
        return bool(role_perms & set(permissions))


# ---------------------------------------------------------------------------
# Anonymous / mini-mode fallback
# ---------------------------------------------------------------------------

_ANONYMOUS_USER = AuthenticatedUser(
    id="anonymous",
    username="anonymous",
    role=Role.ADMIN,
    provider="none",
)


# ---------------------------------------------------------------------------
# API key management
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ApiKeyRecord:
    """Stored API key entry."""

    key_hash: str
    name: str
    role: Role = Role.USER
    tenant_id: str = "default"
    created_at: float = 0.0
    expires_at: float | None = None
    enabled: bool = True


class ApiKeyManager:
    """In-memory API key store with constant-time lookup.

    In production this would be backed by the database; here we keep
    the interface simple for MVP.
    """

    def __init__(self) -> None:
        self._keys: dict[str, ApiKeyRecord] = {}

    @staticmethod
    def hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def create_key(
        self,
        name: str,
        role: Role = Role.USER,
        tenant_id: str = "default",
        expires_in_days: int | None = None,
    ) -> str:
        """Create a new API key and return the raw key (shown only once)."""
        raw = f"kh_{secrets.token_urlsafe(32)}"
        h = self.hash_key(raw)
        now = time.time()
        self._keys[h] = ApiKeyRecord(
            key_hash=h,
            name=name,
            role=role,
            tenant_id=tenant_id,
            created_at=now,
            expires_at=(now + expires_in_days * 86400) if expires_in_days else None,
            enabled=True,
        )
        logger.info("API key created: name=%s role=%s tenant=%s", name, role, tenant_id)
        return raw

    def validate(self, raw_key: str) -> ApiKeyRecord | None:
        """Validate a raw key and return its record, or ``None``."""
        h = self.hash_key(raw_key)
        record = self._keys.get(h)
        if record is None:
            return None
        if not record.enabled:
            return None
        if record.expires_at and time.time() > record.expires_at:
            return None
        return record

    def revoke(self, raw_key: str) -> bool:
        h = self.hash_key(raw_key)
        if h in self._keys:
            self._keys[h].enabled = False
            return True
        return False

    def list_keys(self) -> list[dict[str, Any]]:
        return [
            {
                "name": r.name,
                "role": r.role.value,
                "tenant_id": r.tenant_id,
                "enabled": r.enabled,
                "created_at": r.created_at,
                "expires_at": r.expires_at,
            }
            for r in self._keys.values()
        ]


# Singleton
_api_key_manager = ApiKeyManager()


def get_api_key_manager() -> ApiKeyManager:
    return _api_key_manager


# ---------------------------------------------------------------------------
# OIDC provider (stub – full implementation requires python-jose / authlib)
# ---------------------------------------------------------------------------

class OIDCProvider:
    """OpenID Connect token validation.

    In a real deployment this would:
    1. Fetch the JWKS from ``{issuer}/.well-known/openid-configuration``
    2. Validate the JWT signature, audience, issuer, and expiry
    3. Map claims to ``AuthenticatedUser``

    This stub parses the token structure for integration testing.
    """

    def __init__(self, issuer: str, client_id: str, role_claim: str = "role") -> None:
        self.issuer = issuer
        self.client_id = client_id
        self.role_claim = role_claim
        self._jwks_cache: dict[str, Any] | None = None

    async def validate_token(self, token: str) -> AuthenticatedUser | None:
        """Validate a bearer token and return an ``AuthenticatedUser``.

        Returns ``None`` when validation fails.
        """
        # NOTE: Full implementation would verify JWT signature here.
        # For now, log and reject – callers must override for real OIDC.
        logger.warning("OIDC token validation not fully implemented – rejecting token")
        return None


# ---------------------------------------------------------------------------
# LDAP provider (stub – full implementation requires ldap3)
# ---------------------------------------------------------------------------

class LDAPProvider:
    """LDAP / Active Directory authentication.

    In a real deployment this would:
    1. Bind to the directory with service credentials
    2. Search for the user DN
    3. Attempt a bind with the user's credentials
    4. Map LDAP groups to application roles

    This stub defines the interface for integration testing.
    """

    def __init__(
        self,
        server_url: str,
        base_dn: str,
        bind_dn: str = "",
        bind_password: str = "",
        user_search_filter: str = "(uid={username})",
        group_attribute: str = "memberOf",
    ) -> None:
        self.server_url = server_url
        self.base_dn = base_dn
        self.bind_dn = bind_dn
        self.bind_password = bind_password
        self.user_search_filter = user_search_filter
        self.group_attribute = group_attribute

    async def authenticate(self, username: str, password: str) -> AuthenticatedUser | None:
        """Authenticate a user against LDAP.

        Returns ``None`` when authentication fails.
        """
        # NOTE: Full implementation would perform LDAP bind here.
        logger.warning("LDAP authentication not fully implemented – rejecting credentials")
        return None

    def map_groups_to_role(self, groups: list[str]) -> Role:
        """Map LDAP groups to the highest-privilege application role."""
        group_set = {g.lower() for g in groups}
        if "knowledgehub-admins" in group_set:
            return Role.ADMIN
        if "knowledgehub-curators" in group_set:
            return Role.KNOWLEDGE_CURATOR
        if "knowledgehub-analysts" in group_set:
            return Role.ANALYST
        return Role.USER


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> AuthenticatedUser:
    """Resolve the current user from the request.

    Resolution order:
    1. If enterprise auth is disabled → anonymous admin (mini mode).
    2. ``Authorization: Bearer <token>`` → OIDC validation.
    3. ``X-API-Key`` header → API key lookup.
    4. Reject with 401.
    """
    if not is_enterprise_enabled("auth"):
        return _ANONYMOUS_USER

    # Try API key first (simpler path)
    api_key = request.headers.get("X-API-Key")
    if api_key:
        record = _api_key_manager.validate(api_key)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired API key",
            )
        return AuthenticatedUser(
            id=f"apikey:{record.name}",
            username=record.name,
            role=record.role,
            tenant_id=record.tenant_id,
            provider="api_key",
        )

    # Try Bearer token (OIDC)
    if credentials and credentials.credentials:
        from src.config.settings import get_settings
        settings = get_settings()
        ent = settings.enterprise
        if ent.auth_provider == "oidc":
            provider = OIDCProvider(
                issuer=ent.oidc_issuer,
                client_id=ent.oidc_client_id,
            )
            user = await provider.validate_token(credentials.credentials)
            if user:
                return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_role(*roles: str):
    """FastAPI dependency that enforces role-based access.

    Usage::

        @router.get("/admin/rules")
        async def list_rules(
            _: None = Depends(require_role("admin", "knowledge_curator")),
        ):
            ...
    """
    allowed = {Role(r) for r in roles}

    async def _check(user: AuthenticatedUser = Depends(get_current_user)):
        if user.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role.value}' does not have access. "
                       f"Required: {', '.join(r.value for r in allowed)}",
            )
        return user

    return _check


def require_permission(permission: str):
    """FastAPI dependency that enforces a specific permission.

    Usage::

        @router.delete("/knowledge/{item_id}")
        async def delete_item(
            _: None = Depends(require_permission("knowledge:delete")),
        ):
            ...
    """
    async def _check(user: AuthenticatedUser = Depends(get_current_user)):
        if not user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required",
            )
        return user

    return _check
