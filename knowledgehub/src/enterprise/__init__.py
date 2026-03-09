"""
KnowledgeHub Enterprise Features.

Enterprise features are disabled in the ``mini`` profile and enabled
progressively in the ``full`` profile via feature flags.  Every module
in this package checks its own flag before doing anything, so the rest
of the application can import enterprise helpers unconditionally and
they will gracefully degrade to no-ops.

Feature flags
─────────────
Each flag maps to a Settings / EnterpriseSettings field.  When the flag
evaluates to ``False`` the corresponding subsystem is a transparent
pass-through.

  ┌──────────────────┬──────────────────────────────┬──────────┐
  │ Feature          │ Flag                         │ Default  │
  ├──────────────────┼──────────────────────────────┼──────────┤
  │ Advanced auth    │ enterprise.auth_provider      │ "none"   │
  │ Audit trail      │ enterprise.audit_enabled      │ False    │
  │ Multi-tenancy    │ enterprise.multitenancy_enabled│ False   │
  │ Clustering / HA  │ enterprise.cluster_enabled    │ False    │
  │ Monitoring       │ enterprise.metrics_enabled    │ True     │
  │ Enterprise backup│ enterprise.backup_enabled     │ False    │
  └──────────────────┴──────────────────────────────┴──────────┘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature flag registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FeatureFlags:
    """Snapshot of all enterprise feature flags.

    Computed once from settings and cached for the process lifetime.
    """

    auth_enabled: bool = False
    audit_enabled: bool = False
    multitenancy_enabled: bool = False
    cluster_enabled: bool = False
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    backup_enabled: bool = False

    def summary(self) -> dict[str, bool]:
        """Return a dict suitable for health-check / status endpoints."""
        return {
            "auth": self.auth_enabled,
            "audit": self.audit_enabled,
            "multitenancy": self.multitenancy_enabled,
            "clustering": self.cluster_enabled,
            "metrics": self.metrics_enabled,
            "tracing": self.tracing_enabled,
            "backup": self.backup_enabled,
        }


@lru_cache
def get_feature_flags() -> FeatureFlags:
    """Build feature flags from the current settings.

    Safe to call in ``mini`` profile – all enterprise flags default to
    off when ``EnterpriseSettings`` fields are absent or at their
    defaults.
    """
    from src.config.settings import get_settings

    settings = get_settings()
    ent = getattr(settings, "enterprise", None)

    if ent is None:
        logger.debug("No enterprise settings found – all features disabled")
        return FeatureFlags()

    flags = FeatureFlags(
        auth_enabled=ent.auth_provider != "none",
        audit_enabled=ent.audit_enabled,
        multitenancy_enabled=ent.multitenancy_enabled,
        cluster_enabled=ent.cluster_enabled,
        metrics_enabled=ent.metrics_enabled,
        tracing_enabled=ent.tracing_enabled,
        backup_enabled=ent.backup_enabled,
    )
    logger.info("Enterprise features: %s", flags.summary())
    return flags


def require_enterprise(feature: str) -> None:
    """Raise ``RuntimeError`` if the named feature is not enabled.

    Parameters
    ----------
    feature:
        One of the keys returned by ``FeatureFlags.summary()``.

    Raises
    ------
    RuntimeError
        When the requested feature flag is ``False``.
    """
    flags = get_feature_flags()
    enabled = flags.summary().get(feature, False)
    if not enabled:
        raise RuntimeError(
            f"Enterprise feature '{feature}' is not enabled. "
            f"Set the corresponding flag in EnterpriseSettings to activate it."
        )


def is_enterprise_enabled(feature: str) -> bool:
    """Check whether *feature* is active without raising."""
    flags = get_feature_flags()
    return flags.summary().get(feature, False)
