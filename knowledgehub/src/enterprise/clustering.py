"""
Enterprise high-availability and clustering.

Provides distributed coordination for multi-instance deployments
using Redis as the shared state backend.  When ``cluster_enabled``
is ``False`` (the default), all operations are local no-ops.

Features
────────
- Leader election via Redis distributed lock
- Distributed caching with automatic invalidation
- Session affinity tracking
- Health-based routing metadata

Architecture
────────────
::

    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Gateway  │    │ Gateway  │    │ Gateway  │
    │ Node 1   │    │ Node 2   │    │ Node 3   │
    │ (leader) │    │(follower)│    │(follower)│
    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
                  ┌──────┴──────┐
                  │    Redis    │
                  │  (shared)   │
                  └─────────────┘
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.enterprise import is_enterprise_enabled

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node identity
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class NodeInfo:
    """Identity and health metadata for a cluster node."""

    node_id: str
    hostname: str
    port: int
    started_at: float
    is_leader: bool = False
    healthy: bool = True
    active_connections: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "port": self.port,
            "started_at": self.started_at,
            "is_leader": self.is_leader,
            "healthy": self.healthy,
            "active_connections": self.active_connections,
        }


# ---------------------------------------------------------------------------
# Distributed cache
# ---------------------------------------------------------------------------

class DistributedCache:
    """Redis-backed cache with local fallback.

    When Redis is unavailable or clustering is disabled, falls back to
    a local in-memory dict with TTL eviction.
    """

    def __init__(self, redis_url: str = "", prefix: str = "kh:cache:") -> None:
        self._redis_url = redis_url
        self._prefix = prefix
        self._redis: Any = None
        self._local: dict[str, tuple[Any, float]] = {}  # key → (value, expires_at)

    async def connect(self) -> bool:
        """Connect to Redis.  Returns ``False`` on failure."""
        if not self._redis_url:
            return False
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            await self._redis.ping()
            logger.info("Distributed cache connected to Redis")
            return True
        except Exception as exc:
            logger.warning("Redis connection failed, using local cache: %s", exc)
            self._redis = None
            return False

    async def get(self, key: str) -> Any | None:
        """Get a cached value by key."""
        full_key = f"{self._prefix}{key}"

        if self._redis:
            try:
                raw = await self._redis.get(full_key)
                return json.loads(raw) if raw else None
            except Exception:
                pass

        # Local fallback
        entry = self._local.get(full_key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at and time.time() > expires_at:
            self._local.pop(full_key, None)
            return None
        return value

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """Set a cache value with TTL in seconds."""
        full_key = f"{self._prefix}{key}"

        if self._redis:
            try:
                await self._redis.set(full_key, json.dumps(value), ex=ttl)
                return
            except Exception:
                pass

        # Local fallback
        self._local[full_key] = (value, time.time() + ttl)

    async def delete(self, key: str) -> None:
        """Delete a cached value."""
        full_key = f"{self._prefix}{key}"

        if self._redis:
            try:
                await self._redis.delete(full_key)
            except Exception:
                pass

        self._local.pop(full_key, None)

    async def invalidate_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern.  Returns count deleted."""
        full_pattern = f"{self._prefix}{pattern}"
        count = 0

        if self._redis:
            try:
                async for key in self._redis.scan_iter(full_pattern):
                    await self._redis.delete(key)
                    count += 1
                return count
            except Exception:
                pass

        # Local fallback (simple prefix match since we can't glob)
        prefix = full_pattern.replace("*", "")
        to_delete = [k for k in self._local if k.startswith(prefix)]
        for k in to_delete:
            self._local.pop(k, None)
            count += 1
        return count

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()
            self._redis = None


# ---------------------------------------------------------------------------
# Leader election
# ---------------------------------------------------------------------------

class LeaderElection:
    """Redis-based leader election using distributed locks.

    The leader holds a Redis key with a TTL.  It must renew the lock
    periodically (heartbeat).  If the leader fails to renew, another
    node can acquire the lock.
    """

    LOCK_KEY = "kh:cluster:leader"
    LOCK_TTL = 30  # seconds
    HEARTBEAT_INTERVAL = 10  # seconds

    def __init__(self, node_id: str, redis_url: str = "") -> None:
        self.node_id = node_id
        self._redis_url = redis_url
        self._redis: Any = None
        self._is_leader = False
        self._heartbeat_task: asyncio.Task | None = None

    @property
    def is_leader(self) -> bool:
        return self._is_leader

    async def connect(self) -> bool:
        if not self._redis_url:
            # No Redis → single node is always leader
            self._is_leader = True
            return True
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            await self._redis.ping()
            return True
        except Exception as exc:
            logger.warning("Leader election: Redis unavailable (%s), assuming leader", exc)
            self._is_leader = True
            return False

    async def try_acquire(self) -> bool:
        """Attempt to become the leader."""
        if not self._redis:
            self._is_leader = True
            return True

        try:
            acquired = await self._redis.set(
                self.LOCK_KEY,
                self.node_id,
                nx=True,
                ex=self.LOCK_TTL,
            )
            if acquired:
                self._is_leader = True
                logger.info("Node %s elected as leader", self.node_id)
            else:
                current_leader = await self._redis.get(self.LOCK_KEY)
                self._is_leader = current_leader == self.node_id
            return self._is_leader
        except Exception as exc:
            logger.error("Leader election failed: %s", exc)
            return False

    async def renew(self) -> bool:
        """Renew the leader lock.  Must be called periodically."""
        if not self._redis or not self._is_leader:
            return self._is_leader

        try:
            current = await self._redis.get(self.LOCK_KEY)
            if current == self.node_id:
                await self._redis.expire(self.LOCK_KEY, self.LOCK_TTL)
                return True
            self._is_leader = False
            return False
        except Exception:
            return False

    async def release(self) -> None:
        """Voluntarily release leadership."""
        if self._redis and self._is_leader:
            try:
                current = await self._redis.get(self.LOCK_KEY)
                if current == self.node_id:
                    await self._redis.delete(self.LOCK_KEY)
            except Exception:
                pass
        self._is_leader = False

    async def start_heartbeat(self) -> None:
        """Start the background heartbeat task."""
        if self._heartbeat_task is not None:
            return

        async def _beat():
            while True:
                await self.try_acquire()
                if self._is_leader:
                    await self.renew()
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)

        self._heartbeat_task = asyncio.create_task(_beat())

    async def stop_heartbeat(self) -> None:
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        await self.release()


# ---------------------------------------------------------------------------
# Session affinity
# ---------------------------------------------------------------------------

class SessionAffinityTracker:
    """Track which node handles a given session for sticky routing.

    Stores a mapping of ``session_id → node_id`` in Redis so a load
    balancer can route subsequent requests to the same node.
    """

    KEY_PREFIX = "kh:session_affinity:"
    SESSION_TTL = 3600  # 1 hour

    def __init__(self, cache: DistributedCache) -> None:
        self._cache = cache

    async def bind(self, session_id: str, node_id: str) -> None:
        await self._cache.set(
            f"affinity:{session_id}", node_id, ttl=self.SESSION_TTL,
        )

    async def get_node(self, session_id: str) -> str | None:
        return await self._cache.get(f"affinity:{session_id}")

    async def unbind(self, session_id: str) -> None:
        await self._cache.delete(f"affinity:{session_id}")


# ---------------------------------------------------------------------------
# Cluster manager (facade)
# ---------------------------------------------------------------------------

class ClusterManager:
    """High-level facade for all clustering features.

    Initialise once on startup and access sub-systems through
    properties.
    """

    def __init__(self, redis_url: str = "") -> None:
        self.node_id = f"node-{uuid.uuid4().hex[:8]}"
        self.hostname = os.environ.get("HOSTNAME", "localhost")
        self.port = int(os.environ.get("KNOWLEDGEHUB_GATEWAY_PORT", "8000"))
        self._started_at = time.time()
        self._redis_url = redis_url

        self.cache = DistributedCache(redis_url=redis_url)
        self.election = LeaderElection(node_id=self.node_id, redis_url=redis_url)
        self.affinity = SessionAffinityTracker(cache=self.cache)

    async def start(self) -> None:
        """Initialise clustering subsystems."""
        if not is_enterprise_enabled("clustering"):
            logger.debug("Clustering disabled – running in single-node mode")
            return

        await self.cache.connect()
        await self.election.connect()
        await self.election.start_heartbeat()
        logger.info(
            "Cluster node started: id=%s hostname=%s",
            self.node_id, self.hostname,
        )

    async def stop(self) -> None:
        """Gracefully shut down clustering."""
        await self.election.stop_heartbeat()
        await self.cache.close()
        logger.info("Cluster node stopped: %s", self.node_id)

    @property
    def node_info(self) -> NodeInfo:
        return NodeInfo(
            node_id=self.node_id,
            hostname=self.hostname,
            port=self.port,
            started_at=self._started_at,
            is_leader=self.election.is_leader,
        )

    async def register_node(self) -> None:
        """Register this node in the cluster registry."""
        await self.cache.set(
            f"nodes:{self.node_id}",
            self.node_info.to_dict(),
            ttl=LeaderElection.LOCK_TTL * 2,
        )

    async def get_cluster_status(self) -> dict[str, Any]:
        """Return the current cluster status."""
        return {
            "node": self.node_info.to_dict(),
            "cluster_enabled": is_enterprise_enabled("clustering"),
            "redis_connected": self.cache._redis is not None,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_cluster_manager: ClusterManager | None = None


def get_cluster_manager(redis_url: str = "") -> ClusterManager:
    global _cluster_manager
    if _cluster_manager is None:
        _cluster_manager = ClusterManager(redis_url=redis_url)
    return _cluster_manager
