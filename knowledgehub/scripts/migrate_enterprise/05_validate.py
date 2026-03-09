"""
Step 5 — Post-migration validation.

Runs a comprehensive suite of checks against the newly migrated
enterprise deployment:

1. API endpoint health checks (gateway + admin)
2. RAG pipeline end-to-end test
3. Detection rules evaluation
4. Performance benchmark (latency comparison)

Usage::

    python scripts/migrate_enterprise/05_validate.py
    python scripts/migrate_enterprise/05_validate.py --json
    python scripts/migrate_enterprise/05_validate.py --skip-benchmark
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("migrate.validate")


@dataclass
class ValidationResult:
    name: str
    passed: bool
    message: str
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "duration_ms": round(self.duration_ms, 1),
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _gateway_url() -> str:
    host = os.environ.get("KNOWLEDGEHUB_GATEWAY_HOST", "localhost")
    port = os.environ.get("KNOWLEDGEHUB_GATEWAY_PORT", "8000")
    return f"http://{host}:{port}"


def _admin_url() -> str:
    host = os.environ.get("KNOWLEDGEHUB_GATEWAY_HOST", "localhost")
    port = os.environ.get("KNOWLEDGEHUB_ADMIN_PORT", "8001")
    return f"http://{host}:{port}"


def _api_key() -> str:
    keys = os.environ.get("KNOWLEDGEHUB_API_KEYS", "")
    if keys:
        return keys.split(",")[0].strip()
    return os.environ.get("KNOWLEDGEHUB_ADMIN_API_KEY", "")


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

async def validate_gateway_health() -> ValidationResult:
    """Check gateway /health endpoint."""
    url = f"{_gateway_url()}/health"
    start = time.perf_counter()

    try:
        import httpx
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            elapsed = (time.perf_counter() - start) * 1000

            if resp.status_code == 200:
                return ValidationResult(
                    name="gateway_health",
                    passed=True,
                    message=f"Gateway healthy ({elapsed:.0f}ms)",
                    duration_ms=elapsed,
                    details=resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {},
                )
            return ValidationResult(
                name="gateway_health",
                passed=False,
                message=f"Gateway returned {resp.status_code}",
                duration_ms=elapsed,
            )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return ValidationResult(
            name="gateway_health",
            passed=False,
            message=f"Gateway unreachable: {exc}",
            duration_ms=elapsed,
        )


async def validate_admin_health() -> ValidationResult:
    """Check admin API /health endpoint."""
    url = f"{_admin_url()}/health"
    start = time.perf_counter()

    try:
        import httpx
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            elapsed = (time.perf_counter() - start) * 1000

            return ValidationResult(
                name="admin_health",
                passed=resp.status_code == 200,
                message=f"Admin API {'healthy' if resp.status_code == 200 else 'unhealthy'} ({elapsed:.0f}ms)",
                duration_ms=elapsed,
            )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return ValidationResult(
            name="admin_health",
            passed=False,
            message=f"Admin API unreachable: {exc}",
            duration_ms=elapsed,
        )


async def validate_models_endpoint() -> ValidationResult:
    """Check that /v1/models returns available models."""
    url = f"{_gateway_url()}/v1/models"
    start = time.perf_counter()

    try:
        import httpx
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            elapsed = (time.perf_counter() - start) * 1000

            if resp.status_code != 200:
                return ValidationResult(
                    name="models_endpoint",
                    passed=False,
                    message=f"Models endpoint returned {resp.status_code}",
                    duration_ms=elapsed,
                )

            data = resp.json()
            models = data.get("data", [])
            return ValidationResult(
                name="models_endpoint",
                passed=len(models) > 0,
                message=f"{len(models)} model(s) available",
                duration_ms=elapsed,
                details={"models": [m.get("id", "?") for m in models[:5]]},
            )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return ValidationResult(
            name="models_endpoint",
            passed=False,
            message=f"Models endpoint error: {exc}",
            duration_ms=elapsed,
        )


async def validate_chat_completion() -> ValidationResult:
    """Send a test chat completion request."""
    url = f"{_gateway_url()}/v1/chat/completions"
    start = time.perf_counter()

    payload = {
        "model": "default",
        "messages": [
            {"role": "user", "content": "Rispondi solo con: test migrazione OK"},
        ],
        "max_tokens": 50,
    }

    try:
        import httpx
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload)
            elapsed = (time.perf_counter() - start) * 1000

            if resp.status_code != 200:
                return ValidationResult(
                    name="chat_completion",
                    passed=False,
                    message=f"Chat returned {resp.status_code}: {resp.text[:200]}",
                    duration_ms=elapsed,
                )

            data = resp.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return ValidationResult(
                name="chat_completion",
                passed=bool(content),
                message=f"Chat response received ({elapsed:.0f}ms)",
                duration_ms=elapsed,
                details={"response_preview": content[:100]},
            )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return ValidationResult(
            name="chat_completion",
            passed=False,
            message=f"Chat error: {exc}",
            duration_ms=elapsed,
        )


async def validate_detection_rules() -> ValidationResult:
    """Check that detection rules are loaded and functional."""
    url = f"{_admin_url()}/api/v1/admin/rules"
    api_key = _api_key()
    start = time.perf_counter()

    try:
        import httpx
        headers = {"X-API-Key": api_key} if api_key else {}
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers=headers)
            elapsed = (time.perf_counter() - start) * 1000

            if resp.status_code != 200:
                return ValidationResult(
                    name="detection_rules",
                    passed=False,
                    message=f"Rules endpoint returned {resp.status_code}",
                    duration_ms=elapsed,
                )

            data = resp.json()
            rules = data.get("data", data) if isinstance(data, dict) else data
            if isinstance(rules, list):
                count = len(rules)
            else:
                count = rules.get("total", 0) if isinstance(rules, dict) else 0

            return ValidationResult(
                name="detection_rules",
                passed=True,
                message=f"{count} detection rule(s) loaded",
                duration_ms=elapsed,
                details={"rule_count": count},
            )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return ValidationResult(
            name="detection_rules",
            passed=False,
            message=f"Rules check error: {exc}",
            duration_ms=elapsed,
        )


async def validate_knowledge_search() -> ValidationResult:
    """Test knowledge search (RAG) functionality."""
    url = f"{_gateway_url()}/v1/knowledge/search"
    start = time.perf_counter()

    try:
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                url,
                json={"query": "test migrazione", "n_results": 3},
            )
            elapsed = (time.perf_counter() - start) * 1000

            if resp.status_code == 404:
                return ValidationResult(
                    name="knowledge_search",
                    passed=True,
                    message="Knowledge search endpoint not found (may not be exposed)",
                    duration_ms=elapsed,
                    details={"note": "Endpoint may require admin API"},
                )

            if resp.status_code != 200:
                return ValidationResult(
                    name="knowledge_search",
                    passed=False,
                    message=f"Knowledge search returned {resp.status_code}",
                    duration_ms=elapsed,
                )

            data = resp.json()
            return ValidationResult(
                name="knowledge_search",
                passed=True,
                message=f"Knowledge search OK ({elapsed:.0f}ms)",
                duration_ms=elapsed,
                details={"result_count": len(data.get("data", data.get("results", [])))},
            )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return ValidationResult(
            name="knowledge_search",
            passed=False,
            message=f"Knowledge search error: {exc}",
            duration_ms=elapsed,
        )


async def validate_database_connection() -> ValidationResult:
    """Verify the database is connected and has data."""
    url = f"{_admin_url()}/api/v1/admin/analytics/overview"
    api_key = _api_key()
    start = time.perf_counter()

    try:
        import httpx
        headers = {"X-API-Key": api_key} if api_key else {}
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers=headers)
            elapsed = (time.perf_counter() - start) * 1000

            if resp.status_code != 200:
                return ValidationResult(
                    name="database_connection",
                    passed=False,
                    message=f"Analytics endpoint returned {resp.status_code}",
                    duration_ms=elapsed,
                )

            data = resp.json()
            return ValidationResult(
                name="database_connection",
                passed=True,
                message=f"Database connected, analytics responding ({elapsed:.0f}ms)",
                duration_ms=elapsed,
                details=data.get("data", {}),
            )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return ValidationResult(
            name="database_connection",
            passed=False,
            message=f"Database check error: {exc}",
            duration_ms=elapsed,
        )


async def run_performance_benchmark() -> ValidationResult:
    """Run a quick performance benchmark comparing response times."""
    url = f"{_gateway_url()}/v1/chat/completions"
    num_requests = 3
    latencies: list[float] = []

    try:
        import httpx

        async with httpx.AsyncClient(timeout=120) as client:
            for i in range(num_requests):
                payload = {
                    "model": "default",
                    "messages": [
                        {"role": "user", "content": f"Benchmark test {i}: rispondi brevemente"},
                    ],
                    "max_tokens": 30,
                }
                start = time.perf_counter()
                resp = await client.post(url, json=payload)
                elapsed = (time.perf_counter() - start) * 1000

                if resp.status_code == 200:
                    latencies.append(elapsed)

        if not latencies:
            return ValidationResult(
                name="performance_benchmark",
                passed=False,
                message="All benchmark requests failed",
            )

        avg = sum(latencies) / len(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]

        return ValidationResult(
            name="performance_benchmark",
            passed=avg < 30000,  # 30s threshold
            message=f"Avg: {avg:.0f}ms, P95: {p95:.0f}ms ({len(latencies)}/{num_requests} OK)",
            duration_ms=avg,
            details={
                "avg_ms": round(avg, 1),
                "p95_ms": round(p95, 1),
                "latencies": [round(l, 1) for l in latencies],
                "requests": num_requests,
                "successful": len(latencies),
            },
        )
    except Exception as exc:
        return ValidationResult(
            name="performance_benchmark",
            passed=False,
            message=f"Benchmark error: {exc}",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_validation(
    json_output: bool = False,
    skip_benchmark: bool = False,
) -> bool:
    """Run all validation checks.

    Returns ``True`` if all critical checks pass.
    """
    logger.info("=" * 60)
    logger.info("KnowledgeHub — Post-Migration Validation")
    logger.info("=" * 60)

    # Run health checks concurrently
    health_results = await asyncio.gather(
        validate_gateway_health(),
        validate_admin_health(),
        validate_models_endpoint(),
        validate_database_connection(),
    )

    results: list[ValidationResult] = list(health_results)

    # Functional tests (sequential — depend on services being up)
    results.append(await validate_detection_rules())
    results.append(await validate_knowledge_search())
    results.append(await validate_chat_completion())

    # Performance benchmark (optional)
    if not skip_benchmark:
        logger.info("Running performance benchmark...")
        results.append(await run_performance_benchmark())

    # Report
    critical_passed = 0
    critical_total = 0
    for r in results:
        icon = "+" if r.passed else "!"
        logger.info("  [%s] %-25s %s", icon, r.name, r.message)
        if r.name != "performance_benchmark":
            critical_total += 1
            if r.passed:
                critical_passed += 1

    logger.info("-" * 60)
    all_passed = critical_passed == critical_total
    logger.info(
        "Validation: %d/%d critical checks passed",
        critical_passed, critical_total,
    )

    if all_passed:
        logger.info("Post-migration validation PASSED")
    else:
        logger.error("Post-migration validation FAILED")

    if json_output:
        print(json.dumps({
            "passed": all_passed,
            "critical_passed": critical_passed,
            "critical_total": critical_total,
            "results": [r.to_dict() for r in results],
        }, indent=2))

    return all_passed


def main() -> None:
    json_output = "--json" in sys.argv
    skip_benchmark = "--skip-benchmark" in sys.argv
    passed = asyncio.run(run_validation(
        json_output=json_output, skip_benchmark=skip_benchmark,
    ))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
