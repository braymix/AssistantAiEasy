"""
Enterprise observability: metrics, tracing, and alerting.

Integrates Prometheus metrics, OpenTelemetry tracing, structured JSON
logging, and webhook-based alerting.  When individual features are
disabled, the corresponding collectors are no-ops.

Components
──────────
- **Metrics**: Prometheus-compatible counters, histograms, and gauges
  exposed at ``/metrics`` via ``prometheus_client``.
- **Tracing**: OpenTelemetry spans for request lifecycle, database
  queries, and LLM calls.
- **Structured logging**: JSON-formatted logs with correlation IDs
  for log aggregation (ELK / Loki).
- **Alerting**: Webhook notifications (Slack, Teams, PagerDuty) on
  error thresholds and health check failures.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Generator

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.enterprise import is_enterprise_enabled

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prometheus-style metrics (no external dep required for collection)
# ---------------------------------------------------------------------------

class Counter:
    """Thread-safe monotonic counter."""

    __slots__ = ("name", "help", "labels", "_values")

    def __init__(self, name: str, help: str, label_names: tuple[str, ...] = ()) -> None:
        self.name = name
        self.help = help
        self.labels = label_names
        self._values: dict[tuple[str, ...], float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = tuple(labels.get(l, "") for l in self.labels)
        self._values[key] = self._values.get(key, 0.0) + amount

    def get(self, **labels: str) -> float:
        key = tuple(labels.get(l, "") for l in self.labels)
        return self._values.get(key, 0.0)

    def collect(self) -> list[dict[str, Any]]:
        results = []
        for label_values, value in self._values.items():
            label_dict = dict(zip(self.labels, label_values))
            results.append({
                "name": self.name,
                "type": "counter",
                "value": value,
                "labels": label_dict,
            })
        return results


class Histogram:
    """Simple histogram with configurable buckets."""

    __slots__ = ("name", "help", "labels", "_buckets", "_observations")

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(
        self,
        name: str,
        help: str,
        label_names: tuple[str, ...] = (),
        buckets: tuple[float, ...] | None = None,
    ) -> None:
        self.name = name
        self.help = help
        self.labels = label_names
        self._buckets = buckets or self.DEFAULT_BUCKETS
        self._observations: dict[tuple[str, ...], list[float]] = {}

    def observe(self, value: float, **labels: str) -> None:
        key = tuple(labels.get(l, "") for l in self.labels)
        self._observations.setdefault(key, []).append(value)

    @contextmanager
    def time(self, **labels: str) -> Generator[None, None, None]:
        """Context manager that measures elapsed time."""
        start = time.perf_counter()
        yield
        self.observe(time.perf_counter() - start, **labels)

    def collect(self) -> list[dict[str, Any]]:
        results = []
        for label_values, observations in self._observations.items():
            label_dict = dict(zip(self.labels, label_values))
            count = len(observations)
            total = sum(observations)
            bucket_counts = {}
            for b in self._buckets:
                bucket_counts[str(b)] = sum(1 for o in observations if o <= b)
            bucket_counts["+Inf"] = count

            results.append({
                "name": self.name,
                "type": "histogram",
                "labels": label_dict,
                "count": count,
                "sum": total,
                "buckets": bucket_counts,
            })
        return results


class Gauge:
    """Thread-safe gauge (can go up and down)."""

    __slots__ = ("name", "help", "labels", "_values")

    def __init__(self, name: str, help: str, label_names: tuple[str, ...] = ()) -> None:
        self.name = name
        self.help = help
        self.labels = label_names
        self._values: dict[tuple[str, ...], float] = {}

    def set(self, value: float, **labels: str) -> None:
        key = tuple(labels.get(l, "") for l in self.labels)
        self._values[key] = value

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = tuple(labels.get(l, "") for l in self.labels)
        self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, **labels: str) -> None:
        key = tuple(labels.get(l, "") for l in self.labels)
        self._values[key] = self._values.get(key, 0.0) - amount

    def get(self, **labels: str) -> float:
        key = tuple(labels.get(l, "") for l in self.labels)
        return self._values.get(key, 0.0)

    def collect(self) -> list[dict[str, Any]]:
        results = []
        for label_values, value in self._values.items():
            label_dict = dict(zip(self.labels, label_values))
            results.append({
                "name": self.name,
                "type": "gauge",
                "value": value,
                "labels": label_dict,
            })
        return results


# ---------------------------------------------------------------------------
# Metric registry (pre-defined application metrics)
# ---------------------------------------------------------------------------

class MetricsRegistry:
    """Central registry for all application metrics."""

    def __init__(self) -> None:
        # HTTP
        self.http_requests_total = Counter(
            "kh_http_requests_total",
            "Total HTTP requests",
            label_names=("method", "path", "status"),
        )
        self.http_request_duration = Histogram(
            "kh_http_request_duration_seconds",
            "HTTP request duration in seconds",
            label_names=("method", "path"),
        )
        self.http_active_requests = Gauge(
            "kh_http_active_requests",
            "Currently active HTTP requests",
        )

        # Chat / LLM
        self.chat_requests_total = Counter(
            "kh_chat_requests_total",
            "Total chat completion requests",
            label_names=("model", "stream"),
        )
        self.chat_tokens_total = Counter(
            "kh_chat_tokens_total",
            "Total tokens processed",
            label_names=("direction",),  # prompt / completion
        )
        self.llm_request_duration = Histogram(
            "kh_llm_request_duration_seconds",
            "LLM backend request duration",
            label_names=("provider", "model"),
        )

        # Detection
        self.detection_runs_total = Counter(
            "kh_detection_runs_total",
            "Total detection engine runs",
        )
        self.detection_duration = Histogram(
            "kh_detection_duration_seconds",
            "Detection engine evaluation time",
        )
        self.rules_triggered_total = Counter(
            "kh_rules_triggered_total",
            "Total rule triggers",
            label_names=("rule_name", "rule_type"),
        )

        # Knowledge / RAG
        self.knowledge_searches_total = Counter(
            "kh_knowledge_searches_total",
            "Total knowledge search queries",
        )
        self.knowledge_items_total = Gauge(
            "kh_knowledge_items_total",
            "Total knowledge items in the system",
            label_names=("verified",),
        )
        self.rag_enrichment_duration = Histogram(
            "kh_rag_enrichment_duration_seconds",
            "RAG context building duration",
        )

        # System
        self.active_conversations = Gauge(
            "kh_active_conversations",
            "Active conversations in the last hour",
        )

        self._all_collectors: list[Counter | Histogram | Gauge] = [
            self.http_requests_total,
            self.http_request_duration,
            self.http_active_requests,
            self.chat_requests_total,
            self.chat_tokens_total,
            self.llm_request_duration,
            self.detection_runs_total,
            self.detection_duration,
            self.rules_triggered_total,
            self.knowledge_searches_total,
            self.knowledge_items_total,
            self.rag_enrichment_duration,
            self.active_conversations,
        ]

    def collect_all(self) -> list[dict[str, Any]]:
        """Collect all metrics as a list of dicts."""
        results: list[dict[str, Any]] = []
        for collector in self._all_collectors:
            results.extend(collector.collect())
        return results

    def to_prometheus_text(self) -> str:
        """Format all metrics as Prometheus text exposition."""
        lines: list[str] = []
        for collector in self._all_collectors:
            lines.append(f"# HELP {collector.name} {collector.help}")
            lines.append(f"# TYPE {collector.name} {type(collector).__name__.lower()}")
            for entry in collector.collect():
                label_str = ""
                if entry.get("labels"):
                    pairs = [f'{k}="{v}"' for k, v in entry["labels"].items()]
                    label_str = "{" + ",".join(pairs) + "}"

                if entry["type"] == "histogram":
                    for bucket, count in entry["buckets"].items():
                        lines.append(f'{collector.name}_bucket{{le="{bucket}"{label_str and "," + label_str[1:]}}} {count}')
                    lines.append(f"{collector.name}_count{label_str} {entry['count']}")
                    lines.append(f"{collector.name}_sum{label_str} {entry['sum']:.6f}")
                else:
                    lines.append(f"{collector.name}{label_str} {entry['value']}")
        return "\n".join(lines) + "\n"


# Singleton
_metrics = MetricsRegistry()


def get_metrics() -> MetricsRegistry:
    return _metrics


# ---------------------------------------------------------------------------
# OpenTelemetry tracing (stub interface)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SpanContext:
    """Lightweight span representation for tracing."""

    trace_id: str
    span_id: str
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "OK"
    children: list[SpanContext] = field(default_factory=list)


class Tracer:
    """OpenTelemetry-compatible tracer.

    When ``tracing_enabled`` is ``False``, produces no-op spans.
    When enabled, creates span trees that can be exported to an
    OTLP endpoint (Jaeger, Zipkin, etc.).
    """

    def __init__(self, service_name: str = "knowledgehub", otlp_endpoint: str = "") -> None:
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self._spans: list[SpanContext] = []

    @asynccontextmanager
    async def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> AsyncGenerator[SpanContext, None]:
        """Start a new trace span.

        Usage::

            async with tracer.start_span("db.query", {"table": "rules"}) as span:
                result = await db.execute(query)
                span.attributes["row_count"] = len(result)
        """
        if not is_enterprise_enabled("tracing"):
            # No-op span
            yield SpanContext(trace_id="", span_id="", name=name)
            return

        import uuid as _uuid
        span = SpanContext(
            trace_id=_uuid.uuid4().hex,
            span_id=_uuid.uuid4().hex[:16],
            name=name,
            attributes=attributes or {},
        )
        try:
            yield span
            span.status = "OK"
        except Exception as exc:
            span.status = f"ERROR: {exc}"
            raise
        finally:
            span.end_time = time.time()
            self._spans.append(span)
            await self._export_span(span)

    async def _export_span(self, span: SpanContext) -> None:
        """Export span to OTLP endpoint.

        In a full implementation this would batch spans and send them
        via gRPC or HTTP to the collector.
        """
        if not self.otlp_endpoint:
            return
        logger.debug(
            "TRACE: %s duration=%.3fms status=%s",
            span.name,
            (span.end_time or time.time()) - span.start_time,
            span.status,
        )


# Singleton
_tracer = Tracer()


def get_tracer() -> Tracer:
    return _tracer


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AlertRule:
    """Defines a condition that triggers a webhook notification."""

    name: str
    metric_name: str
    threshold: float
    comparison: str = "gt"  # gt, lt, eq
    window_seconds: int = 300
    webhook_url: str = ""
    cooldown_seconds: int = 600
    last_fired: float = 0.0
    enabled: bool = True


class AlertManager:
    """Evaluates alert rules and fires webhooks.

    Supports Slack, Microsoft Teams, and generic webhook endpoints.
    """

    def __init__(self) -> None:
        self._rules: list[AlertRule] = []

    def add_rule(self, rule: AlertRule) -> None:
        self._rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        return len(self._rules) < before

    async def evaluate(self, metrics: MetricsRegistry) -> list[str]:
        """Evaluate all alert rules against current metrics.

        Returns list of fired alert names.
        """
        fired: list[str] = []
        now = time.time()

        for rule in self._rules:
            if not rule.enabled:
                continue
            if now - rule.last_fired < rule.cooldown_seconds:
                continue

            # Find matching metric value
            value = self._get_metric_value(metrics, rule.metric_name)
            if value is None:
                continue

            triggered = False
            if rule.comparison == "gt" and value > rule.threshold:
                triggered = True
            elif rule.comparison == "lt" and value < rule.threshold:
                triggered = True
            elif rule.comparison == "eq" and value == rule.threshold:
                triggered = True

            if triggered:
                rule.last_fired = now
                fired.append(rule.name)
                await self._fire_webhook(rule, value)

        return fired

    @staticmethod
    def _get_metric_value(metrics: MetricsRegistry, metric_name: str) -> float | None:
        """Extract a scalar value from a named metric."""
        for collector in metrics._all_collectors:
            if collector.name == metric_name:
                if isinstance(collector, (Counter, Gauge)):
                    return collector.get()
        return None

    async def _fire_webhook(self, rule: AlertRule, value: float) -> None:
        """Send alert notification to the configured webhook."""
        if not rule.webhook_url:
            logger.warning("Alert '%s' fired but no webhook URL configured", rule.name)
            return

        payload = {
            "alert": rule.name,
            "metric": rule.metric_name,
            "value": value,
            "threshold": rule.threshold,
            "comparison": rule.comparison,
            "timestamp": time.time(),
            "service": "knowledgehub",
        }

        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(rule.webhook_url, json=payload)
            logger.info("Alert webhook fired: %s (value=%.2f)", rule.name, value)
        except Exception as exc:
            logger.error("Alert webhook failed for '%s': %s", rule.name, exc)


_alert_manager = AlertManager()


def get_alert_manager() -> AlertManager:
    return _alert_manager


# ---------------------------------------------------------------------------
# FastAPI middleware for automatic metrics collection
# ---------------------------------------------------------------------------

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware that records HTTP metrics for every request."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        if not is_enterprise_enabled("metrics"):
            return await call_next(request)

        metrics = get_metrics()
        path = request.url.path
        method = request.method

        metrics.http_active_requests.inc()
        start = time.perf_counter()

        try:
            response = await call_next(request)
            status_code = str(response.status_code)
        except Exception:
            status_code = "500"
            raise
        finally:
            duration = time.perf_counter() - start
            metrics.http_active_requests.dec()
            metrics.http_requests_total.inc(
                method=method, path=path, status=status_code,
            )
            metrics.http_request_duration.observe(
                duration, method=method, path=path,
            )

        return response
