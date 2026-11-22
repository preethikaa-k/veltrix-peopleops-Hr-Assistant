"""Thread-safe operational metrics collector for real-time monitoring and observability.

Tracks request throughput, latency percentiles, route breakdown, tool invocations,
guardrail interventions, and LLM vs fallback execution.
"""
from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MetricsState:
    total_requests: int = 0
    status_codes: Counter = field(default_factory=Counter)
    routes: Counter = field(default_factory=Counter)
    latencies_ms: list[float] = field(default_factory=list)
    guardrail_interventions: Counter = field(default_factory=Counter)
    tools_executed: Counter = field(default_factory=Counter)
    llm_generations: int = 0
    llm_fallbacks: int = 0
    start_time: float = field(default_factory=time.time)


class MetricsCollector:
    def __init__(self, max_latency_samples: int = 1000):
        self._lock = threading.Lock()
        self._state = MetricsState()
        self._max_samples = max_latency_samples

    def record_request(
        self,
        status_code: int,
        latency_ms: float,
        route: Optional[str] = None,
        tool_name: Optional[str] = None,
        guardrail_reason: Optional[str] = None,
        is_llm_fallback: bool = False,
    ) -> None:
        """Record operational telemetry for an individual HTTP / chat request."""
        with self._lock:
            self._state.total_requests += 1
            self._state.status_codes[status_code] += 1
            if route:
                self._state.routes[route] += 1
            if tool_name:
                self._state.tools_executed[tool_name] += 1
            if guardrail_reason:
                self._state.guardrail_interventions[guardrail_reason] += 1
            if is_llm_fallback:
                self._state.llm_fallbacks += 1
            else:
                self._state.llm_generations += 1

            self._state.latencies_ms.append(latency_ms)
            if len(self._state.latencies_ms) > self._max_samples:
                self._state.latencies_ms = self._state.latencies_ms[-self._max_samples:]

    def get_summary(self) -> dict[str, Any]:
        """Compute aggregated statistics, percentiles, and rates."""
        with self._lock:
            total = self._state.total_requests
            uptime_s = round(time.time() - self._state.start_time, 1)
            lats = sorted(self._state.latencies_ms)

            def percentile(p: float) -> float:
                if not lats:
                    return 0.0
                idx = int(len(lats) * p)
                idx = min(idx, len(lats) - 1)
                return round(lats[idx], 1)

            avg_lat = round(sum(lats) / len(lats), 1) if lats else 0.0

            return {
                "uptime_seconds": uptime_s,
                "requests": {
                    "total": total,
                    "rps": round(total / max(1.0, uptime_s), 2),
                    "status_codes": dict(self._state.status_codes),
                },
                "latency_ms": {
                    "avg": avg_lat,
                    "p50": percentile(0.50),
                    "p90": percentile(0.90),
                    "p95": percentile(0.95),
                    "p99": percentile(0.99),
                    "min": round(lats[0], 1) if lats else 0.0,
                    "max": round(lats[-1], 1) if lats else 0.0,
                },
                "routes": dict(self._state.routes),
                "tools_executed": dict(self._state.tools_executed),
                "guardrail_interventions": dict(self._state.guardrail_interventions),
                "llm_stats": {
                    "live_generations": self._state.llm_generations,
                    "fallbacks_used": self._state.llm_fallbacks,
                },
            }

    def to_prometheus(self) -> str:
        """Export metrics in standard Prometheus text exposition format."""
        s = self.get_summary()
        lines = [
            "# HELP veltrix_requests_total Total number of HTTP requests handled.",
            "# TYPE veltrix_requests_total counter",
            f"veltrix_requests_total {s['requests']['total']}",
            "",
            "# HELP veltrix_latency_seconds_summary Request latency in milliseconds.",
            "# TYPE veltrix_latency_seconds_summary gauge",
            f"veltrix_latency_ms_avg {s['latency_ms']['avg']}",
            f"veltrix_latency_ms_p50 {s['latency_ms']['p50']}",
            f"veltrix_latency_ms_p95 {s['latency_ms']['p95']}",
            f"veltrix_latency_ms_p99 {s['latency_ms']['p99']}",
            "",
            "# HELP veltrix_routes_total Number of queries handled by route.",
            "# TYPE veltrix_routes_total counter",
        ]
        for route, cnt in s["routes"].items():
            lines.append(f'veltrix_routes_total{{route="{route}"}} {cnt}')
        return "\n".join(lines) + "\n"


# Global singleton instance
collector = MetricsCollector()
