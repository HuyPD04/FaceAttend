from __future__ import annotations

import threading
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone


class MetricsRegistry:
    def __init__(self):
        self._started_at = datetime.now(timezone.utc)
        self._counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._latency_totals: Counter[str] = Counter()
        self._latency_counts: Counter[str] = Counter()
        self._latencies: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=10_000))
        self._lock = threading.Lock()

    def record(self, operation: str, outcome: str, latency_ms: float) -> None:
        with self._lock:
            self._counts[operation][outcome] += 1
            self._latency_totals[operation] += latency_ms
            self._latency_counts[operation] += 1
            self._latencies[operation].append(latency_ms)

    def snapshot(self) -> dict:
        with self._lock:
            operations = {}
            for operation, counts in self._counts.items():
                total = self._latency_counts[operation]
                latencies = sorted(self._latencies[operation])
                operations[operation] = {
                    "outcomes": dict(counts),
                    "requests": total,
                    "average_latency_ms": round(self._latency_totals[operation] / total, 2)
                    if total
                    else 0.0,
                    "p50_latency_ms": self._percentile(latencies, 50),
                    "p95_latency_ms": self._percentile(latencies, 95),
                    "p99_latency_ms": self._percentile(latencies, 99),
                    "latency_samples": len(latencies),
                }
            return {"started_at": self._started_at, "operations": operations}

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        position = (len(values) - 1) * percentile / 100
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        fraction = position - lower
        value = values[lower] + (values[upper] - values[lower]) * fraction
        return round(value, 2)
