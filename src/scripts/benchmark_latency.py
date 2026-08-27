from __future__ import annotations

import argparse
import json
import os
import platform
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class PreparedRequest:
    name: str
    body: bytes
    content_type: str


def percentile(values: list[float], value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * value / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 2)


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError("--images file must be JPEG or PNG")
        return [path]
    if not path.is_dir():
        raise ValueError(f"Image path does not exist: {path}")
    images = sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError("--images directory does not contain JPEG or PNG files")
    return images


def prepare_request(image_path: Path, client_id: str, camera_id: str) -> PreparedRequest:
    boundary = f"----faceattend-{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in {
        "client_id": client_id,
        "camera_id": camera_id,
        "direction": "auto",
        "dry_run": "true",
    }.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(value.encode())
        body.extend(b"\r\n")
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'.encode()
    )
    body.extend(b"Content-Type: image/jpeg\r\n\r\n")
    body.extend(image_path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return PreparedRequest(
        name=image_path.name,
        body=bytes(body),
        content_type=f"multipart/form-data; boundary={boundary}",
    )


def post(endpoint: str, prepared: PreparedRequest, timeout_seconds: float) -> tuple[int, dict]:
    request = Request(
        endpoint,
        data=prepared.body,
        headers={"Content-Type": prepared.content_type, "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def run_benchmark(
    endpoint: str,
    prepared_requests: list[PreparedRequest],
    warmup: int,
    iterations: int,
    timeout_seconds: float,
) -> dict:
    for index in range(warmup):
        post(endpoint, prepared_requests[index % len(prepared_requests)], timeout_seconds)
    latencies: list[float] = []
    outcomes: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    for index in range(iterations):
        prepared = prepared_requests[index % len(prepared_requests)]
        started_at = time.perf_counter()
        try:
            status_code, response = post(endpoint, prepared, timeout_seconds)
        except (TimeoutError, URLError, OSError) as exc:
            failures[type(exc).__name__] += 1
            continue
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        if not 200 <= status_code < 300:
            failures[f"HTTP_{status_code}"] += 1
            continue
        latencies.append(elapsed_ms)
        outcomes[str(response.get("status", "unknown"))] += 1
    return {
        "mode": "sequential_http_dry_run",
        "completed_requests": len(latencies),
        "requested_iterations": iterations,
        "outcomes": dict(outcomes),
        "failures": dict(failures),
        "latency_ms": {
            "min": round(min(latencies), 2) if latencies else 0.0,
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--warmup", default=20, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--timeout-seconds", default=15.0, type=float)
    parser.add_argument("--camera-id", default="benchmark-camera")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.timeout_seconds <= 0:
        raise SystemExit("warmup must be non-negative; iterations and timeout must be positive")
    image_paths = collect_images(args.images)
    client_id = f"benchmark-{uuid.uuid4()}"
    prepared_requests = [prepare_request(path, client_id, args.camera_id) for path in image_paths]
    result = run_benchmark(
        f"{args.base_url.rstrip('/')}/api/recognition/frame",
        prepared_requests,
        args.warmup,
        args.iterations,
        args.timeout_seconds,
    )
    result.update(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base_url": args.base_url,
            "warmup_requests": args.warmup,
            "image_count": len(image_paths),
            "image_names": [path.name for path in image_paths],
            "environment": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
                "python": platform.python_version(),
            },
        }
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    if result["completed_requests"] == 0:
        raise SystemExit("No successful benchmark requests")


if __name__ == "__main__":
    main()
