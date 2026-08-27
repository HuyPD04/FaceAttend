from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_pairs(path: Path) -> tuple[list[float], list[float]]:
    genuine: list[float] = []
    impostor: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            score = float(row["score"])
            if row["same"].strip().lower() in {"1", "true", "yes"}:
                genuine.append(score)
            else:
                impostor.append(score)
    if not genuine or not impostor:
        raise ValueError("CSV needs at least one genuine and one impostor pair")
    return genuine, impostor


def calibrate(genuine: list[float], impostor: list[float], target_far: float) -> dict:
    thresholds = sorted(set(genuine + impostor + [1.000001]))
    valid: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        far = sum(score >= threshold for score in impostor) / len(impostor)
        frr = sum(score < threshold for score in genuine) / len(genuine)
        if far <= target_far:
            valid.append((threshold, far, frr))
    if not valid:
        raise ValueError("No threshold satisfies the requested FAR")
    threshold, far, frr = min(valid, key=lambda item: item[2])
    return {
        "match_threshold": round(threshold, 6),
        "measured_far": round(far, 6),
        "measured_frr": round(frr, 6),
        "genuine_pairs": len(genuine),
        "impostor_pairs": len(impostor),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--target-far", default=0.001, type=float)
    args = parser.parse_args()
    if not 0.0 <= args.target_far < 1.0:
        raise SystemExit("--target-far must be in [0, 1)")
    genuine, impostor = load_pairs(args.pairs)
    print(json.dumps(calibrate(genuine, impostor, args.target_far), indent=2))


if __name__ == "__main__":
    main()
