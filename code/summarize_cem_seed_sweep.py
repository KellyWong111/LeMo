from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


SUCCESS_RE = re.compile(r"success_rate':\s*([0-9.]+)")


def load_success(path: Path) -> float:
    text = path.read_text()
    match = SUCCESS_RE.search(text)
    if match is None:
        raise ValueError(f"Could not parse success_rate from {path}")
    return float(match.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    groups: dict[str, list[dict[str, float | str]]] = {}
    for path in sorted(root.glob("*.txt")):
        stem = path.stem
        if "_solverseed" not in stem:
            continue
        tag, seed_str = stem.split("_solverseed", 1)
        seed = int(seed_str)
        groups.setdefault(tag, []).append(
            {
                "seed": seed,
                "success_rate": load_success(path),
                "file": path.name,
            }
        )

    summary = {}
    for tag, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda item: int(item["seed"]))
        vals = np.asarray([float(item["success_rate"]) for item in rows], dtype=np.float32)
        summary[tag] = {
            "n": len(rows),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "values": rows,
        }

    payload = {"root": str(root), "summary": summary}
    rendered = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
