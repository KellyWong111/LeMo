from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


STABLE = Path(
    "/data1/jingyixi/.stable_worldmodel/"
    "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07"
)
OUTDIR = Path("/data1/jingyixi/wm_runs/behavior_cem_bundle")


def parse_success(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    patterns = [
        r"success_rate['\"]?\s*[:=]\s*([0-9.]+)",
        r"'success_rate':\s*([0-9.]+)",
        r'"success_rate":\s*([0-9.]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


def collect() -> list[dict]:
    rows: list[dict] = []
    shrink_re = re.compile(
        r"shrink_a([0-9p]+)_gate07_ep(\d+)_seed(\d+)_h4_s1000_k100_n20"
    )
    var_re = re.compile(
        r"varscale_v([0-9p]+)_gate07_ep(\d+)_seed(\d+)_h4_s1000_k100_n20"
    )
    for path in STABLE.glob("*_gate07_ep*_seed*_h4_s1000_k100_n20.txt"):
        stem = path.stem
        m = shrink_re.fullmatch(stem)
        if m:
            value, ep, seed = m.groups()
            rows.append(
                {
                    "method": "shrink",
                    "value": float(value.replace("p", ".")),
                    "epoch": int(ep),
                    "seed": int(seed),
                    "success_rate": parse_success(path),
                    "file": str(path),
                }
            )
            continue
        m = var_re.fullmatch(stem)
        if m:
            value, ep, seed = m.groups()
            rows.append(
                {
                    "method": "varscale",
                    "value": float(value.replace("p", ".")),
                    "epoch": int(ep),
                    "seed": int(seed),
                    "success_rate": parse_success(path),
                    "file": str(path),
                }
            )
    return sorted(rows, key=lambda r: (r["method"], r["value"], r["epoch"], r["seed"]))


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        if row["success_rate"] is not None:
            grouped[(row["method"], row["value"], row["epoch"])].append(row["success_rate"])
    summary = []
    for (method, value, epoch), vals in sorted(grouped.items()):
        summary.append(
            {
                "method": method,
                "value": value,
                "epoch": epoch,
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
            }
        )
    return summary


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = collect()
    summary = summarize(rows)
    (OUTDIR / "raw_results.json").write_text(json.dumps(rows, indent=2))
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "|method|value|epoch|mean|min|max|n|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "|{method}|{value:.2f}|{epoch}|{mean:.1f}|{min:.1f}|{max:.1f}|{n}|".format(
                **row
            )
        )
    text = "\n".join(lines) + "\n"
    (OUTDIR / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
