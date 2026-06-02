from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


STABLE = Path(
    "/data1/jingyixi/.stable_worldmodel/"
    "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07"
)
OUTDIR = Path("/data1/jingyixi/wm_runs/expert_clip_cem_bundle")


def parse_success(path: Path) -> float | None:
    text = path.read_text(errors="ignore")
    for pattern in [
        r"success_rate['\"]?\s*[:=]\s*([0-9.]+)",
        r"'success_rate':\s*([0-9.]+)",
        r'"success_rate":\s*([0-9.]+)',
    ]:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rx = re.compile(
        r"expertclip_c(?P<clip>[0-9p]+)_v(?P<var>[0-9p]+)_"
        r"gate07_ep(?P<ep>\d+)_seed(?P<seed>\d+)_h4_s1000_k100_n20"
    )
    rows = []
    for path in STABLE.glob("expertclip_c*_v*_gate07_ep*_seed*_h4_s1000_k100_n20.txt"):
        match = rx.fullmatch(path.stem)
        if not match:
            continue
        data = match.groupdict()
        rows.append(
            {
                "clip": float(data["clip"].replace("p", ".")),
                "var_scale": float(data["var"].replace("p", ".")),
                "epoch": int(data["ep"]),
                "seed": int(data["seed"]),
                "success_rate": parse_success(path),
                "file": str(path),
            }
        )
    rows.sort(key=lambda r: (r["clip"], r["var_scale"], r["epoch"], r["seed"]))
    grouped = defaultdict(list)
    for row in rows:
        if row["success_rate"] is not None:
            grouped[(row["clip"], row["var_scale"], row["epoch"])].append(
                row["success_rate"]
            )
    summary = []
    for (clip, var_scale, epoch), vals in sorted(grouped.items()):
        summary.append(
            {
                "clip": clip,
                "var_scale": var_scale,
                "epoch": epoch,
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
            }
        )
    (OUTDIR / "raw_results.json").write_text(json.dumps(rows, indent=2))
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "|clip_std|var_scale|epoch|mean|min|max|n|",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "|{clip:.1f}|{var_scale:.1f}|{epoch}|{mean:.1f}|{min:.1f}|{max:.1f}|{n}|".format(
                **row
            )
        )
    text = "\n".join(lines) + "\n"
    (OUTDIR / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
