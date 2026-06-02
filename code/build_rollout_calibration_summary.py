from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


STABLE = Path(
    "/data1/jingyixi/.stable_worldmodel/"
    "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07"
)
OUT = Path("/data1/jingyixi/wm_runs/horizon_sensitivity_multiseed")


def parse_success(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    m = re.search(r"success_rate':\s*([0-9.]+)", text)
    return float(m.group(1)) if m else None


def h5_path(ep: int, seed: int) -> Path:
    if ep == 4 and seed == 42:
        return STABLE / "gate07_ep4_s42_e20_s300_n30_k30.txt"
    if ep == 4:
        p = STABLE / f"gate07_ep4_seed{seed}_s300_n30_k30.txt"
        if not p.exists():
            p = STABLE / f"gate07_ep4_s{seed}_e20_s300_n30_k30.txt"
        return p
    if ep == 9 and seed == 42:
        return STABLE / "gate07_ep9_seed42_s300_n30_k30.txt"
    if ep == 16 and seed == 42:
        return STABLE / "gate07_ep16_seed42_s300_n30_k30.txt"
    return STABLE / f"gate07_ep{ep}_seed{seed}_h5_rh5_b5_s300_n30_k30.txt"


def h3_path(ep: int, seed: int) -> Path:
    if seed == 42:
        return STABLE / f"gate07_ep{ep}_h3_rh3_b5_s300_n30_k30.txt"
    return STABLE / f"gate07_ep{ep}_seed{seed}_h3_rh3_b5_s300_n30_k30.txt"


def h4_path(ep: int, seed: int) -> Path:
    return STABLE / f"gate07_ep{ep}_seed{seed}_h4_rh4_b5_s300_n30_k30.txt"


def main() -> None:
    rows = []
    for ep in [4, 9, 16]:
        for seed in [42, 43, 44]:
            rows.append(
                {
                    "epoch": ep,
                    "seed": seed,
                    "h3_15step": parse_success(h3_path(ep, seed)),
                    "h4_20step": parse_success(h4_path(ep, seed)),
                    "h5_25step": parse_success(h5_path(ep, seed)),
                }
            )

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["epoch"]].append(row)
    summary = {"rows": rows, "by_epoch": {}, "overall": {}}
    all_rows = [r for r in rows]
    for key, group in list(grouped.items()) + [("overall", all_rows)]:
        stats = {}
        for h in ["h3_15step", "h4_20step", "h5_25step"]:
            vals = [r[h] for r in group if r[h] is not None]
            stats[h] = {
                "mean": sum(vals) / len(vals) if vals else None,
                "n": len(vals),
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
            }
        if key == "overall":
            summary["overall"] = stats
        else:
            summary["by_epoch"][str(key)] = stats

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate07_rollout_calibration_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    headers = ["epoch", "seed", "h3_15step", "h4_20step", "h5_25step"]
    lines = [
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append(
            "|"
            + "|".join(
                str(row[h])
                if not isinstance(row[h], float)
                else f"{row[h]:.1f}"
                for h in headers
            )
            + "|"
        )
    lines.append("")
    for ep in ["4", "9", "16"]:
        stats = summary["by_epoch"][ep]
        lines.append(
            f"ep{ep} means: h3={stats['h3_15step']['mean']:.1f} "
            f"(n={stats['h3_15step']['n']}), h4="
            f"{stats['h4_20step']['mean']:.1f} (n={stats['h4_20step']['n']}), "
            f"h5={stats['h5_25step']['mean']:.1f} (n={stats['h5_25step']['n']})"
        )
    stats = summary["overall"]
    lines.append(
        f"overall means: h3={stats['h3_15step']['mean']:.1f} "
        f"(n={stats['h3_15step']['n']}), h4="
        f"{stats['h4_20step']['mean']:.1f} (n={stats['h4_20step']['n']}), "
        f"h5={stats['h5_25step']['mean']:.1f} (n={stats['h5_25step']['n']})"
    )
    table = "\n".join(lines) + "\n"
    (OUT / "gate07_rollout_calibration_summary.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
