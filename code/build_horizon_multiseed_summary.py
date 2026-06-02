from __future__ import annotations

import json
import re
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


def row_for(ep: int, seed: int) -> dict:
    if ep == 9 and seed == 42:
        h5 = parse_success(STABLE / "gate07_ep9_seed42_s300_n30_k30.txt")
        h3 = parse_success(STABLE / "gate07_ep9_h3_rh3_b5_s300_n30_k30.txt")
    elif ep == 4 and seed == 42:
        h5 = parse_success(STABLE / "gate07_ep4_s42_e20_s300_n30_k30.txt")
        h3 = parse_success(STABLE / "gate07_ep4_h3_rh3_b5_s300_n30_k30.txt")
    elif ep == 4:
        h5 = parse_success(STABLE / f"gate07_ep4_seed{seed}_s300_n30_k30.txt")
        if h5 is None:
            h5 = parse_success(STABLE / f"gate07_ep4_s{seed}_e20_s300_n30_k30.txt")
        h3 = parse_success(
            STABLE / f"gate07_ep4_seed{seed}_h3_rh3_b5_s300_n30_k30.txt"
        )
    else:
        h5 = parse_success(
            STABLE / f"gate07_ep{ep}_seed{seed}_h5_rh5_b5_s300_n30_k30.txt"
        )
        h3 = parse_success(
            STABLE / f"gate07_ep{ep}_seed{seed}_h3_rh3_b5_s300_n30_k30.txt"
        )
    return {
        "epoch": ep,
        "seed": seed,
        "h5_25step": h5,
        "h3_15step": h3,
        "delta_h3_minus_h5": None if h3 is None or h5 is None else h3 - h5,
    }


def summarize(rows: list[dict]) -> dict:
    valid = [
        r
        for r in rows
        if r["h5_25step"] is not None and r["h3_15step"] is not None
    ]
    return {
        "mean_h5": sum(r["h5_25step"] for r in valid) / len(valid),
        "mean_h3": sum(r["h3_15step"] for r in valid) / len(valid),
        "mean_delta": sum(r["delta_h3_minus_h5"] for r in valid) / len(valid),
        "wins_h3": sum(1 for r in valid if r["delta_h3_minus_h5"] > 0),
        "num_seeds": len(valid),
    }


def main() -> None:
    rows = []
    for ep in [4, 9]:
        for seed in [42, 43, 44]:
            rows.append(row_for(ep, seed))
    by_epoch = {str(ep): summarize([r for r in rows if r["epoch"] == ep]) for ep in [4, 9]}
    overall = summarize(rows)
    summary = {"rows": rows, "by_epoch": by_epoch, "overall": overall}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate07_ep9_horizon_multiseed_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    lines = [
        "|epoch|seed|h5_25step|h3_15step|delta_h3_minus_h5|",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"|{r['epoch']}|{r['seed']}|{r['h5_25step']:.1f}|{r['h3_15step']:.1f}|{r['delta_h3_minus_h5']:.1f}|"
        )
    lines.append("")
    for ep, s in by_epoch.items():
        lines.append(
            f"ep{ep}: mean_h5={s['mean_h5']:.1f}, mean_h3={s['mean_h3']:.1f}, "
            f"mean_delta={s['mean_delta']:.1f}, h3_wins={s['wins_h3']}/{s['num_seeds']}"
        )
    lines.append(
        f"overall: mean_h5={overall['mean_h5']:.1f}, mean_h3={overall['mean_h3']:.1f}, "
        f"mean_delta={overall['mean_delta']:.1f}, h3_wins={overall['wins_h3']}/{overall['num_seeds']}"
    )
    table = "\n".join(lines) + "\n"
    (OUT / "gate07_ep9_horizon_multiseed_summary.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
