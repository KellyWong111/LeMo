from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path("/data1/jingyixi/wm_runs")
STABLE = Path(
    "/data1/jingyixi/.stable_worldmodel/"
    "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07"
)
OUT = ROOT / "final_planning_calibration"


def parse_success(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    m = re.search(r"success_rate':\s*([0-9.]+)", text)
    if not m:
        m = re.search(r'"success_rate":\s*([0-9.]+)', text)
    return float(m.group(1)) if m else None


def stock_h5_path(ep: int, seed: int) -> Path:
    if ep == 4 and seed == 42:
        return STABLE / "gate07_ep4_s42_e20_s300_n30_k30.txt"
    if ep == 4:
        p = STABLE / f"gate07_ep4_seed{seed}_s300_n30_k30.txt"
        if not p.exists():
            p = STABLE / f"gate07_ep4_s{seed}_e20_s300_n30_k30.txt"
        return p
    if seed == 42:
        return STABLE / f"gate07_ep{ep}_seed42_s300_n30_k30.txt"
    return STABLE / f"gate07_ep{ep}_seed{seed}_h5_rh5_b5_s300_n30_k30.txt"


def summarize(rows: list[dict], group_fields: list[str]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row.get("success_rate") is None:
            continue
        key = tuple(row[field] for field in group_fields)
        groups[key].append(row["success_rate"])
    out = []
    for key, vals in sorted(groups.items()):
        item = {field: value for field, value in zip(group_fields, key)}
        item.update(
            {
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
            }
        )
        out.append(item)
    return out


def main() -> None:
    overnight_rows = []
    for path in STABLE.glob("gate07_ep*_seed*_h*_rh*_b5_s*_k*_n*.txt"):
        m = re.search(
            r"gate07_ep(\d+)_seed(\d+)_h(\d+)_rh\d+_b5_s(\d+)_k(\d+)_n(\d+)",
            path.stem,
        )
        if not m:
            continue
        ep, seed, h, samples, topk, steps = map(int, m.groups())
        if ep not in [4, 9, 16] or h not in [3, 4]:
            continue
        overnight_rows.append(
            {
                "epoch": ep,
                "seed": seed,
                "horizon": h,
                "samples": samples,
                "topk": topk,
                "n_steps": steps,
                "success_rate": parse_success(path),
                "file": str(path),
            }
        )

    bestcand_rows = []
    for ep in [4, 9, 16]:
        for seed in [42, 43, 44]:
            stock = parse_success(stock_h5_path(ep, seed))
            best = parse_success(
                STABLE / f"bestcand_gate07_ep{ep}_seed{seed}_h5_s300_k30_n30.txt"
            )
            bestcand_rows.append(
                {
                    "epoch": ep,
                    "seed": seed,
                    "stock_h5_mean_cem": stock,
                    "best_candidate_h5": best,
                    "delta": None if stock is None or best is None else best - stock,
                }
            )

    report = {
        "overnight_raw": sorted(
            overnight_rows,
            key=lambda r: (
                r["epoch"],
                r["horizon"],
                r["samples"],
                r["topk"],
                r["n_steps"],
                r["seed"],
            ),
        ),
        "overnight_by_setting": summarize(
            overnight_rows, ["epoch", "horizon", "samples", "topk", "n_steps"]
        ),
        "bestcand_raw": bestcand_rows,
        "bestcand_by_epoch": summarize(
            [
                {
                    "epoch": r["epoch"],
                    "success_rate": r["delta"],
                }
                for r in bestcand_rows
                if r["delta"] is not None
            ],
            ["epoch"],
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))

    lines = ["# Planning Calibration Report", ""]
    lines += ["## Rollout Horizon x CEM Budget", ""]
    lines += ["|epoch|horizon|samples|topk|n_steps|mean|min|max|n|"]
    lines += ["|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report["overnight_by_setting"]:
        lines.append(
            f"|{r['epoch']}|{r['horizon']}|{r['samples']}|{r['topk']}|{r['n_steps']}|"
            f"{r['mean']:.1f}|{r['min']:.1f}|{r['max']:.1f}|{r['n']}|"
        )
    lines += ["", "## Best-Candidate CEM vs Stock Elite-Mean CEM", ""]
    lines += ["|epoch|seed|stock_h5|best_candidate|delta|"]
    lines += ["|---|---:|---:|---:|---:|"]
    for r in bestcand_rows:
        def fmt(x):
            return "" if x is None else f"{x:.1f}"
        lines.append(
            f"|{r['epoch']}|{r['seed']}|{fmt(r['stock_h5_mean_cem'])}|"
            f"{fmt(r['best_candidate_h5'])}|{fmt(r['delta'])}|"
        )
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
