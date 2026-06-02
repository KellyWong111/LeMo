from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

WM = Path("/data1/jingyixi/wm_runs")
BSL = WM / "bsl_normalbudget_candidate_pool_s300_steps30_n100" / "proposal_data"
ST = WM / "stateroll_normalbudget_candidate_pool_s300_steps30_n100" / "proposal_data"
OUT_DIR = WM
SEEDS = [42, 43, 44, 45, 46, 47]


def load(seed: int, variant: str):
    root = BSL if variant == "baseline" else ST
    return np.load(root / f"{variant}_seed{seed}.npz", allow_pickle=True)


def pct(x: int, n: int) -> float:
    return 100.0 * x / n if n else 0.0


def summarize_seed(seed: int) -> dict:
    b = load(seed, "baseline")
    s = load(seed, "vf05_mix20")
    assert np.all(b["indices"] == s["indices"])
    b_labels = b["labels"].astype(bool)
    s_labels = s["labels"].astype(bool)
    n = int(b_labels.shape[0])
    b_top1 = b_labels[:, 0]
    s_top1 = s_labels[:, 0]
    b_oracle = b_labels.any(axis=1)
    s_oracle = s_labels.any(axis=1)
    union_oracle = b_oracle | s_oracle

    bsl_only = b_oracle & ~s_oracle
    st_only = s_oracle & ~b_oracle
    both = b_oracle & s_oracle
    neither = ~b_oracle & ~s_oracle

    bsl_fail = ~b_top1
    fix_bsl_only = bsl_fail & b_oracle & ~s_oracle
    fix_st_only = bsl_fail & s_oracle & ~b_oracle
    fix_both = bsl_fail & b_oracle & s_oracle
    fix_any = bsl_fail & union_oracle

    return {
        "seed": seed,
        "episodes": n,
        "bsl_top1_count": int(b_top1.sum()),
        "bsl_top1": pct(int(b_top1.sum()), n),
        "bsl_oracle_count": int(b_oracle.sum()),
        "bsl_oracle": pct(int(b_oracle.sum()), n),
        "stateroll_top1_count": int(s_top1.sum()),
        "stateroll_top1": pct(int(s_top1.sum()), n),
        "stateroll_oracle_count": int(s_oracle.sum()),
        "stateroll_oracle": pct(int(s_oracle.sum()), n),
        "union_oracle_count": int(union_oracle.sum()),
        "union_oracle": pct(int(union_oracle.sum()), n),
        "bsl_only_success_count": int(bsl_only.sum()),
        "bsl_only_success": pct(int(bsl_only.sum()), n),
        "stateroll_only_success_count": int(st_only.sum()),
        "stateroll_only_success": pct(int(st_only.sum()), n),
        "both_success_count": int(both.sum()),
        "both_success": pct(int(both.sum()), n),
        "neither_success_count": int(neither.sum()),
        "neither_success": pct(int(neither.sum()), n),
        "bsl_rank0_failure_count": int(bsl_fail.sum()),
        "bsl_rank0_failure": pct(int(bsl_fail.sum()), n),
        "bsl_rank0_failure_fixable_any_count": int(fix_any.sum()),
        "bsl_rank0_failure_fixable_any": pct(int(fix_any.sum()), int(bsl_fail.sum())),
        "bsl_rank0_failure_fixable_bsl_only_count": int(fix_bsl_only.sum()),
        "bsl_rank0_failure_fixable_bsl_only": pct(int(fix_bsl_only.sum()), int(bsl_fail.sum())),
        "bsl_rank0_failure_fixable_stateroll_only_count": int(fix_st_only.sum()),
        "bsl_rank0_failure_fixable_stateroll_only": pct(int(fix_st_only.sum()), int(bsl_fail.sum())),
        "bsl_rank0_failure_fixable_both_count": int(fix_both.sum()),
        "bsl_rank0_failure_fixable_both": pct(int(fix_both.sum()), int(bsl_fail.sum())),
    }


def aggregate(rows: list[dict]) -> dict:
    keys_count = [k for k in rows[0] if k.endswith("_count") or k == "episodes"]
    total = {"seed": "ALL"}
    for k in keys_count:
        total[k] = int(sum(r[k] for r in rows))
    n = total["episodes"]
    fail = total["bsl_rank0_failure_count"]
    total.update(
        {
            "bsl_top1": pct(total["bsl_top1_count"], n),
            "bsl_oracle": pct(total["bsl_oracle_count"], n),
            "stateroll_top1": pct(total["stateroll_top1_count"], n),
            "stateroll_oracle": pct(total["stateroll_oracle_count"], n),
            "union_oracle": pct(total["union_oracle_count"], n),
            "bsl_only_success": pct(total["bsl_only_success_count"], n),
            "stateroll_only_success": pct(total["stateroll_only_success_count"], n),
            "both_success": pct(total["both_success_count"], n),
            "neither_success": pct(total["neither_success_count"], n),
            "bsl_rank0_failure": pct(total["bsl_rank0_failure_count"], n),
            "bsl_rank0_failure_fixable_any": pct(total["bsl_rank0_failure_fixable_any_count"], fail),
            "bsl_rank0_failure_fixable_bsl_only": pct(total["bsl_rank0_failure_fixable_bsl_only_count"], fail),
            "bsl_rank0_failure_fixable_stateroll_only": pct(total["bsl_rank0_failure_fixable_stateroll_only_count"], fail),
            "bsl_rank0_failure_fixable_both": pct(total["bsl_rank0_failure_fixable_both_count"], fail),
        }
    )
    return total


def main():
    rows = [summarize_seed(seed) for seed in SEEDS]
    total = aggregate(rows)
    all_rows = rows + [total]

    json_obj = {"overall": total, "per_seed": rows}
    (OUT_DIR / "candidate_complementarity_n100.json").write_text(json.dumps(json_obj, indent=2))

    fields = list(total.keys())
    # Put seed and count fields first in stable order.
    preferred = [
        "seed",
        "episodes",
        "bsl_top1",
        "bsl_oracle",
        "stateroll_top1",
        "stateroll_oracle",
        "union_oracle",
        "bsl_only_success_count",
        "bsl_only_success",
        "stateroll_only_success_count",
        "stateroll_only_success",
        "both_success_count",
        "both_success",
        "neither_success_count",
        "neither_success",
        "bsl_rank0_failure_count",
        "bsl_rank0_failure_fixable_any_count",
        "bsl_rank0_failure_fixable_bsl_only_count",
        "bsl_rank0_failure_fixable_stateroll_only_count",
        "bsl_rank0_failure_fixable_both_count",
        "bsl_rank0_failure_fixable_any",
        "bsl_rank0_failure_fixable_bsl_only",
        "bsl_rank0_failure_fixable_stateroll_only",
        "bsl_rank0_failure_fixable_both",
    ]
    fields = preferred + [f for f in fields if f not in preferred]
    with (OUT_DIR / "candidate_complementarity_n100.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    lines = [
        "# Candidate Complementarity n100",
        "",
        f"BSL pool: `{BSL}`",
        f"Stateroll pool: `{ST}`",
        "",
        "## Overall",
        "",
        "|metric|count|percent|",
        "|---|---:|---:|",
        f"|episodes|{total['episodes']}|100.0|",
        f"|bsl top1|{total['bsl_top1_count']}|{total['bsl_top1']:.1f}|",
        f"|bsl oracle|{total['bsl_oracle_count']}|{total['bsl_oracle']:.1f}|",
        f"|stateroll top1|{total['stateroll_top1_count']}|{total['stateroll_top1']:.1f}|",
        f"|stateroll oracle|{total['stateroll_oracle_count']}|{total['stateroll_oracle']:.1f}|",
        f"|union oracle|{total['union_oracle_count']}|{total['union_oracle']:.1f}|",
        f"|bsl only success|{total['bsl_only_success_count']}|{total['bsl_only_success']:.1f}|",
        f"|stateroll only success|{total['stateroll_only_success_count']}|{total['stateroll_only_success']:.1f}|",
        f"|both success|{total['both_success_count']}|{total['both_success']:.1f}|",
        f"|neither success|{total['neither_success_count']}|{total['neither_success']:.1f}|",
        "",
        "## Among BSL Rank0 Failures",
        "",
        "|metric|count|percent of bsl rank0 failures|",
        "|---|---:|---:|",
        f"|bsl rank0 failures|{total['bsl_rank0_failure_count']}|100.0|",
        f"|fixable by any pool|{total['bsl_rank0_failure_fixable_any_count']}|{total['bsl_rank0_failure_fixable_any']:.1f}|",
        f"|fixable by bsl pool only|{total['bsl_rank0_failure_fixable_bsl_only_count']}|{total['bsl_rank0_failure_fixable_bsl_only']:.1f}|",
        f"|fixable by stateroll pool only|{total['bsl_rank0_failure_fixable_stateroll_only_count']}|{total['bsl_rank0_failure_fixable_stateroll_only']:.1f}|",
        f"|fixable by both pools|{total['bsl_rank0_failure_fixable_both_count']}|{total['bsl_rank0_failure_fixable_both']:.1f}|",
        "",
        "## Per Seed",
        "",
        "|seed|episodes|bsl top1|bsl oracle|st top1|st oracle|union oracle|bsl only|st only|both|neither|bsl fail|fix bsl only|fix st only|fix both|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"|{r['seed']}|{r['episodes']}|{r['bsl_top1']:.1f}|{r['bsl_oracle']:.1f}|"
            f"{r['stateroll_top1']:.1f}|{r['stateroll_oracle']:.1f}|{r['union_oracle']:.1f}|"
            f"{r['bsl_only_success_count']}|{r['stateroll_only_success_count']}|{r['both_success_count']}|{r['neither_success_count']}|"
            f"{r['bsl_rank0_failure_count']}|{r['bsl_rank0_failure_fixable_bsl_only_count']}|"
            f"{r['bsl_rank0_failure_fixable_stateroll_only_count']}|{r['bsl_rank0_failure_fixable_both_count']}|"
        )
    (OUT_DIR / "candidate_complementarity_n100.md").write_text("\n".join(lines) + "\n")
    print((OUT_DIR / "candidate_complementarity_n100.md").read_text())


if __name__ == "__main__":
    main()
