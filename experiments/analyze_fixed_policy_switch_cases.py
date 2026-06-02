from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

WM = Path("/data1/jingyixi/wm_runs")
OUT = WM / "env_traj_replacement_n100_20260527_104100"
ENV = WM / "env_traj_features_n100"
SWITCH_CSV = OUT / "fixed_policy_epoch30_vote3_switch_cases.csv"


def load_npz(seed: int, variant: str):
    return np.load(ENV / f"{variant}_seed{seed}.npz", allow_pickle=True)


def row_type(row):
    if row["fixed"] == "True":
        return "fixed"
    if row["harmed"] == "True":
        return "harmed"
    return "neutral"


def boolstr(x):
    return "True" if bool(x) else "False"


def summarize(rows, name):
    if not rows:
        return {"group": name, "n": 0}

    def avg(k):
        return float(np.mean([float(r[k]) for r in rows]))

    return {
        "group": name,
        "n": len(rows),
        "avg_rank": avg("candidate_rank"),
        "avg_score_margin": avg("score_margin"),
        "avg_detector_score": avg("detector_score"),
        "avg_final_distance_improvement": avg("final_distance_improvement"),
        "avg_final_progress_improvement": avg("final_progress_improvement"),
        "avg_min_distance_improvement": avg("min_distance_improvement"),
        "avg_old_cost_gap_selected_minus_bsl": avg("old_cost_gap_selected_minus_bsl"),
        "stateroll_rank0_success_count": int(sum(r["stateroll_rank0_success"] == "True" for r in rows)),
        "selected_top10_count": int(sum(int(r["within_stateroll_top10"]) for r in rows)),
        "selected_top20_count": int(sum(int(r["within_stateroll_top20"]) for r in rows)),
    }


def main():
    switch_rows = list(csv.DictReader(SWITCH_CSV.open()))
    by_seed = {}
    detailed = []
    for row in switch_rows:
        seed = int(row["seed"])
        ep = int(row["episode_in_seed"])
        union_j = int(row["candidate_index_union"])
        source_rank = int(row["source_rank"])
        if seed not in by_seed:
            by_seed[seed] = {"bsl": load_npz(seed, "baseline"), "st": load_npz(seed, "vf05_mix20")}
        bsl_pool = by_seed[seed]["bsl"]
        st_pool = by_seed[seed]["st"]
        selected_pool = st_pool if union_j >= 30 else bsl_pool
        selected_rank = source_rank

        bsl_old_cost = float(bsl_pool["costs"][ep, 0])
        selected_old_cost = float(selected_pool["costs"][ep, selected_rank])
        bsl_final_dist = float(bsl_pool["final_distance"][ep, 0])
        sel_final_dist = float(selected_pool["final_distance"][ep, selected_rank])
        bsl_final_angle = float(bsl_pool["final_angle_error"][ep, 0])
        sel_final_angle = float(selected_pool["final_angle_error"][ep, selected_rank])
        bsl_min_dist = float(bsl_pool["min_distance"][ep, 0])
        sel_min_dist = float(selected_pool["min_distance"][ep, selected_rank])
        bsl_final_prog = float(bsl_pool["final_progress"][ep, 0])
        sel_final_prog = float(selected_pool["final_progress"][ep, selected_rank])
        bsl_max_prog = float(bsl_pool["max_progress"][ep, 0])
        sel_max_prog = float(selected_pool["max_progress"][ep, selected_rank])
        bsl_contact = float(bsl_pool["contact_proxy"][ep, 0])
        sel_contact = float(selected_pool["contact_proxy"][ep, selected_rank])

        rec = {
            "split": row["split"],
            "seed": seed,
            "episode": ep,
            "case_type": row_type(row),
            "bsl_success": row["bsl_success"],
            "selected_success": row["selected_success"],
            "candidate_source": row["source"],
            "candidate_rank": selected_rank,
            "candidate_union_index": union_j,
            "bsl_old_cost": bsl_old_cost,
            "selected_old_cost": selected_old_cost,
            "old_cost_gap_selected_minus_bsl": selected_old_cost - bsl_old_cost,
            "score_margin": float(row["score_margin_selected_minus_bsl"]),
            "detector_score": float(row["detector_score"]),
            "votes": int(row["vote_count"]),
            "bsl_final_distance": bsl_final_dist,
            "selected_final_distance": sel_final_dist,
            "final_distance_improvement": bsl_final_dist - sel_final_dist,
            "bsl_final_angle_error": bsl_final_angle,
            "selected_final_angle_error": sel_final_angle,
            "final_angle_error_improvement": bsl_final_angle - sel_final_angle,
            "bsl_min_distance": bsl_min_dist,
            "selected_min_distance": sel_min_dist,
            "min_distance_improvement": bsl_min_dist - sel_min_dist,
            "bsl_final_progress": bsl_final_prog,
            "selected_final_progress": sel_final_prog,
            "final_progress_improvement": sel_final_prog - bsl_final_prog,
            "bsl_max_progress": bsl_max_prog,
            "selected_max_progress": sel_max_prog,
            "max_progress_improvement": sel_max_prog - bsl_max_prog,
            "bsl_contact_proxy": bsl_contact,
            "selected_contact_proxy": sel_contact,
            "contact_proxy_improvement": sel_contact - bsl_contact,
            "stateroll_rank0_success": boolstr(st_pool["labels"][ep, 0]),
            "within_stateroll_top10": int(row["source"] == "stateroll" and selected_rank < 10),
            "within_stateroll_top20": int(row["source"] == "stateroll" and selected_rank < 20),
            "union_oracle": row["union_oracle"],
        }
        detailed.append(rec)

    fields = list(detailed[0].keys()) if detailed else []
    with (OUT / "switch_case_analysis.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in detailed:
            writer.writerow(rec)

    fixed = [r for r in detailed if r["case_type"] == "fixed"]
    neutral = [r for r in detailed if r["case_type"] == "neutral"]
    harmed = [r for r in detailed if r["case_type"] == "harmed"]
    stats = [summarize(fixed, "fixed"), summarize(neutral, "neutral"), summarize(harmed, "harmed"), summarize(detailed, "all")]
    (OUT / "switch_case_analysis.json").write_text(json.dumps({"cases": detailed, "group_stats": stats}, indent=2))

    lines = [
        "# Switch Case Analysis",
        "",
        "Source fixed policy: `epoch=30, detector_topk=50, threshold=-0.5, vote_k=3, max_switch_frac=0.05`",
        "",
        f"Total switches: {len(detailed)}; fixed={len(fixed)}, neutral={len(neutral)}, harmed={len(harmed)}",
        "",
        "## Case Table",
        "",
        "|seed|episode|type|source|rank|bsl ok|selected ok|bsl cost|sel cost|cost gap|score margin|detector|votes|final dist imp|progress imp|st rank0 ok|top10|top20|",
        "|---:|---:|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for r in detailed:
        lines.append(
            f"|{r['seed']}|{r['episode']}|{r['case_type']}|{r['candidate_source']}|{r['candidate_rank']}|"
            f"{r['bsl_success']}|{r['selected_success']}|{r['bsl_old_cost']:.4f}|{r['selected_old_cost']:.4f}|"
            f"{r['old_cost_gap_selected_minus_bsl']:.4f}|{r['score_margin']:.4f}|{r['detector_score']:.4f}|{r['votes']}|"
            f"{r['final_distance_improvement']:.2f}|{r['final_progress_improvement']:.2f}|{r['stateroll_rank0_success']}|"
            f"{r['within_stateroll_top10']}|{r['within_stateroll_top20']}|"
        )

    lines += [
        "",
        "## Fixed vs Neutral",
        "",
        "|group|n|avg rank|avg score margin|avg detector|avg final dist imp|avg progress imp|avg min dist imp|avg old cost gap|st rank0 success|top10|top20|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in stats:
        if s["n"] == 0:
            lines.append(f"|{s['group']}|0|NA|NA|NA|NA|NA|NA|NA|0|0|0|")
        else:
            lines.append(
                f"|{s['group']}|{s['n']}|{s['avg_rank']:.2f}|{s['avg_score_margin']:.4f}|{s['avg_detector_score']:.4f}|"
                f"{s['avg_final_distance_improvement']:.2f}|{s['avg_final_progress_improvement']:.2f}|"
                f"{s['avg_min_distance_improvement']:.2f}|{s['avg_old_cost_gap_selected_minus_bsl']:.4f}|"
                f"{s['stateroll_rank0_success_count']}|{s['selected_top10_count']}|{s['selected_top20_count']}|"
            )

    lines += [
        "",
        "## Interpretation",
        "",
        "All switched candidates are from the stateroll pool. The fixed cases are not simply stateroll rank0: selected ranks are spread across the pool, showing that raw stateroll cost ranking is not sufficient. The fixed cases have larger average trajectory improvement than the neutral switches, while neutral switches are harmless because the selected candidate did not reduce the outcome.",
    ]
    (OUT / "switch_case_analysis.md").write_text("\n".join(lines) + "\n")
    print((OUT / "switch_case_analysis.md").read_text())


if __name__ == "__main__":
    main()
