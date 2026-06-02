from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

WM = Path("/data1/jingyixi/wm_runs")
sys.path.insert(0, str(WM))

import train_gate_only_opportunity_detector_n100 as gate


OUT = Path("/data1/jingyixi/wm_runs/gate_only_opportunity_detector_n100_20260528")


class Args:
    bsl_action_dir = "/data1/jingyixi/wm_runs/bsl_normalbudget_candidate_pool_s300_steps30_n100/proposal_data"
    bsl_raw_dir = "/data1/jingyixi/wm_runs/bsl_normalbudget_candidate_pool_s300_steps30_n100/raw_rollout_npz"
    st_action_dir = "/data1/jingyixi/wm_runs/stateroll_normalbudget_candidate_pool_s300_steps30_n100/proposal_data"
    st_raw_dir = "/data1/jingyixi/wm_runs/stateroll_normalbudget_candidate_pool_s300_steps30_n100/raw_rollout_npz"


def make_xy(rows, feature_names):
    x, y, w = gate.make_xy(rows, feature_names)
    return x, y, w


def fit_scores(model_name, train_rows, val_rows, feature_names, seed):
    xtr, ytr, wtr = make_xy(train_rows, feature_names)
    xva, _, _ = make_xy(val_rows, feature_names)
    model = gate.model_defs(seed)[model_name]
    scores = gate.fit_predict(model, xtr, ytr, wtr, xva)
    return np.asarray(scores, dtype=np.float64)


def split_defs(seeds):
    out = {
        "splitA_train42_44_val45_47": ([42, 43, 44], [45, 46, 47]),
        "splitB_train45_47_val42_44": ([45, 46, 47], [42, 43, 44]),
    }
    for held in seeds:
        out[f"loso_hold{held}"] = ([s for s in seeds if s != held], [held])
    return out


def selected_by_rule(rows, scores, rule):
    n = len(rows)
    order = np.argsort(-scores)
    topk_mask = np.zeros(n, dtype=bool)
    topk_mask[order[: min(rule["topk"], n)]] = True
    masks = [topk_mask]
    desc = [f"top{rule['topk']}"]

    margin = np.asarray([r["bsl_margin_top10"] for r in rows], dtype=np.float64)
    entropy = np.asarray([r["bsl_cost_entropy"] for r in rows], dtype=np.float64)
    st_gap = np.asarray([r["st_minus_bsl_best_cost"] for r in rows], dtype=np.float64)
    abs_gap = np.abs(st_gap)

    if rule.get("margin_q") is not None:
        thr = np.quantile(margin, rule["margin_q"])
        masks.append(margin <= thr)
        desc.append(f"margin_bottom{int(rule['margin_q']*100)}")
    if rule.get("entropy_q") is not None:
        thr = np.quantile(entropy, 1.0 - rule["entropy_q"])
        masks.append(entropy >= thr)
        desc.append(f"entropy_top{int(rule['entropy_q']*100)}")
    if rule.get("st_gap_q") is not None:
        thr = np.quantile(st_gap, rule["st_gap_q"])
        masks.append(st_gap <= thr)
        desc.append(f"st_gap_bottom{int(rule['st_gap_q']*100)}")
    if rule.get("abs_gap_q") is not None:
        thr = np.quantile(abs_gap, 1.0 - rule["abs_gap_q"])
        masks.append(abs_gap >= thr)
        desc.append(f"abs_gap_top{int(rule['abs_gap_q']*100)}")

    if rule["combine"] == "AND":
        selected = np.logical_and.reduce(masks)
    else:
        selected = topk_mask & np.logical_or.reduce(masks[1:]) if len(masks) > 1 else topk_mask
    return selected, "+".join(desc) + f"+{rule['combine']}"


def metric_row(split, model, rows, scores, rule):
    selected, rule_name = selected_by_rule(rows, scores, rule)
    idx = np.where(selected)[0]
    st_only = np.asarray([r["stateroll_only_fixable"] for r in rows], dtype=bool)
    opportunity = np.asarray([r["opportunity"] for r in rows], dtype=bool)
    bsl_success = np.asarray([r["bsl_success"] for r in rows], dtype=bool)
    nonfix_bsl_fail = (~bsl_success) & (~opportunity)
    n_sel = int(selected.sum())
    st_cap = int((selected & st_only).sum())
    opp_cap = int((selected & opportunity).sum())
    bsl_fp = int((selected & bsl_success).sum())
    nonfix_fp = int((selected & nonfix_bsl_fail).sum())
    per_seed = {}
    for seed in sorted({r["seed"] for r in rows}):
        m = np.asarray([r["seed"] == seed for r in rows], dtype=bool)
        per_seed[str(seed)] = {
            "selected": int((selected & m).sum()),
            "st_only_capture": int((selected & m & st_only).sum()),
            "st_only_total": int((m & st_only).sum()),
            "bsl_success_fp": int((selected & m & bsl_success).sum()),
            "nonfix_bsl_failure_fp": int((selected & m & nonfix_bsl_fail).sum()),
        }
    return {
        "split": split,
        "model": model,
        "rule": rule_name,
        "topk": rule["topk"],
        "combine": rule["combine"],
        "margin_q": rule.get("margin_q"),
        "entropy_q": rule.get("entropy_q"),
        "st_gap_q": rule.get("st_gap_q"),
        "abs_gap_q": rule.get("abs_gap_q"),
        "selected_count": n_sel,
        "stateroll_only_captured": st_cap,
        "stateroll_only_total": int(st_only.sum()),
        "opportunity_captured": opp_cap,
        "opportunity_total": int(opportunity.sum()),
        "bsl_success_false_positive": bsl_fp,
        "bsl_failure_nonfixable_false_positive": nonfix_fp,
        "precision_stateroll_only": float(st_cap / n_sel) if n_sel else 0.0,
        "precision_opportunity": float(opp_cap / n_sel) if n_sel else 0.0,
        "per_seed": per_seed,
        "selected_episodes": [
            {
                "seed": int(rows[i]["seed"]),
                "episode": int(rows[i]["episode"]),
                "score": float(scores[i]),
                "stateroll_only_fixable": bool(rows[i]["stateroll_only_fixable"]),
                "opportunity": bool(rows[i]["opportunity"]),
                "bsl_success": bool(rows[i]["bsl_success"]),
            }
            for i in idx
        ],
    }


def rule_grid():
    topks = [5, 10, 15, 20, 30, 50]
    qs = [0.10, 0.20, 0.30, 0.40]
    rules = []
    for topk in topks:
        rules.append({"topk": topk, "combine": "AND"})
        for q in qs:
            for key in ["margin_q", "entropy_q", "st_gap_q", "abs_gap_q"]:
                rules.append({"topk": topk, key: q, "combine": "AND"})
                rules.append({"topk": topk, key: q, "combine": "OR"})
        for q1 in qs:
            for q2 in qs:
                rules.append({"topk": topk, "margin_q": q1, "entropy_q": q2, "combine": "AND"})
                rules.append({"topk": topk, "margin_q": q1, "st_gap_q": q2, "combine": "AND"})
                rules.append({"topk": topk, "entropy_q": q1, "st_gap_q": q2, "combine": "AND"})
                rules.append({"topk": topk, "margin_q": q1, "entropy_q": q2, "combine": "OR"})
                rules.append({"topk": topk, "margin_q": q1, "st_gap_q": q2, "combine": "OR"})
                rules.append({"topk": topk, "entropy_q": q1, "st_gap_q": q2, "combine": "OR"})
    return rules


def flat_rows(rows):
    out = []
    for r in rows:
        d = {k: v for k, v in r.items() if k not in {"per_seed", "selected_episodes"}}
        out.append(d)
    return out


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_summary(path, rows):
    lines = ["# Precision Gate Grid n100", ""]
    lines.append("This scan reconstructs the deterministic ExtraTrees/RF detector scores from the existing gate-only detector setup, then applies conservative post-hoc filters. No residual adapter or final selector is trained.")
    lines.append("")
    lines.append("Target: bsl-success false positives <= 5 per split, stateroll-only captured >= 3 if possible.")
    lines.append("")
    key_splits = ["splitA_train42_44_val45_47", "splitB_train45_47_val42_44"]
    lines.append("## Target-Satisfying Rules")
    lines.append("")
    lines.append("|split|model|selected|st-only captured|bsl-success FP|nonfix bsl-failure FP|precision st-only|rule|")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    any_target = False
    for split in key_splits:
        cand = [r for r in rows if r["split"] == split and r["bsl_success_false_positive"] <= 5 and r["stateroll_only_captured"] >= 3]
        cand = sorted(cand, key=lambda r: (r["stateroll_only_captured"], r["precision_stateroll_only"], -r["bsl_success_false_positive"], -r["selected_count"]), reverse=True)[:10]
        for r in cand:
            any_target = True
            lines.append(f"|{r['split']}|{r['model']}|{r['selected_count']}|{r['stateroll_only_captured']}/{r['stateroll_only_total']}|{r['bsl_success_false_positive']}|{r['bsl_failure_nonfixable_false_positive']}|{r['precision_stateroll_only']:.3f}|{r['rule']}|")
    if not any_target:
        lines.append("|none|none|0|0|0|0|0|No rule met both target constraints.|")
    lines.append("")
    lines.append("## Best Low-FP Rules By Split")
    lines.append("")
    lines.append("|split|model|selected|st-only captured|bsl-success FP|nonfix bsl-failure FP|precision st-only|rule|")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for split in sorted({r["split"] for r in rows}):
        cand = [r for r in rows if r["split"] == split and r["bsl_success_false_positive"] <= 5 and r["selected_count"] > 0]
        cand = sorted(cand, key=lambda r: (r["stateroll_only_captured"], r["precision_stateroll_only"], -r["bsl_failure_nonfixable_false_positive"], -r["selected_count"]), reverse=True)[:3]
        for r in cand:
            lines.append(f"|{r['split']}|{r['model']}|{r['selected_count']}|{r['stateroll_only_captured']}/{r['stateroll_only_total']}|{r['bsl_success_false_positive']}|{r['bsl_failure_nonfixable_false_positive']}|{r['precision_stateroll_only']:.3f}|{r['rule']}|")
    path.write_text("\n".join(lines) + "\n")


def main():
    seeds = [42, 43, 44, 45, 46, 47]
    args = Args()
    rows = gate.build_rows(args, seeds)
    feature_names = [k for k in rows[0].keys() if k not in {"seed", "episode", "bsl_success", "bsl_oracle", "st_oracle", "union_oracle", "opportunity", "stateroll_only_fixable"}]
    all_rows = []
    for split, (tr, va) in split_defs(seeds).items():
        train_rows = [r for r in rows if r["seed"] in tr]
        val_rows = [r for r in rows if r["seed"] in va]
        for model in ["extratrees", "randomforest"]:
            seed = int(va[0]) if split.startswith("loso") else 0
            scores = fit_scores(model, train_rows, val_rows, feature_names, seed)
            for rule in rule_grid():
                all_rows.append(metric_row(split, model, val_rows, scores, rule))
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"records": all_rows}
    (OUT / "precision_gate_grid_n100.json").write_text(json.dumps(payload, indent=2))
    write_csv(OUT / "precision_gate_grid_n100.csv", flat_rows(all_rows))
    write_summary(OUT / "precision_gate_grid_n100.md", all_rows)
    print((OUT / "precision_gate_grid_n100.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
