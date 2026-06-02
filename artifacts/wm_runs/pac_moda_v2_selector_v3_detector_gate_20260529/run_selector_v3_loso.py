from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path("/data1/jingyixi/wm_runs")
BASE = ROOT / "pac_moda_v2_selector_v3_detector_gate_20260529"
RUN = BASE / "run_selector_v3.py"
OUT = BASE

spec = importlib.util.spec_from_file_location("selector_v3", RUN)
sv3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sv3)


def loso_gate(split: str):
    records = json.loads(sv3.DETECTOR_V2.read_text())["records"]
    for r in records:
        if r["split"] == split and r["model"] == "logistic" and r["mode"] == "fp_le_3_max30":
            gate = {(int(e["seed"]), int(e["episode"])) for e in r["selected_episodes"]}
            return gate, {"rule": r["mode"], "model": r["model"], "selected_count": r["selected"], "bsl_success_false_positive": r["bsl_success_fp"]}
    raise RuntimeError(f"missing loso gate {split}")


def eval_with_gate(gate_set, gate_row, meta, y, scores, val_seeds, threshold, candidate_topk=1):
    ep = sv3.by_episode(meta, y, scores, val_seeds, candidate_topk=candidate_topk)
    fixed = harmed = switches = st = total = bsl_succ = 0
    for key, v in ep.items():
        total += 1
        bsl_succ += int(v["bsl_success"])
        if key not in gate_set or v["best_score"] <= threshold:
            continue
        switches += 1
        if (not v["bsl_success"]) and v["best_success"]:
            fixed += 1
        if v["bsl_success"] and (not v["best_success"]):
            harmed += 1
        if v["stateroll_only_fixable"] and v["best_success"]:
            st += 1
    return {
        "fixed": fixed,
        "harmed": harmed,
        "net": fixed - harmed,
        "switches": switches,
        "stateroll_only_recovered": st,
        "bsl_top1": bsl_succ / total * 100.0,
        "selector_top1": (bsl_succ + fixed - harmed) / total * 100.0,
        "gate_selected": len(gate_set),
        "gate_bsl_fp": gate_row["bsl_success_false_positive"],
    }


def train_threshold_with_gate(gate_set, meta, y, scores, train_seeds, harmed_budget, candidate_topk=1):
    ep = sv3.by_episode(meta, y, scores, train_seeds, candidate_topk=candidate_topk)
    vals = np.asarray([v["best_score"] for k, v in ep.items() if k in gate_set], dtype=float)
    if len(vals) == 0:
        vals = np.asarray([v["best_score"] for v in ep.values()], dtype=float)
    thrs = np.unique(np.quantile(vals, np.linspace(0, 1, 17)).round(6))
    best = None
    for thr in thrs:
        r = eval_with_gate(gate_set, {"bsl_success_false_positive": 0}, meta, y, scores, train_seeds, thr, candidate_topk)
        if r["harmed"] > harmed_budget:
            continue
        key = (r["net"], r["fixed"], -r["harmed"], -r["switches"])
        if best is None or key > best[0]:
            best = (key, thr)
    if best is None:
        return float(thrs[-1])
    return float(best[1])


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


def main():
    data = sv3.load_all()
    X, y, meta, raw = sv3.candidate_matrix(data, sv3.SEEDS)
    rows = []
    for held in sv3.SEEDS:
        train_seeds = [s for s in sv3.SEEDS if s != held]
        split = f"loso_hold{held}"
        gate_set, gate_row = loso_gate(split)
        bce = sv3.normalize_by_train_st(meta, sv3.score_model(sv3.fit_bce(X, y, meta, train_seeds), X), train_seeds)
        rank = sv3.normalize_by_train_st(meta, sv3.score_model(sv3.fit_rank_preserve(X, y, meta, train_seeds), X), train_seeds)
        rawz = sv3.normalize_by_train_st(meta, raw, train_seeds)
        configs = [
            ("raw_only", rawz, 1),
            ("rank_preserve", rank, 1),
            ("bce", bce, 1),
            ("raw_top3", rawz, 3),
        ]
        for name, scores, cand_topk in configs:
            for hb in [0, 1, 2]:
                thr = train_threshold_with_gate(gate_set, meta, y, scores, train_seeds, hb, cand_topk)
                r = eval_with_gate(gate_set, gate_row, meta, y, scores, [held], thr, cand_topk)
                r.update({"held_seed": held, "method": name, "harmed_budget": hb, "threshold": thr, "candidate_topk": cand_topk})
                rows.append(r)
    write_csv(OUT / "pac_moda_v2_selector_v3_loso.csv", rows)
    (OUT / "pac_moda_v2_selector_v3_loso.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    lines = ["# PAC-MoDA v2 Selector v3 LOSO", "", "|method|harmed budget|fixed|harmed|net|switches|st-only recovered|", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in sorted({r["method"] for r in rows}):
        for hb in [0, 1, 2]:
            rs = [r for r in rows if r["method"] == method and r["harmed_budget"] == hb]
            lines.append(
                f"|{method}|{hb}|{sum(r['fixed'] for r in rs)}|{sum(r['harmed'] for r in rs)}|{sum(r['net'] for r in rs)}|{sum(r['switches'] for r in rs)}|{sum(r['stateroll_only_recovered'] for r in rs)}|"
            )
    (OUT / "pac_moda_v2_selector_v3_loso.md").write_text("\n".join(lines) + "\n")
    print((OUT / "pac_moda_v2_selector_v3_loso.md").read_text())


if __name__ == "__main__":
    main()
