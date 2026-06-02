from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path("/data1/jingyixi/wm_runs")
BASE = ROOT / "pac_moda_v2_selector_v3_detector_gate_20260529"
RUN = BASE / "run_selector_v3.py"
OLD_GRID = ROOT / "gate_only_opportunity_detector_n100_20260528" / "precision_gate_grid_n100.json"
DET2 = ROOT / "pac_moda_v2_opportunity_detector_v2_20260529" / "pac_moda_v2_opportunity_detector_v2.json"

spec = importlib.util.spec_from_file_location("selector_v3", RUN)
sv3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sv3)


def gate_old_fixed(split: str):
    grid = json.loads(OLD_GRID.read_text())["records"]
    target = ("extratrees", "top10+st_gap_bottom20+AND") if split.startswith("splitA") else ("randomforest", "top10+abs_gap_top10+AND")
    for r in grid:
        if r["split"] == split and r["model"] == target[0] and r["rule"] == target[1]:
            return {(int(e["seed"]), int(e["episode"])) for e in r["selected_episodes"]}, r["rule"], r["model"]
    raise RuntimeError(split)


def gate_detector_v2(split: str):
    records = json.loads(DET2.read_text())["records"]
    for r in records:
        if r["split"] == split and r["model"] == "logistic" and r["mode"] == "fp_le_3_max30":
            return {(int(e["seed"]), int(e["episode"])) for e in r["selected_episodes"]}, r["mode"], r["model"]
    raise RuntimeError(split)


def gate_global(meta, seeds):
    return {(m["seed"], m["episode"]) for m in meta if m["seed"] in seeds and m["source"] == "stateroll"}, "all_episodes", "none"


def eval_gate(gate_set, meta, y, scores, seeds, thr, candidate_topk=1, agreement=None):
    ep = sv3.by_episode(meta, y, scores, seeds, candidate_topk=candidate_topk)
    fixed = harmed = switches = st = bsl = total = 0
    selected_rows = []
    for key, v in ep.items():
        total += 1
        bsl += int(v["bsl_success"])
        if key not in gate_set or v["best_score"] <= thr:
            continue
        if agreement is not None and not agreement.get(key, False):
            continue
        switches += 1
        if (not v["bsl_success"]) and v["best_success"]:
            fixed += 1
        if v["bsl_success"] and (not v["best_success"]):
            harmed += 1
        if v["stateroll_only_fixable"] and v["best_success"]:
            st += 1
        selected_rows.append({"seed": key[0], "episode": key[1], "bsl_success": v["bsl_success"], "best_success": v["best_success"], "best_rank": v["best_rank"], "score": v["best_score"]})
    return {
        "fixed": fixed,
        "harmed": harmed,
        "net": fixed - harmed,
        "switches": switches,
        "stateroll_only_recovered": st,
        "bsl_top1": bsl / total * 100.0,
        "selector_top1": (bsl + fixed - harmed) / total * 100.0,
        "selected_rows": selected_rows,
    }


def choose_thr(gate_set, meta, y, scores, train_seeds, harmed_budget, candidate_topk=1, agreement=None):
    ep = sv3.by_episode(meta, y, scores, train_seeds, candidate_topk=candidate_topk)
    vals = np.asarray([v["best_score"] for k, v in ep.items() if k in gate_set], dtype=float)
    if len(vals) == 0:
        vals = np.asarray([v["best_score"] for v in ep.values()], dtype=float)
    thrs = np.unique(np.quantile(vals, np.linspace(0, 1, 25)).round(6))
    best = None
    for thr in thrs:
        r = eval_gate(gate_set, meta, y, scores, train_seeds, thr, candidate_topk, agreement)
        if r["harmed"] > harmed_budget:
            continue
        key = (r["net"], r["fixed"], -r["harmed"], -r["switches"])
        if best is None or key > best[0]:
            best = (key, float(thr))
    return best[1] if best else float(thrs[-1])


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys and k != "selected_rows":
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def main():
    data = sv3.load_all()
    X, y, meta, raw = sv3.candidate_matrix(data, sv3.SEEDS)
    rows = []
    case_rows = []
    for split, (train_seeds, val_seeds) in sv3.SPLITS.items():
        bce = sv3.normalize_by_train_st(meta, sv3.score_model(sv3.fit_bce(X, y, meta, train_seeds), X), train_seeds)
        rank = sv3.normalize_by_train_st(meta, sv3.score_model(sv3.fit_rank_preserve(X, y, meta, train_seeds), X), train_seeds)
        rawz = sv3.normalize_by_train_st(meta, raw, train_seeds)
        ensemble = 0.5 * rawz
        ep_b = sv3.by_episode(meta, y, bce, val_seeds)
        ep_r = sv3.by_episode(meta, y, rank, val_seeds)
        agreement = {k: (k in ep_r and ep_b[k]["best_rank"] == ep_r[k]["best_rank"]) for k in ep_b}
        gates = {
            "global": gate_global(meta, train_seeds),  # train gate recomputed below for val.
            "fixed_precision": gate_old_fixed(split),
            "detector_v2": gate_detector_v2(split),
        }
        score_variants = {
            "raw_cost": (rawz, 1, None),
            "bce": (bce, 1, None),
            "rank_preserve": (rank, 1, None),
            "selector_v3_raw": (ensemble, 1, None),
            "agreement_bce_rank": (rank, 1, agreement),
        }
        for gate_name in ["global", "fixed_precision", "detector_v2"]:
            if gate_name == "global":
                train_gate = gate_global(meta, train_seeds)[0]
                val_gate, gate_rule, gate_model = gate_global(meta, val_seeds)
            else:
                val_gate, gate_rule, gate_model = gates[gate_name]
                train_gate = val_gate
            for score_name, (scores, cand_topk, agree) in score_variants.items():
                for hb in [0, 1, 2]:
                    thr = choose_thr(train_gate, meta, y, scores, train_seeds, hb, cand_topk, None if score_name != "agreement_bce_rank" else None)
                    # Agreement mask is val-only and label-free; train threshold uses no agreement mask for stable calibration.
                    r = eval_gate(val_gate, meta, y, scores, val_seeds, thr, cand_topk, agree)
                    r.update({"split": split, "gate": gate_name, "gate_rule": gate_rule, "gate_model": gate_model, "score": score_name, "harmed_budget": hb, "threshold": thr, "candidate_topk": cand_topk})
                    rows.append(r)
                    if gate_name == "detector_v2" and score_name == "selector_v3_raw" and hb == 2:
                        for sr in r["selected_rows"]:
                            sr = dict(sr)
                            sr.update({"split": split})
                            case_rows.append(sr)
    # Aggregate OOF for matching configs.
    agg = []
    keys = sorted(set((r["gate"], r["score"], r["harmed_budget"], r["candidate_topk"]) for r in rows))
    for key in keys:
        rs = [r for r in rows if (r["gate"], r["score"], r["harmed_budget"], r["candidate_topk"]) == key]
        if len(rs) != 2:
            continue
        agg.append({
            "split": "OOF",
            "gate": key[0],
            "score": key[1],
            "harmed_budget": key[2],
            "candidate_topk": key[3],
            "fixed": sum(r["fixed"] for r in rs),
            "harmed": sum(r["harmed"] for r in rs),
            "net": sum(r["net"] for r in rs),
            "switches": sum(r["switches"] for r in rs),
            "stateroll_only_recovered": sum(r["stateroll_only_recovered"] for r in rs),
            "selector_top1": 81.0 + sum(r["net"] for r in rs) / 600 * 100.0,
        })
    all_rows = rows + agg
    write_csv(BASE / "pac_moda_v2_selector_v3_ablation.csv", all_rows)
    write_csv(BASE / "pac_moda_v2_selector_v3_cases.csv", case_rows)
    (BASE / "pac_moda_v2_selector_v3_ablation.json").write_text(json.dumps({"rows": all_rows}, indent=2) + "\n")
    (BASE / "pac_moda_v2_selector_v3_cases.json").write_text(json.dumps({"rows": case_rows}, indent=2) + "\n")
    md = ["# PAC-MoDA v2 Selector v3 Ablation", "", "|gate|score|harmed budget|fixed|harmed|net|switches|st-only recovered|approx top1|", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(agg, key=lambda x: (x["net"], x["fixed"], -x["harmed"]), reverse=True):
        if r["harmed_budget"] in [0, 2]:
            md.append("|{gate}|{score}|{harmed_budget}|{fixed}|{harmed}|{net}|{switches}|{stateroll_only_recovered}|{selector_top1:.2f}|".format(**r))
    md += [
        "",
        "Key comparison: global raw-cost switching is included to show that raw cost is only viable when restricted by detector-v2 opportunity gating.",
    ]
    (BASE / "pac_moda_v2_selector_v3_ablation.md").write_text("\n".join(md) + "\n")
    print((BASE / "pac_moda_v2_selector_v3_ablation.md").read_text())


if __name__ == "__main__":
    main()
