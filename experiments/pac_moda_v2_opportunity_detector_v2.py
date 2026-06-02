from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("/data1/jingyixi/wm_runs")
OUT = ROOT / "pac_moda_v2_opportunity_detector_v2_20260529"
BSL_ACTION = ROOT / "bsl_normalbudget_candidate_pool_s300_steps30_n100" / "proposal_data"
BSL_RAW = ROOT / "bsl_normalbudget_candidate_pool_s300_steps30_n100" / "raw_rollout_npz"
ST_ACTION = ROOT / "stateroll_normalbudget_candidate_pool_s300_steps30_n100" / "proposal_data"
ST_RAW = ROOT / "stateroll_normalbudget_candidate_pool_s300_steps30_n100" / "raw_rollout_npz"
OLD_GATE = ROOT / "gate_only_opportunity_detector_n100_20260528" / "precision_gate_grid_n100.json"
GAIN_CASES = ROOT / "pac_moda_v2_gain_boost_20260529" / "pac_moda_v2_remaining_fixed_gate_cases.csv"
CORRECTED_JSON = ROOT / "pac_moda_v2_full_n100_corrected_20260529" / "pac_moda_v2_ablation_n100.json"
SEEDS = [42, 43, 44, 45, 46, 47]
SPLITS = {
    "splitA_train42_44_val45_47": ([42, 43, 44], [45, 46, 47]),
    "splitB_train45_47_val42_44": ([45, 46, 47], [42, 43, 44]),
}


def entropy_from_cost(costs):
    x = -costs.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    p = np.exp(x)
    p = p / (p.sum(axis=-1, keepdims=True) + 1e-12)
    return -(p * np.log(p + 1e-12)).sum(axis=-1)


def load_src(action_dir, raw_dir, variant, seed):
    a = np.load(Path(action_dir) / f"{variant}_seed{seed}.npz", allow_pickle=True)
    r = np.load(Path(raw_dir) / f"{variant}_seed{seed}.npz", allow_pickle=True)
    return {"actions": a["actions"].astype(np.float64), "costs": a["costs"].astype(np.float64), "labels": a["labels"].astype(bool), "pred": r["pred"].astype(np.float64), "goal": r["goal"].astype(np.float64), "indices": a["indices"]}


def goal_for_pred(goal, pred):
    g = goal
    if g.ndim == 2:
        g = g[:, None, :]
    if g.ndim == 3 and pred.ndim == 4:
        g = g[:, None, :, :]
    if g.shape[1] == 1:
        g = np.repeat(g, pred.shape[1], axis=1)
    if g.shape[2] == 1:
        g = np.repeat(g, pred.shape[2], axis=2)
    elif g.shape[2] != pred.shape[2]:
        g = g[:, :, -pred.shape[2] :, :]
    return g


def traj_stats(pred, goal):
    g = goal_for_pred(goal, pred)
    dist = np.sqrt(((pred - g) ** 2).sum(axis=-1))
    return {"final": dist[:, :, -1], "mean": dist.mean(axis=2), "min": dist.min(axis=2), "progress": dist[:, :, 0] - dist[:, :, -1], "latent_mean": pred.mean(axis=(2, 3)), "latent_std": pred.std(axis=(2, 3))}


def action_stats(actions):
    norm_t = np.sqrt((actions**2).sum(axis=-1))
    return {"norm": norm_t.mean(axis=2), "std": norm_t.std(axis=2)}


def safe_margin(sorted_cost, k):
    k = min(k, len(sorted_cost) - 1)
    return float(sorted_cost[k] - sorted_cost[0])


def hard_sets():
    hp = set()
    hn = set()
    if GAIN_CASES.exists():
        for r in csv.DictReader(GAIN_CASES.open()):
            if r.get("missed") == "True":
                hp.add((int(r["seed"]), int(r["episode"])))
    if OLD_GATE.exists():
        j = json.loads(OLD_GATE.read_text())
        for rec in j.get("records", []):
            if rec.get("topk") in [15, 20, 30] and rec.get("selected_count", 0) <= 30:
                for e in rec.get("selected_episodes", []):
                    if e.get("bsl_success"):
                        hn.add((int(e["seed"]), int(e["episode"])))
    return hp, hn


def build_rows():
    rows = []
    hp, hn = hard_sets()
    for seed in SEEDS:
        b = load_src(BSL_ACTION, BSL_RAW, "baseline", seed)
        st = load_src(ST_ACTION, ST_RAW, "vf05_mix20", seed)
        assert np.all(b["indices"] == st["indices"])
        b_tr, st_tr = traj_stats(b["pred"], b["goal"]), traj_stats(st["pred"], st["goal"])
        b_act, st_act = action_stats(b["actions"]), action_stats(st["actions"])
        for ep in range(b["labels"].shape[0]):
            bc, sc = b["costs"][ep], st["costs"][ep]
            bs, ss = np.sort(bc), np.sort(sc)
            b_best, st_best = int(np.argmin(bc)), int(np.argmin(sc))
            bsl_success = bool(b["labels"][ep, 0])
            bsl_oracle = bool(b["labels"][ep].any())
            st_oracle = bool(st["labels"][ep].any())
            fixable = (not bsl_success) and (bsl_oracle or st_oracle)
            st_only = (not bsl_success) and st_oracle and (not bsl_oracle)
            key = (seed, ep)
            rows.append({
                "seed": seed, "episode": ep, "bsl_success": bsl_success, "bsl_oracle": bsl_oracle, "st_oracle": st_oracle, "fixable": fixable, "stateroll_only_fixable": st_only,
                "hard_positive": key in hp and fixable, "hard_negative": key in hn and bsl_success,
                "bsl_margin_top2": safe_margin(bs, 1), "bsl_margin_top5": safe_margin(bs, 4), "bsl_margin_top10": safe_margin(bs, 9),
                "bsl_cost_std": float(bc.std()), "bsl_cost_entropy": float(entropy_from_cost(bc[None])[0]), "bsl_best_cost": float(bs[0]), "st_best_cost": float(ss[0]),
                "st_minus_bsl_best_cost": float(ss[0] - bs[0]), "st_mean5_minus_bsl_mean5_cost": float(ss[:5].mean() - bs[:5].mean()),
                "bsl_rank0_minus_st_best_final_dist": float(b_tr["final"][ep, 0] - st_tr["final"][ep, st_best]),
                "bsl_best_minus_st_best_final_dist": float(b_tr["final"][ep, b_best] - st_tr["final"][ep, st_best]),
                "bsl_rank0_minus_st_best_mean_dist": float(b_tr["mean"][ep, 0] - st_tr["mean"][ep, st_best]),
                "bsl_rank0_minus_st_best_min_dist": float(b_tr["min"][ep, 0] - st_tr["min"][ep, st_best]),
                "st_best_minus_bsl_rank0_progress": float(st_tr["progress"][ep, st_best] - b_tr["progress"][ep, 0]),
                "bsl_rank0_final_dist": float(b_tr["final"][ep, 0]), "bsl_rank0_mean_dist": float(b_tr["mean"][ep, 0]), "bsl_rank0_min_dist": float(b_tr["min"][ep, 0]), "bsl_rank0_progress": float(b_tr["progress"][ep, 0]),
                "st_best_final_dist": float(st_tr["final"][ep, st_best]), "st_best_mean_dist": float(st_tr["mean"][ep, st_best]), "st_best_min_dist": float(st_tr["min"][ep, st_best]), "st_best_progress": float(st_tr["progress"][ep, st_best]),
                "bsl_action_norm": float(b_act["norm"][ep, 0]), "bsl_action_std": float(b_act["std"][ep, 0]), "st_best_action_norm": float(st_act["norm"][ep, st_best]), "st_best_action_std": float(st_act["std"][ep, st_best]),
                "st_best_rank": float(st_best) / 29.0, "bsl_best_rank": float(b_best) / 29.0,
                "bsl_latent_mean": float(b_tr["latent_mean"][ep, 0]), "bsl_latent_std": float(b_tr["latent_std"][ep, 0]), "st_latent_mean": float(st_tr["latent_mean"][ep, st_best]), "st_latent_std": float(st_tr["latent_std"][ep, st_best]),
            })
    return rows


def feature_names(rows):
    excluded = {"seed", "episode", "bsl_success", "bsl_oracle", "st_oracle", "fixable", "stateroll_only_fixable", "hard_positive", "hard_negative"}
    return [k for k in rows[0] if k not in excluded]


def make_xy(rows, feats):
    x = np.asarray([[r[k] for k in feats] for r in rows], dtype=np.float64)
    y = np.asarray([r["fixable"] for r in rows], dtype=np.float64)
    w = np.ones(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        if r["hard_positive"]:
            w[i] = 7.0
        elif r["stateroll_only_fixable"]:
            w[i] = 3.0
        elif r["fixable"]:
            w[i] = 2.0
        if r["hard_negative"]:
            w[i] = max(w[i], 6.0)
        elif r["bsl_success"]:
            w[i] = max(w[i], 3.0)
    return x, y, w


def fit_logistic(xtr, ytr, wtr, xva, lr=0.04, epochs=4000, l2=1e-3):
    mean, std = xtr.mean(axis=0), xtr.std(axis=0) + 1e-6
    xs, xv = (xtr - mean) / std, (xva - mean) / std
    xb = np.concatenate([xs, np.ones((xs.shape[0], 1))], axis=1)
    vb = np.concatenate([xv, np.ones((xv.shape[0], 1))], axis=1)
    w = np.zeros(xb.shape[1])
    sw = wtr / (wtr.mean() + 1e-12)
    for _ in range(epochs):
        z = np.clip(xb @ w, -40, 40)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = (xb.T @ ((p - ytr) * sw)) / len(ytr)
        grad[:-1] += l2 * w[:-1]
        w -= lr * grad
    return 1.0 / (1.0 + np.exp(-np.clip(vb @ w, -40, 40)))


def fit_mlp(xtr, ytr, wtr, xva, hidden=24, lr=0.01, epochs=2500, l2=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    mean, std = xtr.mean(axis=0), xtr.std(axis=0) + 1e-6
    xs, xv = (xtr - mean) / std, (xva - mean) / std
    W1 = rng.normal(0, 0.08, size=(xs.shape[1], hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.08, size=(hidden,))
    b2 = 0.0
    sw = wtr / (wtr.mean() + 1e-12)
    for _ in range(epochs):
        hpre = xs @ W1 + b1
        h = np.tanh(hpre)
        z = np.clip(h @ W2 + b2, -40, 40)
        p = 1.0 / (1.0 + np.exp(-z))
        dz = ((p - ytr) * sw) / len(ytr)
        gW2 = h.T @ dz + l2 * W2
        gb2 = dz.sum()
        dh = dz[:, None] * W2[None, :]
        dhpre = dh * (1.0 - h * h)
        gW1 = xs.T @ dhpre + l2 * W1
        gb1 = dhpre.sum(axis=0)
        W2 -= lr * gW2
        b2 -= lr * gb2
        W1 -= lr * gW1
        b1 -= lr * gb1
    hv = np.tanh(xv @ W1 + b1)
    return 1.0 / (1.0 + np.exp(-np.clip(hv @ W2 + b2, -40, 40)))


def selected_topk(scores, k):
    sel = np.zeros(len(scores), dtype=bool)
    order = np.argsort(-scores, kind="stable")
    sel[order[: min(k, len(scores))]] = True
    return sel


def metric(split, model, mode, rows, scores, sel):
    bsl = np.asarray([r["bsl_success"] for r in rows], dtype=bool)
    st = np.asarray([r["stateroll_only_fixable"] for r in rows], dtype=bool)
    fix = np.asarray([r["fixable"] for r in rows], dtype=bool)
    nonfix = (~bsl) & (~fix)
    n = int(sel.sum())
    return {"split": split, "model": model, "mode": mode, "selected": n, "st_only_captured": int((sel & st).sum()), "st_only_total": int(st.sum()), "fixable_captured": int((sel & fix).sum()), "fixable_total": int(fix.sum()), "bsl_success_fp": int((sel & bsl).sum()), "nonfixable_fp": int((sel & nonfix).sum()), "precision": float((sel & fix).sum() / n) if n else 0.0, "recall": float((sel & fix).sum() / max(1, fix.sum())), "selected_episodes": [{"seed": int(rows[i]["seed"]), "episode": int(rows[i]["episode"]), "score": float(scores[i]), "bsl_success": bool(rows[i]["bsl_success"]), "fixable": bool(rows[i]["fixable"]), "stateroll_only_fixable": bool(rows[i]["stateroll_only_fixable"])} for i in np.where(sel)[0]]}


def deploy(gate_row, pac_selected):
    fixed = harmed = switches = st = 0
    for e in gate_row["selected_episodes"]:
        key = (int(e["seed"]), int(e["episode"]))
        if key not in pac_selected:
            continue
        s = pac_selected[key]
        switches += 1
        succ, bsl = bool(s["best_success"]), bool(s["bsl_success"])
        if (not bsl) and succ:
            fixed += 1
        if bsl and (not succ):
            harmed += 1
        if bool(s["stateroll_only_fixable"]) and succ:
            st += 1
    out = {k: v for k, v in gate_row.items() if k != "selected_episodes"}
    out.update({"fixed": fixed, "harmed": harmed, "net": fixed - harmed, "switches": switches, "stateroll_only_recovered": st})
    return out


def pac_selected(split):
    j = json.loads(CORRECTED_JSON.read_text())
    rec = next(r for r in j["records"] if r["split"] == split and r["method"] == "legacy_rank_combined")
    return {(int(r["seed"]), int(r["episode"])): r for r in rec["final"]["selected_rows"]}


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys and k != "selected_episodes":
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    feats = feature_names(rows)
    records, deployments = [], []
    split_defs = dict(SPLITS)
    for held in SEEDS:
        split_defs[f"loso_hold{held}"] = ([s for s in SEEDS if s != held], [held])
    for split, (tr, va) in split_defs.items():
        train, val = [r for r in rows if r["seed"] in tr], [r for r in rows if r["seed"] in va]
        xtr, ytr, wtr = make_xy(train, feats)
        xva, _, _ = make_xy(val, feats)
        score_map = {"logistic": fit_logistic(xtr, ytr, wtr, xva)}
        if not split.startswith("loso"):
            score_map["mlp"] = fit_mlp(xtr, ytr, wtr, xva, seed=0)
        for model, scores in score_map.items():
            for k in [5, 10, 15, 20, 30]:
                records.append(metric(split, model, f"top{k}", val, scores, selected_topk(scores, k)))
            for fp_cap in [3, 5, 8]:
                order = np.argsort(-scores, kind="stable")
                sel = np.zeros(len(scores), dtype=bool)
                fp = 0
                for idx in order:
                    if val[idx]["bsl_success"] and fp >= fp_cap:
                        continue
                    sel[idx] = True
                    if val[idx]["bsl_success"]:
                        fp += 1
                    if sel.sum() >= 30:
                        break
                records.append(metric(split, model, f"fp_le_{fp_cap}_max30", val, scores, sel))
    for r in records:
        if r["split"] in SPLITS:
            deployments.append(deploy(r, pac_selected(r["split"])))
    write_csv(OUT / "pac_moda_v2_opportunity_detector_v2.csv", records)
    write_csv(OUT / "pac_moda_v2_detector_v2_deployment.csv", deployments)
    (OUT / "pac_moda_v2_opportunity_detector_v2.json").write_text(json.dumps({"features": feats, "records": records}, indent=2))
    (OUT / "pac_moda_v2_detector_v2_deployment.json").write_text(json.dumps({"records": deployments}, indent=2))
    lines = ["# PAC-MoDA v2 Opportunity Detector v2", "", "Pure numpy detector because sklearn is unavailable in the remote runtime. ExtraTrees/RF/MLP-sklearn are marked unavailable; implemented models are weighted logistic and a small numpy MLP for splitA/B.", "", "## Gate Metrics", "", "|split|model|mode|selected|fixable cap|st-only cap|bsl FP|nonfix FP|precision|recall|", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split in SPLITS:
        for cap in [3, 5, 8]:
            cand = [r for r in records if r["split"] == split and r["bsl_success_fp"] <= cap and r["selected"] > 0]
            cand = sorted(cand, key=lambda r: (r["fixable_captured"], r["st_only_captured"], r["precision"], -r["bsl_success_fp"]), reverse=True)[:3]
            lines.append(f"### FP <= {cap}")
            for r in cand:
                lines.append(f"|{r['split']}|{r['model']}|{r['mode']}|{r['selected']}|{r['fixable_captured']}/{r['fixable_total']}|{r['st_only_captured']}/{r['st_only_total']}|{r['bsl_success_fp']}|{r['nonfixable_fp']}|{r['precision']:.3f}|{r['recall']:.3f}|")
    lines.extend(["", "## Deployment with Corrected PAC-MoDA v2 Rank-Preserve", "", "|split|model|mode|selected|fixed|harmed|net|switches|st-only recovered|bsl FP|", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for hb in [0, 1, 2]:
        lines.append(f"### harmed <= {hb}")
        cand = sorted([r for r in deployments if r["harmed"] <= hb], key=lambda r: (r["net"], r["fixed"], -r["bsl_success_fp"]), reverse=True)[:8]
        for r in cand:
            lines.append(f"|{r['split']}|{r['model']}|{r['mode']}|{r['selected']}|{r['fixed']}|{r['harmed']}|{r['net']}|{r['switches']}|{r['stateroll_only_recovered']}|{r['bsl_success_fp']}|")
    (OUT / "pac_moda_v2_opportunity_detector_v2.md").write_text("\n".join(lines) + "\n")
    (OUT / "pac_moda_v2_detector_v2_deployment.md").write_text("\n".join(lines) + "\n")
    print((OUT / "pac_moda_v2_opportunity_detector_v2.md").read_text())


if __name__ == "__main__":
    main()
