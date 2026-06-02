from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("/data1/jingyixi/wm_runs")
OUT = ROOT / "pac_moda_v2_selector_v3_detector_gate_20260529"
BSL_ACTION = ROOT / "bsl_normalbudget_candidate_pool_s300_steps30_n100" / "proposal_data"
BSL_RAW = ROOT / "bsl_normalbudget_candidate_pool_s300_steps30_n100" / "raw_rollout_npz"
ST_ACTION = ROOT / "stateroll_normalbudget_candidate_pool_s300_steps30_n100" / "proposal_data"
ST_RAW = ROOT / "stateroll_normalbudget_candidate_pool_s300_steps30_n100" / "raw_rollout_npz"
GRID = ROOT / "gate_only_opportunity_detector_n100_20260528" / "precision_gate_grid_n100.json"
DETECTOR_V2 = ROOT / "pac_moda_v2_opportunity_detector_v2_20260529" / "pac_moda_v2_opportunity_detector_v2.json"
SEEDS = [42, 43, 44, 45, 46, 47]
SPLITS = {
    "splitA_train42_44_val45_47": ([42, 43, 44], [45, 46, 47]),
    "splitB_train45_47_val42_44": ([45, 46, 47], [42, 43, 44]),
}


def load_src(action_dir: Path, raw_dir: Path, variant: str, seed: int) -> dict:
    a = np.load(action_dir / f"{variant}_seed{seed}.npz", allow_pickle=True)
    r = np.load(raw_dir / f"{variant}_seed{seed}.npz", allow_pickle=True)
    return {"actions": a["actions"].astype(np.float64), "costs": a["costs"].astype(np.float64), "labels": a["labels"].astype(bool), "pred": r["pred"].astype(np.float64), "goal": r["goal"].astype(np.float64), "indices": a["indices"]}


def load_all() -> dict[int, dict]:
    data = {}
    for seed in SEEDS:
        b = load_src(BSL_ACTION, BSL_RAW, "baseline", seed)
        st = load_src(ST_ACTION, ST_RAW, "vf05_mix20", seed)
        assert np.all(b["indices"] == st["indices"])
        data[seed] = {"b": b, "st": st}
    return data


def goal_for_pred(goal: np.ndarray, pred: np.ndarray) -> np.ndarray:
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


def traj_stats(pred: np.ndarray, goal: np.ndarray) -> dict:
    g = goal_for_pred(goal, pred)
    dist = np.sqrt(((pred - g) ** 2).sum(axis=-1))
    return {"final": dist[:, :, -1], "mean": dist.mean(axis=2), "min": dist.min(axis=2), "progress": dist[:, :, 0] - dist[:, :, -1], "latent_mean": pred.mean(axis=(2, 3)), "latent_std": pred.std(axis=(2, 3))}


def action_stats(actions: np.ndarray) -> dict:
    norm_t = np.sqrt((actions**2).sum(axis=-1))
    return {"norm": norm_t.mean(axis=2), "std": norm_t.std(axis=2)}


def entropy_from_cost(costs: np.ndarray) -> np.ndarray:
    x = -costs.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    p = np.exp(x)
    p = p / (p.sum(axis=-1, keepdims=True) + 1e-12)
    return -(p * np.log(p + 1e-12)).sum(axis=-1)


def zscore_row(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


def candidate_matrix(data: dict, seeds: list[int]):
    X, y, meta, raw_score = [], [], [], []
    for seed in seeds:
        b = data[seed]["b"]
        st = data[seed]["st"]
        btr = traj_stats(b["pred"], b["goal"])
        strj = traj_stats(st["pred"], st["goal"])
        bact = action_stats(b["actions"])
        stact = action_stats(st["actions"])
        for ep in range(b["labels"].shape[0]):
            bcost = b["costs"][ep]
            scost = st["costs"][ep]
            bs = np.sort(bcost)
            ss = np.sort(scost)
            union_z = zscore_row(np.concatenate([bcost, scost]))
            b_z = zscore_row(bcost)
            st_z = zscore_row(scost)
            bsl_success = bool(b["labels"][ep, 0])
            bsl_oracle = bool(b["labels"][ep].any())
            st_oracle = bool(st["labels"][ep].any())
            st_only = (not bsl_success) and st_oracle and (not bsl_oracle)
            st_best = int(np.argmin(scost))
            ep_feat = [
                float(bs[1] - bs[0]),
                float(bs[min(4, len(bs) - 1)] - bs[0]),
                float(bs[min(9, len(bs) - 1)] - bs[0]),
                float(bcost.std()),
                float(entropy_from_cost(bcost[None])[0]),
                float(ss[0] - bs[0]),
                float(btr["final"][ep, 0] - strj["final"][ep, st_best]),
                float(btr["mean"][ep, 0] - strj["mean"][ep, st_best]),
                float(strj["progress"][ep, st_best] - btr["progress"][ep, 0]),
            ]
            for source, labels, tr, act, costs, local_z, offset in [
                ("bsl", b["labels"][ep], btr, bact, bcost, b_z, 0),
                ("stateroll", st["labels"][ep], strj, stact, scost, st_z, 30),
            ]:
                for j in range(costs.shape[0]):
                    feat = [
                        1.0 if source == "stateroll" else 0.0,
                        float(j) / max(1, costs.shape[0] - 1),
                        float(costs[j]),
                        float(local_z[j]),
                        float(union_z[offset + j]),
                        float(costs[j] - costs.min()),
                        float(costs[j] - bcost[0]),
                        float(tr["final"][ep, j]),
                        float(tr["mean"][ep, j]),
                        float(tr["min"][ep, j]),
                        float(tr["progress"][ep, j]),
                        float(act["norm"][ep, j]),
                        float(act["std"][ep, j]),
                        float(tr["latent_mean"][ep, j]),
                        float(tr["latent_std"][ep, j]),
                        *ep_feat,
                        1.0 if bsl_success else 0.0,
                        1.0 if st_only else 0.0,
                    ]
                    X.append(feat)
                    y.append(bool(labels[j]))
                    raw_score.append(float(-costs[j]))
                    meta.append({"seed": seed, "episode": ep, "source": source, "local_rank": j, "bsl_success": bsl_success, "bsl_oracle": bsl_oracle, "st_oracle": st_oracle, "stateroll_only_fixable": st_only})
    return np.asarray(X), np.asarray(y, dtype=bool), meta, np.asarray(raw_score)


def st_episode_groups(meta, seeds):
    groups = {}
    for i, m in enumerate(meta):
        if m["seed"] in seeds and m["source"] == "stateroll":
            groups.setdefault((m["seed"], m["episode"]), []).append(i)
    return groups


def fit_bce(X, y, meta, train_seeds, lr=0.03, epochs=2500, l2=1e-3):
    mask = np.asarray([m["seed"] in train_seeds for m in meta], dtype=bool)
    mean = X[mask].mean(axis=0)
    std = X[mask].std(axis=0) + 1e-6
    Xs = (X - mean) / std
    Xb = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    idx = np.where(mask)[0]
    yy = y[idx].astype(float)
    pos = yy.sum()
    neg = len(yy) - pos
    weights = np.ones(len(idx))
    weights[yy.astype(bool)] = max(1.0, neg / max(1.0, pos))
    weights = weights / (weights.mean() + 1e-12)
    w = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        z = np.clip(Xb[idx] @ w, -40, 40)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = (Xb[idx].T @ ((p - yy) * weights)) / len(idx)
        grad[:-1] += l2 * w[:-1]
        w -= lr * grad
    return {"w": w, "mean": mean, "std": std, "sigmoid": True}


def fit_rank_preserve(X, y, meta, train_seeds, lr=0.02, epochs=1800, l2=1e-3, margin=0.25):
    mask = np.asarray([m["seed"] in train_seeds for m in meta], dtype=bool)
    mean = X[mask].mean(axis=0)
    std = X[mask].std(axis=0) + 1e-6
    Xs = (X - mean) / std
    Xb = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    pairs, groups, preserve = [], [], []
    for idxs0 in st_episode_groups(meta, train_seeds).values():
        idxs = np.asarray(idxs0)
        pos = idxs[y[idxs]]
        neg = idxs[~y[idxs]]
        if len(pos):
            groups.append(idxs)
        if len(pos) and len(neg):
            hard = sorted(neg.tolist(), key=lambda i: (meta[i]["local_rank"], X[i, 2]))[: min(8, len(neg))]
            for pi in pos[: min(6, len(pos))]:
                for ni in hard:
                    pairs.append((int(pi), int(ni)))
        if meta[idxs[0]]["bsl_success"] and len(neg):
            preserve.extend(sorted(neg.tolist(), key=lambda j: (meta[j]["local_rank"], X[j, 2]))[: min(10, len(neg))])
    pairs = np.asarray(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64)
    preserve = np.asarray(preserve, dtype=np.int64) if preserve else np.zeros(0, dtype=np.int64)
    w = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        grad = np.zeros_like(w)
        if len(pairs):
            diff = Xb[pairs[:, 0]] - Xb[pairs[:, 1]]
            z = np.clip(diff @ w - margin, -40, 40)
            grad += 0.7 * (diff.T @ (-1.0 / (1.0 + np.exp(z)))) / len(pairs)
        if groups:
            g2 = np.zeros_like(w)
            for idxs in groups:
                z = np.clip(Xb[idxs] @ w, -40, 40)
                z -= z.max()
                p = np.exp(z)
                p /= p.sum() + 1e-12
                t = y[idxs].astype(float)
                t /= t.sum() + 1e-12
                g2 += Xb[idxs].T @ (p - t)
            grad += 0.3 * g2 / len(groups)
        if len(preserve):
            z = np.clip(Xb[preserve] @ w, -40, 40)
            p = 1.0 / (1.0 + np.exp(-z))
            grad += 0.15 * (Xb[preserve].T @ p) / len(preserve)
        grad[:-1] += l2 * w[:-1]
        w -= lr * grad
    return {"w": w, "mean": mean, "std": std, "sigmoid": False}


def score_model(model, X):
    Xs = (X - model["mean"]) / model["std"]
    Xb = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    z = np.clip(Xb @ model["w"], -40, 40)
    if model.get("sigmoid"):
        return 1.0 / (1.0 + np.exp(-z))
    return Xb @ model["w"]


def normalize_by_train_st(meta, scores, train_seeds):
    mask = np.asarray([m["seed"] in train_seeds and m["source"] == "stateroll" for m in meta], dtype=bool)
    return (scores - scores[mask].mean()) / (scores[mask].std() + 1e-6)


def fixed_gate_selected(split: str):
    records = json.loads(DETECTOR_V2.read_text())["records"]
    # High-recall but still controlled detector-v2 gate.
    target = ("logistic", "fp_le_3_max30")
    for r in records:
        if r["split"] == split and r["model"] == target[0] and r["mode"] == target[1]:
            gate = {(int(e["seed"]), int(e["episode"])) for e in r["selected_episodes"]}
            return gate, {"rule": r["mode"], "model": r["model"], "selected_count": r["selected"], "bsl_success_false_positive": r["bsl_success_fp"]}
    raise RuntimeError(f"missing detector-v2 gate {split}")


def by_episode(meta, y, scores, seeds, candidate_topk=1):
    out = {}
    for i, m in enumerate(meta):
        if m["seed"] not in seeds or m["source"] != "stateroll":
            continue
        key = (m["seed"], m["episode"])
        v = out.setdefault(key, {"idxs": [], "bsl_success": m["bsl_success"], "stateroll_only_fixable": m["stateroll_only_fixable"]})
        v["idxs"].append(i)
    for v in out.values():
        idxs = np.asarray(v["idxs"])
        order = idxs[np.argsort(-scores[idxs], kind="stable")]
        # Candidate_topk is a deployment rule selected on train split. It does not use labels.
        pick = order[min(candidate_topk, len(order)) - 1]
        v["best_idx"] = int(pick)
        v["best_score"] = float(scores[pick])
        v["best_success"] = bool(y[pick])
        v["best_rank"] = int(meta[pick]["local_rank"])
        v["top_success"] = bool(y[order[0]])
        v["top2_success"] = bool(y[order[:2]].any())
        v["top3_success"] = bool(y[order[:3]].any())
    return out


def evaluate(split, meta, y, scores, val_seeds, threshold, agreement_mask=None, candidate_topk=1):
    gate_set, gate_row = fixed_gate_selected(split)
    ep = by_episode(meta, y, scores, val_seeds, candidate_topk=candidate_topk)
    fixed = harmed = switches = st = total = bsl_succ = 0
    rows = []
    for key, v in ep.items():
        total += 1
        bsl_succ += int(v["bsl_success"])
        if key not in gate_set or v["best_score"] <= threshold:
            continue
        if agreement_mask is not None and not agreement_mask.get(key, False):
            continue
        switches += 1
        if (not v["bsl_success"]) and v["best_success"]:
            fixed += 1
        if v["bsl_success"] and (not v["best_success"]):
            harmed += 1
        if v["stateroll_only_fixable"] and v["best_success"]:
            st += 1
        rows.append({"seed": key[0], "episode": key[1], **{k: v[k] for k in ["bsl_success", "stateroll_only_fixable", "best_score", "best_success", "best_rank"]}})
    return {"split": split, "threshold": float(threshold), "candidate_topk": int(candidate_topk), "fixed": fixed, "harmed": harmed, "net": fixed - harmed, "switches": switches, "stateroll_only_recovered": st, "bsl_top1": bsl_succ / total * 100.0, "selector_top1": (bsl_succ + fixed - harmed) / total * 100.0, "gate_selected": len(gate_set), "gate_rule": gate_row["rule"], "selected_rows": rows}


def train_thresholds(split, meta, y, scores, train_seeds, harmed_budget, candidate_topk=1):
    # Train-only threshold selection, but still restricted to the split's fixed gate rows.
    gate_set, _ = fixed_gate_selected(split)
    ep = by_episode(meta, y, scores, train_seeds, candidate_topk=candidate_topk)
    vals = np.asarray([v["best_score"] for k, v in ep.items() if k in gate_set], dtype=float)
    if len(vals) == 0:
        vals = np.asarray([v["best_score"] for v in ep.values()], dtype=float)
    thrs = np.unique(np.quantile(vals, np.linspace(0, 1, 17)).round(6))
    rows = []
    for thr in thrs:
        r = evaluate(split, meta, y, scores, train_seeds, thr, candidate_topk=candidate_topk)
        rows.append(r)
    safe = [r for r in rows if r["harmed"] <= harmed_budget]
    cand = safe if safe else rows
    best = max(cand, key=lambda r: (r["net"], r["fixed"], -r["harmed"], -r["switches"]))
    return best["threshold"], rows


def case_rows(split, meta, y, scores, raw_scores, val_seeds, threshold):
    gate_set, _ = fixed_gate_selected(split)
    ep = by_episode(meta, y, scores, val_seeds)
    raw_ep = by_episode(meta, y, raw_scores, val_seeds)
    rows = []
    for key, v in ep.items():
        if key not in gate_set:
            continue
        idxs = np.asarray(v["idxs"])
        labels = y[idxs]
        order = idxs[np.argsort(-scores[idxs], kind="stable")]
        raw_order = idxs[np.argsort(-raw_scores[idxs], kind="stable")]
        if not labels.any():
            continue
        fixed = (not v["bsl_success"]) and v["best_success"] and v["best_score"] > threshold
        missed = (not v["bsl_success"]) and labels.any() and not fixed
        if fixed or missed:
            rows.append({
                "split": split,
                "seed": key[0],
                "episode": key[1],
                "fixed": fixed,
                "missed": missed,
                "bsl_success": v["bsl_success"],
                "stateroll_only_fixable": v["stateroll_only_fixable"],
                "success_count": int(labels.sum()),
                "raw_first_success_rank": int(np.where(y[raw_order])[0][0] + 1),
                "cal_first_success_rank": int(np.where(y[order])[0][0] + 1),
                "selected_rank": v["best_rank"],
                "selected_success": v["best_success"],
                "selected_score": v["best_score"],
                "threshold": threshold,
                "top2_has_success": v["top2_success"],
                "top3_has_success": v["top3_success"],
                "raw_selected_rank": raw_ep[key]["best_rank"],
                "raw_selected_success": raw_ep[key]["best_success"],
            })
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
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
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_all()
    X, y, meta, raw = candidate_matrix(data, SEEDS)
    pareto_rows, ensemble_rows, agreement_rows, remaining = [], [], [], []
    for split, (train_seeds, val_seeds) in SPLITS.items():
        bce = normalize_by_train_st(meta, score_model(fit_bce(X, y, meta, train_seeds), X), train_seeds)
        rank = normalize_by_train_st(meta, score_model(fit_rank_preserve(X, y, meta, train_seeds), X), train_seeds)
        rawz = normalize_by_train_st(meta, raw, train_seeds)
        score_bank = {"bce": bce, "rank_preserve": rank, "raw_cost": rawz}
        weight_grid = []
        for a in [0.0, 0.5, 1.0]:
            for b in [0.0, 0.5, 1.0]:
                for c in [0.0, 0.5]:
                    if a + b + c == 0:
                        continue
                    weight_grid.append((a, b, c))
        for a, b, c in weight_grid:
            scores = a * bce + b * rank + c * rawz
            for cand_topk in [1, 2, 3, 5]:
                for hb in [0, 1, 2, 3]:
                    thr, _ = train_thresholds(split, meta, y, scores, train_seeds, hb, candidate_topk=cand_topk)
                    r = evaluate(split, meta, y, scores, val_seeds, thr, candidate_topk=cand_topk)
                    r.update({"method": "ensemble", "split": split, "harmed_budget": hb, "w_bce": a, "w_rank": b, "w_raw": c})
                    ensemble_rows.append(r)
        # Standalone corrected rank-preserve operating points.
        for hb in [0, 1, 2, 3]:
            thr, train_curve = train_thresholds(split, meta, y, rank, train_seeds, hb, candidate_topk=1)
            r = evaluate(split, meta, y, rank, val_seeds, thr, candidate_topk=1)
            r.update({"method": "rank_preserve", "split": split, "harmed_budget": hb, "w_bce": 0, "w_rank": 1, "w_raw": 0})
            pareto_rows.append(r)
            if hb == 0:
                remaining.extend(case_rows(split, meta, y, rank, raw, val_seeds, thr))
        # Agreement: top candidate local rank from BCE and rank-preserve must agree.
        ep_b = by_episode(meta, y, bce, val_seeds)
        ep_r = by_episode(meta, y, rank, val_seeds)
        agree = {k: (k in ep_r and ep_b[k]["best_rank"] == ep_r[k]["best_rank"]) for k in ep_b}
        for hb in [0, 1, 2, 3]:
            thr, _ = train_thresholds(split, meta, y, rank, train_seeds, hb, candidate_topk=1)
            r = evaluate(split, meta, y, rank, val_seeds, thr, agreement_mask=agree, candidate_topk=1)
            r.update({"method": "agreement_bce_rank", "split": split, "harmed_budget": hb})
            agreement_rows.append(r)
    # OOF best ensemble per harmed budget selected by train-only split rows already; aggregate best val per split for each hb by train-selected row score. For reporting search frontier, pick max OOF net from evaluated grids.
    best_oof = []
    for hb in [0, 1, 2, 3]:
        # pair rows by same weights across splits
        combos = sorted({(r["w_bce"], r["w_rank"], r["w_raw"], r["candidate_topk"]) for r in ensemble_rows if r["harmed_budget"] == hb})
        best = None
        for combo in combos:
            rows = [r for r in ensemble_rows if r["harmed_budget"] == hb and (r["w_bce"], r["w_rank"], r["w_raw"], r["candidate_topk"]) == combo]
            if len(rows) != 2:
                continue
            agg = {"method": "ensemble_oof", "harmed_budget": hb, "w_bce": combo[0], "w_rank": combo[1], "w_raw": combo[2], "candidate_topk": combo[3], "fixed": sum(r["fixed"] for r in rows), "harmed": sum(r["harmed"] for r in rows), "net": sum(r["net"] for r in rows), "switches": sum(r["switches"] for r in rows), "stateroll_only_recovered": sum(r["stateroll_only_recovered"] for r in rows)}
            if best is None or (agg["harmed"] <= hb and (agg["net"], agg["fixed"], -agg["harmed"]) > (best["net"], best["fixed"], -best["harmed"])):
                best = agg
        if best:
            best_oof.append(best)
    rank_oof = []
    agree_oof = []
    for hb in [0, 1, 2, 3]:
        for src, dst in [(pareto_rows, rank_oof), (agreement_rows, agree_oof)]:
            rows = [r for r in src if r["harmed_budget"] == hb]
            dst.append({"harmed_budget": hb, "fixed": sum(r["fixed"] for r in rows), "harmed": sum(r["harmed"] for r in rows), "net": sum(r["net"] for r in rows), "switches": sum(r["switches"] for r in rows), "stateroll_only_recovered": sum(r["stateroll_only_recovered"] for r in rows)})
    write_csv(OUT / "pac_moda_v2_score_ensemble_pareto.csv", ensemble_rows + best_oof)
    write_csv(OUT / "pac_moda_v2_gain_boost_pareto.csv", pareto_rows + rank_oof + best_oof)
    write_csv(OUT / "pac_moda_v2_agreement_deployment.csv", agreement_rows + agree_oof)
    write_csv(OUT / "pac_moda_v2_remaining_fixed_gate_cases.csv", remaining)
    (OUT / "pac_moda_v2_gain_boost_pareto.json").write_text(json.dumps({"rank_rows": pareto_rows, "rank_oof": rank_oof, "ensemble_best_oof": best_oof, "agreement_oof": agree_oof}, indent=2))
    (OUT / "pac_moda_v2_score_ensemble_pareto.json").write_text(json.dumps({"rows": ensemble_rows, "best_oof": best_oof}, indent=2))
    (OUT / "pac_moda_v2_agreement_deployment.json").write_text(json.dumps({"rows": agreement_rows, "oof": agree_oof}, indent=2))
    (OUT / "pac_moda_v2_remaining_fixed_gate_cases.json").write_text(json.dumps({"rows": remaining}, indent=2))

    lines = ["# PAC-MoDA v2 Gain Boost Pareto", "", "All thresholds are selected on train seeds only. Deployment remains restricted to the fixed precision gates; no global override and no gate expansion in this corrected pass.", "", "## Rank-Preserve Operating Points", "", "|harmed budget|fixed|harmed|net|switches|st-only recovered|", "|---:|---:|---:|---:|---:|---:|"]
    for r in rank_oof:
        lines.append(f"|{r['harmed_budget']}|{r['fixed']}|{r['harmed']}|{r['net']}|{r['switches']}|{r['stateroll_only_recovered']}|")
    lines.extend(["", "## Best Ensemble OOF Search", "", "|harmed budget|w_bce|w_rank|w_raw|fixed|harmed|net|switches|st-only recovered|", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for r in best_oof:
        lines.append(f"|{r['harmed_budget']}|{r['w_bce']}|{r['w_rank']}|{r['w_raw']}|{r['fixed']}|{r['harmed']}|{r['net']}|{r['switches']}|{r['stateroll_only_recovered']}|")
    lines.extend(["", "## Agreement Deployment OOF", "", "|harmed budget|fixed|harmed|net|switches|st-only recovered|", "|---:|---:|---:|---:|---:|---:|"])
    for r in agree_oof:
        lines.append(f"|{r['harmed_budget']}|{r['fixed']}|{r['harmed']}|{r['net']}|{r['switches']}|{r['stateroll_only_recovered']}|")
    (OUT / "pac_moda_v2_gain_boost_pareto.md").write_text("\n".join(lines) + "\n")

    lines = ["# PAC-MoDA v2 Remaining Fixed-Gate Cases", "", "|split|seed|episode|fixed|missed|success count|raw first success rank|cal first success rank|selected rank|selected success|top2 success|top3 success|", "|---|---:|---:|---|---|---:|---:|---:|---:|---|---|---|"]
    for r in remaining:
        lines.append(f"|{r['split']}|{r['seed']}|{r['episode']}|{r['fixed']}|{r['missed']}|{r['success_count']}|{r['raw_first_success_rank']}|{r['cal_first_success_rank']}|{r['selected_rank']}|{r['selected_success']}|{r['top2_has_success']}|{r['top3_has_success']}|")
    (OUT / "pac_moda_v2_remaining_fixed_gate_cases.md").write_text("\n".join(lines) + "\n")
    (OUT / "pac_moda_v2_score_ensemble_pareto.md").write_text((OUT / "pac_moda_v2_gain_boost_pareto.md").read_text())
    (OUT / "pac_moda_v2_agreement_deployment.md").write_text((OUT / "pac_moda_v2_gain_boost_pareto.md").read_text())
    print((OUT / "pac_moda_v2_gain_boost_pareto.md").read_text())


if __name__ == "__main__":
    main()
