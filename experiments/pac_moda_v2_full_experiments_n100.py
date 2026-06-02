from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("/data1/jingyixi/wm_runs")
OUT = ROOT / "pac_moda_v2_full_n100_20260529"
BSL_ACTION = ROOT / "bsl_normalbudget_candidate_pool_s300_steps30_n100" / "proposal_data"
BSL_RAW = ROOT / "bsl_normalbudget_candidate_pool_s300_steps30_n100" / "raw_rollout_npz"
ST_ACTION = ROOT / "stateroll_normalbudget_candidate_pool_s300_steps30_n100" / "proposal_data"
ST_RAW = ROOT / "stateroll_normalbudget_candidate_pool_s300_steps30_n100" / "raw_rollout_npz"
GRID = ROOT / "gate_only_opportunity_detector_n100_20260528" / "precision_gate_grid_n100.json"
SEEDS = [42, 43, 44, 45, 46, 47]
SPLITS = {
    "splitA_train42_44_val45_47": ([42, 43, 44], [45, 46, 47]),
    "splitB_train45_47_val42_44": ([45, 46, 47], [42, 43, 44]),
}


def load_src(action_dir: Path, raw_dir: Path, variant: str, seed: int) -> dict:
    a = np.load(action_dir / f"{variant}_seed{seed}.npz", allow_pickle=True)
    r = np.load(raw_dir / f"{variant}_seed{seed}.npz", allow_pickle=True)
    return {
        "actions": a["actions"].astype(np.float64),
        "costs": a["costs"].astype(np.float64),
        "labels": a["labels"].astype(bool),
        "pred": r["pred"].astype(np.float64),
        "goal": r["goal"].astype(np.float64),
        "indices": a["indices"],
    }


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
    return {
        "final": dist[:, :, -1],
        "mean": dist.mean(axis=2),
        "min": dist.min(axis=2),
        "progress": dist[:, :, 0] - dist[:, :, -1],
        "latent_mean": pred.mean(axis=(2, 3)),
        "latent_std": pred.std(axis=(2, 3)),
    }


def action_stats(actions: np.ndarray) -> dict:
    norm_t = np.sqrt((actions**2).sum(axis=-1))
    return {
        "norm": norm_t.mean(axis=2),
        "std": norm_t.std(axis=2),
        "final_norm": norm_t[:, :, -1],
    }


def entropy_from_cost(costs: np.ndarray) -> np.ndarray:
    x = -costs.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    p = np.exp(x)
    p = p / (p.sum(axis=-1, keepdims=True) + 1e-12)
    return -(p * np.log(p + 1e-12)).sum(axis=-1)


def binary_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    i = 0
    while i < len(score):
        j = i + 1
        while j < len(score) and sorted_score[j] == sorted_score[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    sum_pos = ranks[y].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def zscore_row(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


def build_episode_features(data: dict, seed: int, ep: int) -> dict:
    b = data[seed]["b"]
    st = data[seed]["st"]
    bc = b["costs"][ep]
    sc = st["costs"][ep]
    bs = np.sort(bc)
    ss = np.sort(sc)
    btr = traj_stats(b["pred"][ep : ep + 1], b["goal"][ep : ep + 1])
    strj = traj_stats(st["pred"][ep : ep + 1], st["goal"][ep : ep + 1])
    st_best = int(np.argmin(sc))
    return {
        "bsl_margin_top2": float(bs[1] - bs[0]),
        "bsl_margin_top5": float(bs[min(4, len(bs) - 1)] - bs[0]),
        "bsl_margin_top10": float(bs[min(9, len(bs) - 1)] - bs[0]),
        "bsl_cost_std": float(bc.std()),
        "bsl_cost_entropy": float(entropy_from_cost(bc[None])[0]),
        "bsl_top1_cost": float(bc[0]),
        "st_minus_bsl_best_cost": float(ss[0] - bs[0]),
        "bsl_rank0_minus_st_best_final_dist": float(btr["final"][0, 0] - strj["final"][0, st_best]),
        "bsl_rank0_minus_st_best_mean_dist": float(btr["mean"][0, 0] - strj["mean"][0, st_best]),
        "st_best_minus_bsl_rank0_progress": float(strj["progress"][0, st_best] - btr["progress"][0, 0]),
    }


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
            ep_feat = build_episode_features(data, seed, ep)
            bcost = b["costs"][ep]
            scost = st["costs"][ep]
            union_cost = np.concatenate([bcost, scost])
            union_z = zscore_row(union_cost)
            b_z = zscore_row(bcost)
            st_z = zscore_row(scost)
            bsl_success = bool(b["labels"][ep, 0])
            bsl_oracle = bool(b["labels"][ep].any())
            st_oracle = bool(st["labels"][ep].any())
            st_only = (not bsl_success) and st_oracle and (not bsl_oracle)

            for source, arr, labels, tr, act, offset in [
                ("bsl", b, b["labels"][ep], btr, bact, 0),
                ("stateroll", st, st["labels"][ep], strj, stact, 30),
            ]:
                costs = bcost if source == "bsl" else scost
                local_z = b_z if source == "bsl" else st_z
                for j in range(costs.shape[0]):
                    uj = offset + j
                    feat = [
                        1.0 if source == "stateroll" else 0.0,
                        float(j) / max(1, costs.shape[0] - 1),
                        float(costs[j]),
                        float(local_z[j]),
                        float(union_z[uj]),
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
                        ep_feat["bsl_margin_top2"],
                        ep_feat["bsl_margin_top5"],
                        ep_feat["bsl_margin_top10"],
                        ep_feat["bsl_cost_std"],
                        ep_feat["bsl_cost_entropy"],
                        ep_feat["st_minus_bsl_best_cost"],
                        ep_feat["bsl_rank0_minus_st_best_final_dist"],
                        ep_feat["bsl_rank0_minus_st_best_mean_dist"],
                        ep_feat["st_best_minus_bsl_rank0_progress"],
                        1.0 if bsl_success else 0.0,
                        1.0 if st_only else 0.0,
                    ]
                    X.append(feat)
                    y.append(bool(labels[j]))
                    raw_score.append(float(-costs[j]))
                    meta.append(
                        {
                            "seed": seed,
                            "episode": ep,
                            "source": source,
                            "local_rank": j,
                            "union_index": uj,
                            "bsl_success": bsl_success,
                            "bsl_oracle": bsl_oracle,
                            "st_oracle": st_oracle,
                            "stateroll_only_fixable": st_only,
                        }
                    )
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=bool), meta, np.asarray(raw_score, dtype=np.float64)


def fit_logistic(X: np.ndarray, y: np.ndarray, weights: np.ndarray, lr=0.03, epochs=2500, l2=1e-3):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xs = (X - mean) / std
    Xb = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    w = np.zeros(Xb.shape[1], dtype=np.float64)
    yv = y.astype(np.float64)
    weights = weights.astype(np.float64)
    weights = weights / (weights.mean() + 1e-12)
    for _ in range(epochs):
        z = np.clip(Xb @ w, -40, 40)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = (Xb.T @ ((p - yv) * weights)) / Xb.shape[0]
        grad[:-1] += l2 * w[:-1]
        w -= lr * grad
    return {"w": w, "mean": mean, "std": std}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    Xs = (X - model["mean"]) / model["std"]
    Xb = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    z = np.clip(Xb @ model["w"], -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def linear_score(model: dict, X: np.ndarray) -> np.ndarray:
    Xs = (X - model["mean"]) / model["std"]
    Xb = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    return Xb @ model["w"]


def st_episode_groups(meta, seeds):
    groups = {}
    for i, m in enumerate(meta):
        if m["seed"] in seeds and m["source"] == "stateroll":
            groups.setdefault((m["seed"], m["episode"]), []).append(i)
    return groups


def fit_pairwise_ranker(X: np.ndarray, y: np.ndarray, meta, train_seeds, lr=0.02, epochs=1200, l2=1e-3, margin=0.25):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xs = (X - mean) / std
    Xb = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    w = np.zeros(Xb.shape[1], dtype=np.float64)
    pairs = []
    for idxs0 in st_episode_groups(meta, train_seeds).values():
        idxs = np.asarray(idxs0, dtype=np.int64)
        pos = idxs[y[idxs]]
        neg = idxs[~y[idxs]]
        if len(pos) == 0 or len(neg) == 0:
            continue
        hard_negs = sorted(neg.tolist(), key=lambda i: (meta[i]["local_rank"], X[i, 2]))[: min(8, len(neg))]
        for pi in pos[: min(6, len(pos))]:
            for ni in hard_negs:
                pairs.append((int(pi), int(ni)))
    if not pairs:
        return {"w": w, "mean": mean, "std": std, "pairs": 0}
    pairs = np.asarray(pairs, dtype=np.int64)
    for _ in range(epochs):
        diff = Xb[pairs[:, 0]] - Xb[pairs[:, 1]]
        z = np.clip(diff @ w - margin, -40, 40)
        coeff = -1.0 / (1.0 + np.exp(z))
        grad = (diff.T @ coeff) / diff.shape[0]
        grad[:-1] += l2 * w[:-1]
        w -= lr * grad
    return {"w": w, "mean": mean, "std": std, "pairs": int(len(pairs))}


def fit_listwise_ranker(X: np.ndarray, y: np.ndarray, meta, train_seeds, lr=0.03, epochs=1600, l2=1e-3):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xs = (X - mean) / std
    Xb = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    w = np.zeros(Xb.shape[1], dtype=np.float64)
    groups = []
    for idxs0 in st_episode_groups(meta, train_seeds).values():
        idxs = np.asarray(idxs0, dtype=np.int64)
        if y[idxs].any():
            groups.append(idxs)
    for _ in range(epochs):
        grad = np.zeros_like(w)
        for idxs in groups:
            z = np.clip(Xb[idxs] @ w, -40, 40)
            z = z - z.max()
            p = np.exp(z)
            p = p / (p.sum() + 1e-12)
            t = y[idxs].astype(np.float64)
            t = t / (t.sum() + 1e-12)
            grad += Xb[idxs].T @ (p - t)
        grad /= max(1, len(groups))
        grad[:-1] += l2 * w[:-1]
        w -= lr * grad
    return {"w": w, "mean": mean, "std": std, "groups": int(len(groups))}


def fit_combined_ranker(X: np.ndarray, y: np.ndarray, meta, train_seeds, lr=0.02, epochs=1800, l2=1e-3, margin=0.25):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xs = (X - mean) / std
    Xb = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    w = np.zeros(Xb.shape[1], dtype=np.float64)
    groups, pairs, preserve = [], [], []
    for idxs0 in st_episode_groups(meta, train_seeds).values():
        idxs = np.asarray(idxs0, dtype=np.int64)
        pos = idxs[y[idxs]]
        neg = idxs[~y[idxs]]
        if len(pos):
            groups.append(idxs)
        if len(pos) and len(neg):
            hard_negs = sorted(neg.tolist(), key=lambda i: (meta[i]["local_rank"], X[i, 2]))[: min(8, len(neg))]
            for pi in pos[: min(6, len(pos))]:
                for ni in hard_negs:
                    pairs.append((int(pi), int(ni)))
        if meta[idxs[0]]["bsl_success"]:
            preserve.extend([int(i) for i in sorted(neg.tolist(), key=lambda j: (meta[j]["local_rank"], X[j, 2]))[: min(10, len(neg))]])
    pairs = np.asarray(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64)
    preserve = np.asarray(preserve, dtype=np.int64) if preserve else np.zeros(0, dtype=np.int64)
    for _ in range(epochs):
        grad = np.zeros_like(w)
        if len(pairs):
            diff = Xb[pairs[:, 0]] - Xb[pairs[:, 1]]
            z = np.clip(diff @ w - margin, -40, 40)
            coeff = -1.0 / (1.0 + np.exp(z))
            grad += 0.7 * (diff.T @ coeff) / diff.shape[0]
        if groups:
            g2 = np.zeros_like(w)
            for idxs in groups:
                z = np.clip(Xb[idxs] @ w, -40, 40)
                z = z - z.max()
                p = np.exp(z)
                p = p / (p.sum() + 1e-12)
                t = y[idxs].astype(np.float64)
                t = t / (t.sum() + 1e-12)
                g2 += Xb[idxs].T @ (p - t)
            grad += 0.3 * g2 / len(groups)
        if len(preserve):
            z = np.clip(Xb[preserve] @ w, -40, 40)
            p = 1.0 / (1.0 + np.exp(-z))
            grad += 0.15 * (Xb[preserve].T @ p) / len(preserve)
        grad[:-1] += l2 * w[:-1]
        w -= lr * grad
    return {"w": w, "mean": mean, "std": std, "pairs": int(len(pairs)), "groups": int(len(groups)), "preserve": int(len(preserve))}


def make_training_structures(X: np.ndarray, y: np.ndarray, meta, train_seeds):
    groups, pairs, preserve = [], [], []
    for idxs0 in st_episode_groups(meta, train_seeds).values():
        idxs = np.asarray(idxs0, dtype=np.int64)
        pos = idxs[y[idxs]]
        neg = idxs[~y[idxs]]
        if len(pos):
            groups.append(idxs)
        if len(pos) and len(neg):
            hard_negs = sorted(neg.tolist(), key=lambda i: (meta[i]["local_rank"], X[i, 2]))[: min(8, len(neg))]
            for pi in pos[: min(6, len(pos))]:
                for ni in hard_negs:
                    pairs.append((int(pi), int(ni)))
        if meta[idxs[0]]["bsl_success"] and len(neg):
            preserve.extend([int(i) for i in sorted(neg.tolist(), key=lambda j: (meta[j]["local_rank"], X[j, 2]))[: min(10, len(neg))]])
    return {
        "groups": groups,
        "pairs": np.asarray(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64),
        "preserve": np.asarray(preserve, dtype=np.int64) if preserve else np.zeros(0, dtype=np.int64),
    }


def fit_pac_objective(
    X: np.ndarray,
    y: np.ndarray,
    meta,
    train_seeds,
    use_pairwise=False,
    use_listwise=False,
    use_preserve=False,
    lr=0.025,
    epochs=1800,
    l2=1e-3,
    margin=0.25,
):
    train_mask = np.asarray([m["seed"] in train_seeds for m in meta], dtype=bool)
    mean = X[train_mask].mean(axis=0)
    std = X[train_mask].std(axis=0) + 1e-6
    Xs = (X - mean) / std
    Xb = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    w = np.zeros(Xb.shape[1], dtype=np.float64)
    train_idx = np.where(train_mask)[0]
    yv = y[train_idx].astype(np.float64)
    pos = yv.sum()
    neg = len(yv) - pos
    weights = np.ones(len(train_idx), dtype=np.float64)
    weights[yv.astype(bool)] = max(1.0, neg / max(1.0, pos))
    for k, i in enumerate(train_idx):
        m = meta[int(i)]
        if m["source"] == "stateroll" and m["stateroll_only_fixable"] and y[int(i)]:
            weights[k] *= 3.0
    weights = weights / (weights.mean() + 1e-12)
    structs = make_training_structures(X, y, meta, train_seeds)
    pairs = structs["pairs"]
    groups = structs["groups"]
    preserve = structs["preserve"]
    for _ in range(epochs):
        grad = np.zeros_like(w)
        z = np.clip(Xb[train_idx] @ w, -40, 40)
        p = 1.0 / (1.0 + np.exp(-z))
        grad += (Xb[train_idx].T @ ((p - yv) * weights)) / len(train_idx)
        if use_pairwise and len(pairs):
            diff = Xb[pairs[:, 0]] - Xb[pairs[:, 1]]
            zpair = np.clip(diff @ w - margin, -40, 40)
            coeff = -1.0 / (1.0 + np.exp(zpair))
            grad += 0.6 * (diff.T @ coeff) / diff.shape[0]
        if use_listwise and groups:
            g2 = np.zeros_like(w)
            for idxs in groups:
                zl = np.clip(Xb[idxs] @ w, -40, 40)
                zl = zl - zl.max()
                pp = np.exp(zl)
                pp = pp / (pp.sum() + 1e-12)
                target = y[idxs].astype(np.float64)
                target = target / (target.sum() + 1e-12)
                g2 += Xb[idxs].T @ (pp - target)
            grad += 0.35 * g2 / len(groups)
        if use_preserve and len(preserve):
            zp = np.clip(Xb[preserve] @ w, -40, 40)
            pp = 1.0 / (1.0 + np.exp(-zp))
            grad += 0.25 * (Xb[preserve].T @ pp) / len(preserve)
        grad[:-1] += l2 * w[:-1]
        w -= lr * grad
    return {
        "w": w,
        "mean": mean,
        "std": std,
        "pairs": int(len(pairs)),
        "groups": int(len(groups)),
        "preserve": int(len(preserve)),
        "use_pairwise": bool(use_pairwise),
        "use_listwise": bool(use_listwise),
        "use_preserve": bool(use_preserve),
    }


def rank_metrics(scores_by_ep: dict, labels_by_ep: dict, max_k: int):
    first_ranks = []
    no_success = 0
    top_hits = {1: 0, 3: 0, 5: 0, 10: 0, 30: 0}
    near_miss = 0
    for key, scores in scores_by_ep.items():
        labels = labels_by_ep[key]
        order = np.argsort(-scores, kind="stable")
        if not labels.any():
            no_success += 1
            continue
        pos = np.where(labels[order])[0]
        first_ranks.append(int(pos[0] + 1))
        if not labels[order[0]]:
            near_miss += 1
        for k in top_hits:
            kk = min(k, max_k)
            if labels[order[:kk]].any():
                top_hits[k] += 1
    n = len(scores_by_ep)
    arr = np.asarray(first_ranks, dtype=np.float64)
    return {
        "episodes": n,
        "episodes_with_success": int(n - no_success),
        "episodes_without_success": int(no_success),
        "first_success_rank_mean": float(arr.mean()) if arr.size else None,
        "first_success_rank_median": float(np.median(arr)) if arr.size else None,
        "near_miss_failure_count": int(near_miss),
        **{f"top{k}_success_recall": float(top_hits[k] / n * 100.0) for k in top_hits},
    }


def evaluate_candidate_scores(X, y, meta, scores, raw_scores, seeds):
    rows = []
    for pool in ["bsl", "stateroll", "union"]:
        mask_seed = np.asarray([m["seed"] in seeds for m in meta], dtype=bool)
        if pool != "union":
            mask_pool = np.asarray([m["source"] == pool for m in meta], dtype=bool)
        else:
            mask_pool = np.ones(len(meta), dtype=bool)
        mask = mask_seed & mask_pool
        auc_cal = binary_auc(y[mask].astype(int), scores[mask])
        auc_raw = binary_auc(y[mask].astype(int), raw_scores[mask])

        scores_by_ep, raw_by_ep, labels_by_ep = {}, {}, {}
        for i in np.where(mask)[0]:
            m = meta[i]
            key = (m["seed"], m["episode"])
            scores_by_ep.setdefault(key, []).append(scores[i])
            raw_by_ep.setdefault(key, []).append(raw_scores[i])
            labels_by_ep.setdefault(key, []).append(y[i])
        scores_by_ep = {k: np.asarray(v, dtype=np.float64) for k, v in scores_by_ep.items()}
        raw_by_ep = {k: np.asarray(v, dtype=np.float64) for k, v in raw_by_ep.items()}
        labels_by_ep = {k: np.asarray(v, dtype=bool) for k, v in labels_by_ep.items()}
        rm_cal = rank_metrics(scores_by_ep, labels_by_ep, 60 if pool == "union" else 30)
        rm_raw = rank_metrics(raw_by_ep, labels_by_ep, 60 if pool == "union" else 30)
        row = {"pool": pool, "candidate_auc_raw": auc_raw, "candidate_auc_calibrated": auc_cal}
        for k, v in rm_raw.items():
            row[f"raw_{k}"] = v
        for k, v in rm_cal.items():
            row[f"cal_{k}"] = v
        rows.append(row)
    return rows


def choose_threshold(meta, y, scores, train_seeds):
    # Threshold on best stateroll calibrated score. Train-only labels decide the threshold.
    by_ep = defaultdict_episode(meta, y, scores, train_seeds)
    max_scores = np.asarray([v["best_score"] for v in by_ep.values()], dtype=np.float64)
    thresholds = np.unique(np.quantile(max_scores, np.linspace(0, 0.95, 20)).round(6))
    rows = []
    for thr in thresholds:
        fixed = harmed = switches = 0
        for v in by_ep.values():
            if v["best_score"] <= thr:
                continue
            switches += 1
            succ = v["best_success"]
            if (not v["bsl_success"]) and succ:
                fixed += 1
            if v["bsl_success"] and (not succ):
                harmed += 1
        rows.append({"threshold": float(thr), "fixed": fixed, "harmed": harmed, "switches": switches})
    safe = [r for r in rows if r["harmed"] <= 1 and r["switches"] <= 20]
    cand = safe if safe else rows
    cand.sort(key=lambda r: (r["fixed"] - 3 * r["harmed"], -r["harmed"], -r["switches"], r["threshold"]), reverse=True)
    return cand[0], rows


def defaultdict_episode(meta, y, scores, seeds):
    out = {}
    for i, m in enumerate(meta):
        if m["seed"] not in seeds or m["source"] != "stateroll":
            continue
        key = (m["seed"], m["episode"])
        v = out.setdefault(
            key,
            {
                "best_score": -1e30,
                "best_success": False,
                "best_rank": -1,
                "bsl_success": m["bsl_success"],
                "stateroll_only_fixable": m["stateroll_only_fixable"],
            },
        )
        if scores[i] > v["best_score"]:
            v["best_score"] = float(scores[i])
            v["best_success"] = bool(y[i])
            v["best_rank"] = int(m["local_rank"])
    return out


def fixed_gate_selected(split: str):
    grid = json.loads(GRID.read_text())["records"]
    if split.startswith("splitA"):
        target = ("extratrees", "top10+st_gap_bottom20+AND")
    else:
        target = ("randomforest", "top10+abs_gap_top10+AND")
    for r in grid:
        if r["split"] == split and r["model"] == target[0] and r["rule"] == target[1]:
            return {(int(e["seed"]), int(e["episode"])) for e in r["selected_episodes"]}, r
    raise RuntimeError(f"missing fixed gate row for {split}")


def eval_final_selection(split: str, meta, y, scores, val_seeds, threshold):
    gate_set, gate_row = fixed_gate_selected(split)
    by_ep = defaultdict_episode(meta, y, scores, val_seeds)
    fixed = harmed = switches = st_only_recovered = 0
    bsl_success_count = 0
    total_eps = 0
    selected_rows = []
    for key, v in by_ep.items():
        total_eps += 1
        if v["bsl_success"]:
            bsl_success_count += 1
        if key not in gate_set:
            continue
        if v["best_score"] <= threshold:
            continue
        switches += 1
        succ = v["best_success"]
        if (not v["bsl_success"]) and succ:
            fixed += 1
        if v["bsl_success"] and (not succ):
            harmed += 1
        if v["stateroll_only_fixable"] and succ:
            st_only_recovered += 1
        selected_rows.append(
            {
                "seed": key[0],
                "episode": key[1],
                "best_score": v["best_score"],
                "best_rank": v["best_rank"],
                "best_success": succ,
                "bsl_success": v["bsl_success"],
                "stateroll_only_fixable": v["stateroll_only_fixable"],
            }
        )
    return {
        "split": split,
        "gate_rule": gate_row["rule"],
        "gate_model": gate_row["model"],
        "gate_selected": len(gate_set),
        "gate_st_only": int(gate_row["stateroll_only_captured"]),
        "gate_bsl_fp": int(gate_row["bsl_success_false_positive"]),
        "threshold": float(threshold),
        "episodes": int(total_eps),
        "bsl_top1": float(bsl_success_count / total_eps * 100.0) if total_eps else 0.0,
        "calibrated_selector_top1": float((bsl_success_count + fixed - harmed) / total_eps * 100.0) if total_eps else 0.0,
        "fixed": int(fixed),
        "harmed": int(harmed),
        "switches": int(switches),
        "stateroll_only_recovered": int(st_only_recovered),
        "selected_rows": selected_rows,
    }


def eval_selection_on_gate_name(split: str, meta, y, scores, val_seeds, threshold):
    return eval_final_selection(split, meta, y, scores, val_seeds, threshold)


def threshold_curve(split: str, meta, y, scores, val_seeds):
    gate_set, _ = fixed_gate_selected(split)
    by_ep = defaultdict_episode(meta, y, scores, val_seeds)
    vals = [v["best_score"] for k, v in by_ep.items() if k in gate_set]
    if not vals:
        return []
    thresholds = np.unique(np.quantile(np.asarray(vals), np.linspace(0, 1, 51)).round(6))
    rows = []
    for thr in thresholds:
        fixed = harmed = switches = st_only = 0
        for key, v in by_ep.items():
            if key not in gate_set or v["best_score"] <= thr:
                continue
            switches += 1
            succ = v["best_success"]
            if (not v["bsl_success"]) and succ:
                fixed += 1
            if v["bsl_success"] and (not succ):
                harmed += 1
            if v["stateroll_only_fixable"] and succ:
                st_only += 1
        rows.append({"split": split, "threshold": float(thr), "fixed": fixed, "harmed": harmed, "switches": switches, "stateroll_only_recovered": st_only})
    return rows


def extract_case_rows(split: str, meta, y, scores, raw_scores, val_seeds, threshold):
    gate_set, _ = fixed_gate_selected(split)
    by = {}
    for i, m in enumerate(meta):
        if m["seed"] not in val_seeds or m["source"] != "stateroll":
            continue
        key = (m["seed"], m["episode"])
        by.setdefault(key, []).append(i)
    rows = []
    for key, idxs in by.items():
        if key not in gate_set:
            continue
        idxs = np.asarray(idxs, dtype=np.int64)
        labels = y[idxs]
        cal_order = idxs[np.argsort(-scores[idxs], kind="stable")]
        raw_order = idxs[np.argsort(-raw_scores[idxs], kind="stable")]
        bsl_success = meta[idxs[0]]["bsl_success"]
        st_only = meta[idxs[0]]["stateroll_only_fixable"]
        has_success = bool(labels.any())
        raw_first = None
        cal_first = None
        if has_success:
            raw_first = int(np.where(y[raw_order])[0][0] + 1)
            cal_first = int(np.where(y[cal_order])[0][0] + 1)
        selected = cal_order[0]
        raw_selected = raw_order[0]
        switched = bool(scores[selected] > threshold)
        fixed = bool(switched and (not bsl_success) and y[selected])
        harmed = bool(switched and bsl_success and (not y[selected]))
        missed = bool((not bsl_success) and has_success and (not fixed))
        rows.append(
            {
                "split": split,
                "seed": key[0],
                "episode": key[1],
                "bsl_success": bool(bsl_success),
                "stateroll_only_fixable": bool(st_only),
                "stateroll_success_count": int(labels.sum()),
                "raw_first_success_rank": raw_first,
                "cal_first_success_rank": cal_first,
                "raw_selected_rank": int(meta[raw_selected]["local_rank"]),
                "raw_selected_success": bool(y[raw_selected]),
                "cal_selected_rank": int(meta[selected]["local_rank"]),
                "cal_selected_success": bool(y[selected]),
                "raw_selected_cost_score": float(raw_scores[raw_selected]),
                "cal_selected_score": float(scores[selected]),
                "threshold": float(threshold),
                "switched": switched,
                "fixed": fixed,
                "harmed": harmed,
                "remaining_missed_fixable": missed,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys and not isinstance(r[k], list):
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def train_eval_method(method, X, y, meta, raw_scores, train_seeds, val_seeds, split):
    if method == "legacy_rank_combined":
        model = fit_combined_ranker(X, y, meta, train_seeds)
        scores = linear_score(model, X)
        info = {k: v for k, v in model.items() if k not in {"w", "mean", "std"}}
    else:
        flags = {
            "bce": (False, False, False),
            "bce_pairwise": (True, False, False),
            "bce_listwise": (False, True, False),
            "bce_preserve": (False, False, True),
            "bce_pairwise_preserve": (True, False, True),
            "bce_listwise_preserve": (False, True, True),
            "bce_pairwise_listwise": (True, True, False),
            "full_bce_pairwise_listwise_preserve": (True, True, True),
        }[method]
        model = fit_pac_objective(X, y, meta, train_seeds, use_pairwise=flags[0], use_listwise=flags[1], use_preserve=flags[2])
        scores = linear_score(model, X)
        info = {k: v for k, v in model.items() if k not in {"w", "mean", "std"}}
    train_metrics = evaluate_candidate_scores(X, y, meta, scores, raw_scores, train_seeds)
    val_metrics = evaluate_candidate_scores(X, y, meta, scores, raw_scores, val_seeds)
    thr_best, thr_rows = choose_threshold(meta, y, scores, train_seeds)
    final = eval_final_selection(split, meta, y, scores, val_seeds, thr_best["threshold"])
    final["method"] = method
    final["train_threshold_fixed"] = thr_best["fixed"]
    final["train_threshold_harmed"] = thr_best["harmed"]
    final["train_threshold_switches"] = thr_best["switches"]
    return {
        "method": method,
        "scores": scores,
        "model_info": info,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "threshold_choice": thr_best,
        "threshold_rows": thr_rows,
        "final": final,
    }


def stateroll_row(metrics):
    return next(r for r in metrics if r["pool"] == "stateroll")


def method_total(deploy_rows, method):
    rows = [r for r in deploy_rows if r["method"] == method]
    return {
        "method": method,
        "fixed": sum(int(r["fixed"]) for r in rows),
        "harmed": sum(int(r["harmed"]) for r in rows),
        "switches": sum(int(r["switches"]) for r in rows),
        "stateroll_only_recovered": sum(int(r["stateroll_only_recovered"]) for r in rows),
    }


def write_ablation_report(records, deployment_rows):
    lines = ["# PAC-MoDA v2 Ablation n100", ""]
    lines.append("No encoder/predictor/world-model retraining. Candidate pools are frozen. Thresholds are selected on train seeds only.")
    lines.append("")
    lines.append("## Stateroll Candidate-Level Metrics")
    lines.append("")
    lines.append("|split|method|raw AUC|cal AUC|raw first rank|cal first rank|raw top1|cal top1|raw top3|cal top3|raw top5|cal top5|raw top10|cal top10|near-miss raw|near-miss cal|")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rec in records:
        row = stateroll_row(rec["val_metrics"])
        lines.append(
            f"|{rec['split']}|{rec['method']}|{row['candidate_auc_raw']:.3f}|{row['candidate_auc_calibrated']:.3f}|"
            f"{row['raw_first_success_rank_mean']:.2f}|{row['cal_first_success_rank_mean']:.2f}|"
            f"{row['raw_top1_success_recall']:.1f}|{row['cal_top1_success_recall']:.1f}|"
            f"{row['raw_top3_success_recall']:.1f}|{row['cal_top3_success_recall']:.1f}|"
            f"{row['raw_top5_success_recall']:.1f}|{row['cal_top5_success_recall']:.1f}|"
            f"{row['raw_top10_success_recall']:.1f}|{row['cal_top10_success_recall']:.1f}|"
            f"{row['raw_near_miss_failure_count']}|{row['cal_near_miss_failure_count']}|"
        )
    lines.append("")
    lines.append("## Fixed-Gate Deployment")
    lines.append("")
    lines.append("|split|method|bsl top1|selector top1|fixed|harmed|switches|st-only recovered|threshold|")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in deployment_rows:
        lines.append(f"|{r['split']}|{r['method']}|{r['bsl_top1']:.1f}|{r['calibrated_selector_top1']:.1f}|{r['fixed']}|{r['harmed']}|{r['switches']}|{r['stateroll_only_recovered']}|{r['threshold']:.4f}|")
    lines.append("")
    lines.append("## OOF Totals")
    lines.append("")
    lines.append("|method|fixed|harmed|switches|st-only recovered|")
    lines.append("|---|---:|---:|---:|---:|")
    for method in sorted({r["method"] for r in deployment_rows}):
        t = method_total(deployment_rows, method)
        lines.append(f"|{method}|{t['fixed']}|{t['harmed']}|{t['switches']}|{t['stateroll_only_recovered']}|")
    (OUT / "pac_moda_v2_ablation_n100.md").write_text("\n".join(lines) + "\n")


def make_plots(ablation_records, deployment_rows, threshold_rows, case_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)
    methods = sorted({r["method"] for r in ablation_records})
    auc_vals, rank_vals = [], []
    for m in methods:
        rows = [stateroll_row(r["val_metrics"]) for r in ablation_records if r["method"] == m]
        auc_vals.append(np.mean([x["candidate_auc_calibrated"] for x in rows]))
        rank_vals.append(np.mean([x["cal_first_success_rank_mean"] for x in rows]))
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(methods)), auc_vals)
    plt.xticks(range(len(methods)), methods, rotation=35, ha="right", fontsize=8)
    plt.ylabel("Stateroll candidate AUC")
    plt.tight_layout()
    plt.savefig(figdir / "pac_moda_v2_auc_bar.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(methods)), rank_vals)
    plt.xticks(range(len(methods)), methods, rotation=35, ha="right", fontsize=8)
    plt.ylabel("First-success rank")
    plt.tight_layout()
    plt.savefig(figdir / "pac_moda_v2_first_success_rank_bar.png", dpi=180)
    plt.close()

    totals = [method_total(deployment_rows, m) for m in methods]
    x = np.arange(len(methods))
    plt.figure(figsize=(10, 4))
    plt.bar(x - 0.18, [t["fixed"] for t in totals], width=0.36, label="fixed")
    plt.bar(x + 0.18, [t["harmed"] for t in totals], width=0.36, label="harmed")
    plt.xticks(x, methods, rotation=35, ha="right", fontsize=8)
    plt.ylabel("Episodes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "pac_moda_v2_fixed_harmed_bar.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    for m in methods:
        rows = [stateroll_row(r["val_metrics"]) for r in ablation_records if r["method"] == m]
        vals = [np.mean([row[f"cal_top{k}_success_recall"] for row in rows]) for k in [1, 3, 5, 10, 30]]
        plt.plot([1, 3, 5, 10, 30], vals, marker="o", label=m)
    plt.xlabel("Top-k")
    plt.ylabel("Success recall")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(figdir / "pac_moda_v2_topk_recall_curve.png", dpi=180)
    plt.close()

    full_thr = [r for r in threshold_rows if r.get("method") == "legacy_rank_combined"]
    if full_thr:
        plt.figure(figsize=(6, 4))
        plt.scatter([r["harmed"] for r in full_thr], [r["fixed"] for r in full_thr], s=18)
        plt.xlabel("Harmed")
        plt.ylabel("Fixed")
        plt.tight_layout()
        plt.savefig(figdir / "pareto_fixed_harmed.png", dpi=180)
        plt.close()

    fixed_cases = [r for r in case_rows if r["fixed"]]
    if fixed_cases:
        labels = [f"{r['seed']}-{r['episode']}" for r in fixed_cases]
        x = np.arange(len(fixed_cases))
        plt.figure(figsize=(8, 4))
        plt.bar(x - 0.18, [r["raw_first_success_rank"] or 0 for r in fixed_cases], width=0.36, label="raw")
        plt.bar(x + 0.18, [r["cal_first_success_rank"] or 0 for r in fixed_cases], width=0.36, label="cal")
        plt.xticks(x, labels, rotation=35, ha="right")
        plt.ylabel("First-success rank")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figdir / "pac_moda_v2_case_rank_shift.png", dpi=180)
        plt.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_all()
    X, y, meta, raw_scores = candidate_matrix(data, SEEDS)
    methods = [
        "bce",
        "bce_pairwise",
        "bce_listwise",
        "bce_preserve",
        "bce_pairwise_preserve",
        "bce_listwise_preserve",
        "bce_pairwise_listwise",
        "full_bce_pairwise_listwise_preserve",
        "legacy_rank_combined",
    ]

    ablation_records, ablation_metric_rows, deployment_rows, threshold_rows = [], [], [], []
    case_rows = []
    for split, (train_seeds, val_seeds) in SPLITS.items():
        for method in methods:
            rec = train_eval_method(method, X, y, meta, raw_scores, train_seeds, val_seeds, split)
            rec["split"] = split
            ablation_records.append({k: v for k, v in rec.items() if k != "scores"})
            for row in rec["val_metrics"]:
                ablation_metric_rows.append({"split": split, "method": method, **row})
            deployment_rows.append({k: v for k, v in rec["final"].items() if k != "selected_rows"})
            threshold_rows.extend([{"split": split, "method": method, **r} for r in rec["threshold_rows"]])
            if method == "legacy_rank_combined":
                threshold_rows.extend([{"split": split, "method": method, "curve": "val_fixed_gate", **r} for r in threshold_curve(split, meta, y, rec["scores"], val_seeds)])
                case_rows.extend(extract_case_rows(split, meta, y, rec["scores"], raw_scores, val_seeds, rec["threshold_choice"]["threshold"]))

    loso_rows, loso_records = [], []
    for held in SEEDS:
        train_seeds = [s for s in SEEDS if s != held]
        split = "splitB_train45_47_val42_44" if held in [42, 43, 44] else "splitA_train42_44_val45_47"
        rec = train_eval_method("legacy_rank_combined", X, y, meta, raw_scores, train_seeds, [held], split)
        st = stateroll_row(rec["val_metrics"])
        f = rec["final"]
        row = {
            "held_seed": held,
            "split_gate_used": split,
            "raw_auc": st["candidate_auc_raw"],
            "cal_auc": st["candidate_auc_calibrated"],
            "raw_first_success_rank_mean": st["raw_first_success_rank_mean"],
            "cal_first_success_rank_mean": st["cal_first_success_rank_mean"],
            "raw_top1": st["raw_top1_success_recall"],
            "cal_top1": st["cal_top1_success_recall"],
            "raw_top10": st["raw_top10_success_recall"],
            "cal_top10": st["cal_top10_success_recall"],
            "raw_near_miss": st["raw_near_miss_failure_count"],
            "cal_near_miss": st["cal_near_miss_failure_count"],
            "bsl_top1": f["bsl_top1"],
            "selector_top1": f["calibrated_selector_top1"],
            "fixed": f["fixed"],
            "harmed": f["harmed"],
            "switches": f["switches"],
            "stateroll_only_recovered": f["stateroll_only_recovered"],
            "threshold": f["threshold"],
        }
        loso_rows.append(row)
        loso_records.append({k: v for k, v in rec.items() if k != "scores"})

    write_csv(OUT / "pac_moda_v2_ablation_n100.csv", ablation_metric_rows)
    write_csv(OUT / "pac_moda_v2_ablation_deployment_n100.csv", deployment_rows)
    write_csv(OUT / "pac_moda_v2_threshold_robustness_n100.csv", threshold_rows)
    write_csv(OUT / "pac_moda_v2_loso_n100.csv", loso_rows)
    write_csv(OUT / "pac_moda_v2_fixed_case_studies.csv", case_rows)
    (OUT / "pac_moda_v2_ablation_n100.json").write_text(json.dumps({"records": ablation_records}, indent=2))
    (OUT / "pac_moda_v2_loso_n100.json").write_text(json.dumps({"records": loso_records, "rows": loso_rows}, indent=2))
    (OUT / "pac_moda_v2_threshold_robustness_n100.json").write_text(json.dumps({"rows": threshold_rows}, indent=2))
    (OUT / "pac_moda_v2_fixed_case_studies.json").write_text(json.dumps({"rows": case_rows}, indent=2))

    write_ablation_report(ablation_records, deployment_rows)

    lines = ["# PAC-MoDA v2 LOSO n100", "", "|held seed|raw AUC|cal AUC|raw first rank|cal first rank|bsl top1|selector top1|fixed|harmed|switches|st-only recovered|", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in loso_rows:
        lines.append(f"|{r['held_seed']}|{r['raw_auc']:.3f}|{r['cal_auc']:.3f}|{r['raw_first_success_rank_mean']:.2f}|{r['cal_first_success_rank_mean']:.2f}|{r['bsl_top1']:.1f}|{r['selector_top1']:.1f}|{r['fixed']}|{r['harmed']}|{r['switches']}|{r['stateroll_only_recovered']}|")
    lines.append("")
    lines.append(f"Total fixed={sum(r['fixed'] for r in loso_rows)}, harmed={sum(r['harmed'] for r in loso_rows)}, switches={sum(r['switches'] for r in loso_rows)}.")
    (OUT / "pac_moda_v2_loso_n100.md").write_text("\n".join(lines) + "\n")

    lines = ["# PAC-MoDA v2 Threshold Robustness n100", "", "Rows in CSV include the train-threshold grid for all methods and the validation fixed-gate threshold curve for legacy_rank_combined.", ""]
    for harmed_cap in [0, 1, 2]:
        cand = [r for r in threshold_rows if r.get("curve") == "val_fixed_gate" and r["harmed"] <= harmed_cap]
        if cand:
            best = max(cand, key=lambda r: (r["fixed"], -r["switches"]))
            lines.append(f"- harmed <= {harmed_cap}: best fixed={best['fixed']}, harmed={best['harmed']}, switches={best['switches']}, split={best['split']}, threshold={best['threshold']:.4f}")
    (OUT / "pac_moda_v2_threshold_robustness_n100.md").write_text("\n".join(lines) + "\n")

    fixed_cases = [r for r in case_rows if r["fixed"]]
    missed_cases = [r for r in case_rows if r["remaining_missed_fixable"]]
    lines = ["# PAC-MoDA v2 Fixed Case Studies n100", "", f"Fixed cases: {len(fixed_cases)}. Remaining missed fixable cases inside fixed gates: {len(missed_cases)}.", ""]
    lines.append("|split|seed|episode|st-only|success count|raw first success rank|cal first success rank|cal selected rank|cal score|")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for r in fixed_cases:
        lines.append(f"|{r['split']}|{r['seed']}|{r['episode']}|{r['stateroll_only_fixable']}|{r['stateroll_success_count']}|{r['raw_first_success_rank']}|{r['cal_first_success_rank']}|{r['cal_selected_rank']}|{r['cal_selected_score']:.4f}|")
    (OUT / "pac_moda_v2_fixed_case_studies.md").write_text("\n".join(lines) + "\n")

    make_plots(ablation_records, deployment_rows, threshold_rows, case_rows)
    print((OUT / "pac_moda_v2_ablation_n100.md").read_text())


if __name__ == "__main__":
    main()
