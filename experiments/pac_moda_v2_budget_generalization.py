from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("/data1/jingyixi/wm_runs")
OUT = ROOT / "pac_moda_v2_budget_generalization_20260529"
SEEDS = [42, 43, 44, 45, 46, 47]
SPLITS = {
    "splitA_train42_44_val45_47": ([42, 43, 44], [45, 46, 47]),
    "splitB_train45_47_val42_44": ([45, 46, 47], [42, 43, 44]),
}
POOLS = {
    "n50": (
        ROOT / "bsl_normalbudget_candidate_pool_s300_steps30_n50",
        ROOT / "stateroll_normalbudget_candidate_pool_s300_steps30_n50",
    ),
    "n100": (
        ROOT / "bsl_normalbudget_candidate_pool_s300_steps30_n100",
        ROOT / "stateroll_normalbudget_candidate_pool_s300_steps30_n100",
    ),
}


def load_src(base: Path, variant: str, seed: int) -> dict:
    a = np.load(base / "proposal_data" / f"{variant}_seed{seed}.npz", allow_pickle=True)
    r = np.load(base / "raw_rollout_npz" / f"{variant}_seed{seed}.npz", allow_pickle=True)
    return {
        "actions": a["actions"].astype(np.float64),
        "costs": a["costs"].astype(np.float64),
        "labels": a["labels"].astype(bool),
        "pred": r["pred"].astype(np.float64),
        "goal": r["goal"].astype(np.float64),
        "indices": a["indices"],
    }


def load_all(bsl_base: Path, st_base: Path) -> dict[int, dict]:
    data = {}
    for seed in SEEDS:
        b = load_src(bsl_base, "baseline", seed)
        st = load_src(st_base, "vf05_mix20", seed)
        if not np.all(b["indices"] == st["indices"]):
            raise RuntimeError(f"indices mismatch seed {seed}")
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
    return {"norm": norm_t.mean(axis=2), "std": norm_t.std(axis=2)}


def entropy_from_cost(costs: np.ndarray) -> np.ndarray:
    x = -costs.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    p = np.exp(x)
    p = p / (p.sum(axis=-1, keepdims=True) + 1e-12)
    return -(p * np.log(p + 1e-12)).sum(axis=-1)


def zscore_row(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


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
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def candidate_matrix(data: dict, seeds: list[int]):
    X, y, meta, raw = [], [], [], []
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
            bz = zscore_row(bcost)
            sz = zscore_row(scost)
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
                ("bsl", b["labels"][ep], btr, bact, bcost, bz, 0),
                ("stateroll", st["labels"][ep], strj, stact, scost, sz, 30),
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
                    raw.append(float(-costs[j]))
                    meta.append({"seed": seed, "episode": ep, "source": source, "local_rank": j, "bsl_success": bsl_success})
    return np.asarray(X), np.asarray(y, dtype=bool), meta, np.asarray(raw)


def st_episode_groups(meta, seeds):
    groups = {}
    for i, m in enumerate(meta):
        if m["seed"] in seeds and m["source"] == "stateroll":
            groups.setdefault((m["seed"], m["episode"]), []).append(i)
    return groups


def fit_legacy_rank_combined(X, y, meta, train_seeds, lr=0.02, epochs=1800, l2=1e-3, margin=0.25):
    train_mask = np.asarray([m["seed"] in train_seeds for m in meta], dtype=bool)
    mean = X[train_mask].mean(axis=0)
    std = X[train_mask].std(axis=0) + 1e-6
    Xs = (X - mean) / std
    Xb = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    w = np.zeros(Xb.shape[1])
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
    return {"w": w, "mean": mean, "std": std}


def score_model(model, X):
    Xs = (X - model["mean"]) / model["std"]
    Xb = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    return Xb @ model["w"]


def rank_metrics(meta, y, raw, scores, seeds):
    rows = []
    for score_name, score_arr in [("raw", raw), ("cal", scores)]:
        mask = np.asarray([m["seed"] in seeds and m["source"] == "stateroll" for m in meta], dtype=bool)
        auc = binary_auc(y[mask], score_arr[mask])
        by = {}
        for i in np.where(mask)[0]:
            by.setdefault((meta[i]["seed"], meta[i]["episode"]), []).append(i)
        first, top = [], {1: 0, 3: 0, 5: 0, 10: 0, 30: 0}
        near = 0
        no_success = 0
        for idxs0 in by.values():
            idxs = np.asarray(idxs0)
            labels = y[idxs]
            order = idxs[np.argsort(-score_arr[idxs], kind="stable")]
            if not labels.any():
                no_success += 1
                continue
            first.append(int(np.where(y[order])[0][0] + 1))
            if not y[order[0]]:
                near += 1
            for k in top:
                top[k] += int(y[order[:k]].any())
        n = len(by)
        arr = np.asarray(first, dtype=float)
        rows.append({
            "score": score_name,
            "auc": auc,
            "first_success_rank_mean": float(arr.mean()) if len(arr) else None,
            "first_success_rank_median": float(np.median(arr)) if len(arr) else None,
            "near_miss": near,
            "episodes_without_success": no_success,
            **{f"top{k}": top[k] / n * 100.0 for k in top},
        })
    return rows


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for name, (bsl_base, st_base) in POOLS.items():
        if not (bsl_base / "proposal_data").exists() or not (st_base / "proposal_data").exists():
            missing.append(name)
            continue
        data = load_all(bsl_base, st_base)
        X, y, meta, raw = candidate_matrix(data, SEEDS)
        for split, (train_seeds, val_seeds) in SPLITS.items():
            model = fit_legacy_rank_combined(X, y, meta, train_seeds)
            scores = score_model(model, X)
            for r in rank_metrics(meta, y, raw, scores, val_seeds):
                rows.append({"pool": name, "split": split, **r})
    write_csv(OUT / "pac_moda_v2_budget_generalization.csv", rows)
    (OUT / "pac_moda_v2_budget_generalization.json").write_text(json.dumps({"rows": rows, "missing": missing, "note": "n50 lacks matching fixed precision gate grid; this report is candidate-level calibration only."}, indent=2))
    lines = ["# PAC-MoDA v2 Budget Generalization", "", "Candidate-level calibration only for n50/n100. n50 has no matching fixed precision gate grid, so fixed-gate deployment is not compared here.", ""]
    lines.append("|pool|split|score|AUC|first rank mean|top1|top3|top5|top10|top30|near-miss|")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(f"|{r['pool']}|{r['split']}|{r['score']}|{r['auc']:.3f}|{r['first_success_rank_mean']:.2f}|{r['top1']:.1f}|{r['top3']:.1f}|{r['top5']:.1f}|{r['top10']:.1f}|{r['top30']:.1f}|{r['near_miss']}|")
    if missing:
        lines.append("")
        lines.append("Missing matching pools: " + ", ".join(missing))
    (OUT / "pac_moda_v2_budget_generalization.md").write_text("\n".join(lines) + "\n")
    print((OUT / "pac_moda_v2_budget_generalization.md").read_text())


if __name__ == "__main__":
    main()
