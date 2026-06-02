from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/data1/jingyixi/wm_runs")
ST_ACTION = ROOT / "stateroll_normalbudget_candidate_pool_s300_steps30_n100" / "proposal_data"
ST_RAW = ROOT / "stateroll_normalbudget_candidate_pool_s300_steps30_n100" / "raw_rollout_npz"
OUT = ROOT / "moda_only_action_sensitive_contrastive_20260530"
SEEDS = [42, 43, 44, 45, 46, 47]
SPLITS = {
    "splitA_train42_44_val45_47": ([42, 43, 44], [45, 46, 47]),
    "splitB_train45_47_val42_44": ([45, 46, 47], [42, 43, 44]),
}


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


def load_seed(seed: int):
    a = np.load(ST_ACTION / f"vf05_mix20_seed{seed}.npz", allow_pickle=True)
    r = np.load(ST_RAW / f"vf05_mix20_seed{seed}.npz", allow_pickle=True)
    costs = a["costs"].astype(np.float64)
    labels = a["labels"].astype(bool)
    actions = a["actions"].astype(np.float64)
    pred = r["pred"].astype(np.float64)
    goal = r["goal"].astype(np.float64)
    g = goal_for_pred(goal, pred)
    dist = np.sqrt(((pred - g) ** 2).sum(axis=-1))
    anorm = np.sqrt((actions**2).sum(axis=-1))
    rows = []
    xs = []
    E, K = costs.shape
    for ep in range(E):
        c = costs[ep]
        order = np.argsort(c, kind="stable")
        ranks = np.empty(K, dtype=int)
        ranks[order] = np.arange(K)
        sorted_c = c[order]
        cz = (c - c.mean()) / (c.std() + 1e-6)
        for j in range(K):
            feat = [
                1.0,
                float(c[j]),
                float(-c[j]),
                float(ranks[j] / max(1, K - 1)),
                float(cz[j]),
                float(c[j] - sorted_c[0]),
                float(c[j] - sorted_c[min(4, K - 1)]),
                float(sorted_c[1] - sorted_c[0]) if K > 1 else 0.0,
                float(sorted_c[min(4, K - 1)] - sorted_c[0]),
                float(c.std()),
                float(dist[ep, j, -1]),
                float(dist[ep, j].mean()),
                float(dist[ep, j].min()),
                float(dist[ep, j, 0] - dist[ep, j, -1]),
                float(pred[ep, j].mean()),
                float(pred[ep, j].std()),
                float(anorm[ep, j].mean()),
                float(anorm[ep, j].std()),
            ]
            xs.append(feat)
            rows.append({"seed": seed, "episode": ep, "rank": j, "label": bool(labels[ep, j]), "raw_rank0": j == int(order[0])})
    return np.asarray(xs, dtype=np.float32), rows


def build_dataset():
    xs, rows = [], []
    offset = 0
    groups = {}
    for seed in SEEDS:
        x, r = load_seed(seed)
        xs.append(x)
        for i, row in enumerate(r):
            row = dict(row)
            row["idx"] = offset + i
            rows.append(row)
            groups.setdefault((seed, row["episode"]), []).append(offset + i)
        offset += len(r)
    return np.concatenate(xs, axis=0), rows, groups


def build_pairs(rows: list[dict], groups: dict, train_seeds: list[int]):
    y = np.asarray([r["label"] for r in rows], dtype=bool)
    hard_pairs = []
    same_pairs = []
    for (seed, _ep), idxs0 in groups.items():
        if seed not in train_seeds:
            continue
        idxs = np.asarray(idxs0)
        pos = idxs[y[idxs]]
        neg = idxs[~y[idxs]]
        if len(pos) >= 2:
            same_pairs.extend([(int(pos[i]), int(pos[(i + 1) % len(pos)])) for i in range(len(pos))])
        if len(neg) >= 2:
            low = neg[: min(3, len(neg))]
            same_pairs.extend([(int(low[i]), int(low[(i + 1) % len(low)])) for i in range(len(low))])
        if len(pos) and len(neg):
            rank0_neg = [i for i in neg if rows[int(i)]["raw_rank0"]]
            hard = rank0_neg + neg[: min(5, len(neg))].tolist()
            for p in pos[: min(5, len(pos))]:
                for n in hard:
                    hard_pairs.append((int(p), int(n)))
    return np.asarray(same_pairs, dtype=np.int64), np.asarray(hard_pairs, dtype=np.int64)


def train_model(x: np.ndarray, rows: list[dict], groups: dict, train_seeds: list[int], alpha: float, margin: float, dim: int, epochs: int, lr: float):
    train_mask = np.asarray([r["seed"] in train_seeds for r in rows], dtype=bool)
    y = np.asarray([r["label"] for r in rows], dtype=np.float32)
    mean, std = x[train_mask].mean(axis=0), x[train_mask].std(axis=0) + 1e-6
    z_np = (x - mean) / std
    z_np[:, 0] = 1.0
    xt = torch.tensor(z_np, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    idx = torch.tensor(np.where(train_mask)[0], dtype=torch.long)
    same_pairs, hard_pairs = build_pairs(rows, groups, train_seeds)
    same = torch.tensor(same_pairs, dtype=torch.long) if len(same_pairs) else torch.zeros((0, 2), dtype=torch.long)
    hard = torch.tensor(hard_pairs, dtype=torch.long) if len(hard_pairs) else torch.zeros((0, 2), dtype=torch.long)
    g = torch.Generator().manual_seed(0)
    W = torch.randn(xt.shape[1], dim, generator=g) * 0.05
    W.requires_grad_(True)
    v = torch.zeros(dim, requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([W, v, b], lr=lr, weight_decay=1e-4)
    pos = yt[idx].sum().item()
    neg = len(idx) - pos
    pos_weight = torch.tensor(max(1.0, neg / max(pos, 1.0)), dtype=torch.float32)
    for _ in range(epochs):
        emb = xt @ W
        emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-6)
        logits = emb @ v + b
        pred_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[idx], yt[idx], pos_weight=pos_weight)
        cont = torch.zeros((), dtype=torch.float32)
        if len(same):
            d_same = ((emb[same[:, 0]] - emb[same[:, 1]]) ** 2).sum(dim=1)
            cont = cont + d_same.mean()
        if len(hard):
            d_hard = ((emb[hard[:, 0]] - emb[hard[:, 1]]) ** 2).sum(dim=1).sqrt()
            cont = cont + torch.relu(margin - d_hard).pow(2).mean()
            score_margin = logits[hard[:, 0]] - logits[hard[:, 1]]
            cont = cont + 0.5 * torch.relu(margin - score_margin).mean()
        loss = pred_loss + alpha * cont
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        emb = xt @ W
        emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-6)
        score = (emb @ v + b).numpy()
    return score


def row_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None
    pos, neg = scores[labels], scores[~labels]
    wins = 0.0
    for p in pos:
        wins += float((p > neg).sum()) + 0.5 * float((p == neg).sum())
    return float(wins / (len(pos) * len(neg)))


def evaluate(scores: np.ndarray, rows: list[dict], groups: dict, val_seeds: list[int], method: str, alpha: float, margin: float):
    y = np.asarray([r["label"] for r in rows], dtype=bool)
    top1 = top3 = top5 = top10 = oracle = near = over = denom = 0
    aucs = []
    n = 0
    for (seed, _ep), idxs0 in groups.items():
        if seed not in val_seeds:
            continue
        n += 1
        idxs = np.asarray(idxs0)
        order = idxs[np.argsort(-scores[idxs], kind="stable")]
        lab = y[order]
        top1 += int(lab[0])
        top3 += int(lab[:3].any())
        top5 += int(lab[:5].any())
        top10 += int(lab[:10].any())
        oracle += int(lab.any())
        near += int((not lab[0]) and lab.any())
        auc = row_auc(scores[idxs], y[idxs])
        if auc is not None:
            aucs.append(auc)
        raw_rank0 = next((i for i in idxs if rows[int(i)]["raw_rank0"]), idxs[0])
        if (not y[raw_rank0]) and y[idxs].any():
            denom += 1
            over += int(scores[idxs[y[idxs]]].max() > scores[raw_rank0])
    return {
        "method": method,
        "alpha": alpha,
        "margin": margin,
        "episodes": n,
        "top1_success": top1 / n * 100,
        "top3_success": top3 / n * 100,
        "top5_success": top5 / n * 100,
        "top10_success": top10 / n * 100,
        "oracle": oracle / n * 100,
        "near_miss_count": near,
        "intra_episode_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "success_over_rank0": over,
        "rank0_failure_with_success": denom,
        "success_over_rank0_rate": over / denom * 100 if denom else 0.0,
    }


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0,0.1,0.5,1.0,2.0")
    ap.add_argument("--margins", default="0.5,1.0")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    x, rows, groups = build_dataset()
    raw_scores = -x[:, 1]
    all_rows = []
    for split, (train, val) in SPLITS.items():
        all_rows.append(evaluate(raw_scores, rows, groups, val, f"{split}:raw_cost", 0.0, 0.0))
        for alpha in [float(a) for a in args.alphas.split(",") if a.strip()]:
            for margin in [float(m) for m in args.margins.split(",") if m.strip()]:
                scores = train_model(x, rows, groups, train, alpha, margin, args.dim, args.epochs, args.lr)
                r = evaluate(scores, rows, groups, val, f"{split}:contrastive", alpha, margin)
                all_rows.append(r)
                write_csv(out / "moda_only_action_sensitive_contrastive.csv", all_rows)
    agg = []
    keys = sorted({(r["method"].split(":")[-1], r["alpha"], r["margin"]) for r in all_rows})
    for meth, alpha, margin in keys:
        rs = [r for r in all_rows if r["method"].endswith(meth) and r["alpha"] == alpha and r["margin"] == margin]
        agg.append({
            "method": meth,
            "alpha": alpha,
            "margin": margin,
            "top1_success": float(np.mean([r["top1_success"] for r in rs])),
            "intra_episode_auc": float(np.mean([r["intra_episode_auc"] for r in rs])),
            "success_over_rank0_rate": float(np.mean([r["success_over_rank0_rate"] for r in rs])),
            "near_miss_count": int(sum(r["near_miss_count"] for r in rs)),
        })
    best = max(agg, key=lambda r: (r["top1_success"], r["intra_episode_auc"]))
    write_csv(out / "moda_only_action_sensitive_contrastive_aggregate.csv", agg)
    (out / "moda_only_action_sensitive_contrastive.json").write_text(json.dumps({"settings": vars(args), "rows": all_rows, "aggregate": agg, "best": best}, indent=2) + "\n")
    md = ["# MoDA-Only Action-Sensitive Contrastive Objective", "", "|method|alpha|margin|top1|intra AUC|success>rank0 %|near-miss|", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in agg:
        md.append(f"|{r['method']}|{r['alpha']}|{r['margin']}|{r['top1_success']:.2f}|{r['intra_episode_auc']:.3f}|{r['success_over_rank0_rate']:.2f}|{r['near_miss_count']}|")
    md += ["", "## Verdict", "", f"Best top1 is {best['top1_success']:.2f} with method={best['method']} alpha={best['alpha']} margin={best['margin']}."]
    (out / "moda_only_action_sensitive_contrastive.md").write_text("\n".join(md) + "\n")
    print((out / "moda_only_action_sensitive_contrastive.md").read_text())


if __name__ == "__main__":
    main()
