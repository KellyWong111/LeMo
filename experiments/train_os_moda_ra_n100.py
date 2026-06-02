from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import stable_worldmodel as swm
import torch
import torch.nn as nn
import torch.nn.functional as F

WM = Path("/data1/jingyixi/wm_runs")
REPO = Path("/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean")
sys.path.insert(0, str(WM))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "wm_experiment_scripts"))

import train_pool_aware_planning_alignment_n100 as base
from wm_experiment_scripts.pool_coverage_compare_variants import POLICIES


def entropy_from_cost(costs):
    x = -costs
    x = x - x.max(axis=-1, keepdims=True)
    p = np.exp(x)
    p = p / (p.sum(axis=-1, keepdims=True) + 1e-8)
    return -(p * np.log(p + 1e-8)).sum(axis=-1)


def episode_uncertainty(d, ep):
    b = np.sort(d["bsl_costs"][ep])
    st = np.sort(d["st_costs"][ep])
    bc = d["bsl_costs"][ep]
    sc = d["st_costs"][ep]
    return np.asarray(
        [
            b[1] - b[0],
            b[4] - b[0],
            b[9] - b[0],
            bc.std(),
            entropy_from_cost(bc[None])[0],
            st[0] - b[0],
            st[:5].mean() - b[:5].mean(),
            sc.std() - bc.std(),
        ],
        dtype=np.float32,
    )


def scalar_features(d, seed, ep, cand):
    costs = d["costs"][ep]
    labels = d["labels"][ep]
    source = 0.0 if cand < 30 else 1.0
    rank = cand if cand < 30 else cand - 30
    b0 = costs[0]
    sorted_cost = np.sort(costs)
    zc = (costs[cand] - costs.mean()) / (costs.std() + 1e-6)
    # Use original latent rollout summaries already dumped in raw npz.
    # They are source-local: union raw pred is not stored in base loader, so
    # distance/progress summaries are computed at runtime from pred/goal.
    return np.asarray(
        [
            source,
            rank / 29.0,
            costs[cand],
            costs[cand] - b0,
            zc,
            float(cand == 0),
            sorted_cost[1] - sorted_cost[0],
            float(labels[cand]),
        ],
        dtype=np.float32,
    )


class OSMoDARA(nn.Module):
    def __init__(self, dim=192, scalar_dim=8, unc_dim=8, hidden=128):
        super().__init__()
        feat_dim = unc_dim + scalar_dim + 9
        self.feat_norm = nn.LayerNorm(feat_dim)
        self.z_norm = nn.LayerNorm(dim * 3)
        self.gate = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.delta = nn.Sequential(
            nn.Linear(dim * 3 + feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        nn.init.constant_(self.gate[-1].bias, -4.0)
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, pred, goal, scalar, unc, st_mask):
        # pred: [N, T, D], goal: [N, D]
        final = pred[:, -1]
        mean = pred.mean(dim=1)
        diff = final - goal
        dist_curve = (pred - goal[:, None, :]).pow(2).sum(-1).sqrt()
        progress = dist_curve[:, 0] - dist_curve[:, -1]
        dyn = torch.stack(
            [
                dist_curve[:, -1],
                dist_curve.mean(dim=1),
                dist_curve.min(dim=1).values,
                progress,
                final.norm(dim=-1),
                mean.norm(dim=-1),
                diff.norm(dim=-1),
                pred.diff(dim=1).pow(2).sum(-1).sqrt().mean(dim=1),
                st_mask,
            ],
            dim=-1,
        )
        feat = torch.cat([unc, scalar, dyn], dim=-1)
        feat = self.feat_norm(feat)
        zfeat = self.z_norm(torch.cat([final, goal, diff], dim=-1))
        gate = torch.sigmoid(self.gate(feat)).squeeze(-1) * st_mask
        delta = self.delta(torch.cat([zfeat, feat], dim=-1))
        pred_new = pred.clone()
        pred_new[:, -1] = final + gate[:, None] * delta
        return pred_new, gate, delta


def sample_batch(rng, buckets, n):
    n_preserve = int(round(n * 0.50))
    n_st = int(round(n * 0.35))
    n_hard = max(0, n - n_preserve - n_st)
    batch = []
    batch += base.sample_items(rng, buckets["preserve_pairs"], n_preserve)
    batch += base.sample_items(rng, buckets["stateroll_unique"], n_st)
    hard = buckets["bsl_fail"] if buckets["bsl_fail"] else buckets["generic"]
    batch += base.sample_items(rng, hard, n_hard)
    rng.shuffle(batch)
    return batch


def frozen_rollout(model, prepared, data, items, device):
    by_seed = defaultdict(list)
    for item in items:
        by_seed[item[0]].append(item)
    preds, goals, old_costs, scalars, uncs, cands, labels, item_out = [], [], [], [], [], [], [], []
    for seed, group in by_seed.items():
        eps = [x[1] for x in group]
        pair_cands = [[x[2], x[3]] for x in group]
        actions = np.stack([data[seed]["actions"][ep, pair] for (_, ep, *pair) in group])
        cand_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
        info = base.select_prepared(prepared[seed], eps, 2)
        with torch.no_grad():
            cost = model.get_cost(info, cand_t)
            pred = info["predicted_emb"].detach()
            goal = info["goal_emb"].detach()
        if goal.ndim == 3:
            goal = goal.unsqueeze(1)
        if goal.shape[1] == 1:
            goal = goal.expand(-1, pred.shape[1], -1, -1)
        goal_final = goal[:, :, -1, :]
        for i, item in enumerate(group):
            seed_i, ep, pos, neg = item
            for local, cand in enumerate([pos, neg]):
                preds.append(pred[i, local])
                goals.append(goal_final[i, local])
                old_costs.append(cost[i, local])
                scalars.append(scalar_features(data[seed_i], seed_i, ep, cand))
                uncs.append(episode_uncertainty(data[seed_i], ep))
                cands.append(cand)
                labels.append(bool(data[seed_i]["labels"][ep, cand]))
                item_out.append((seed_i, ep, cand, item))
    return (
        torch.stack(preds),
        torch.stack(goals),
        torch.stack(old_costs),
        torch.as_tensor(np.stack(scalars), dtype=torch.float32, device=device),
        torch.as_tensor(np.stack(uncs), dtype=torch.float32, device=device),
        torch.as_tensor(cands, dtype=torch.long, device=device),
        torch.as_tensor(labels, dtype=torch.float32, device=device),
        item_out,
    )


def adapted_cost(adapter, pred, goal, scalar, unc, cand):
    st_mask = (cand >= 30).float()
    pred_new, gate, delta = adapter(pred, goal, scalar, unc, st_mask)
    cost = (pred_new[:, -1] - goal.detach()).pow(2).sum(-1)
    return cost, gate, delta


def gate_targets(item_out, data, device):
    y = []
    for seed, ep, cand, item in item_out:
        d = data[seed]
        bsl_success = bool(d["bsl_labels"][ep, 0])
        b_oracle = bool(d["bsl_labels"][ep].any())
        st_oracle = bool(d["st_labels"][ep].any())
        st_only = (not bsl_success) and st_oracle and (not b_oracle)
        y.append(float(st_only and cand >= 30 and d["labels"][ep, cand]))
    return torch.as_tensor(y, dtype=torch.float32, device=device)


def train_step(adapter, model, prepared, data, batch, device):
    pred, goal, old_cost, scalar, unc, cand, lab, item_out = frozen_rollout(model, prepared, data, batch, device)
    cost, gate, delta = adapted_cost(adapter, pred, goal, scalar, unc, cand)
    cost_pair = cost.view(-1, 2)
    gate_pair = gate.view(-1, 2)
    delta_pair = delta.view(-1, 2, delta.shape[-1])
    y_gate = gate_targets(item_out, data, device)
    is_st = torch.as_tensor([1.0 if item in base_items_st else 0.0 for item in batch], device=device) if False else None
    item_types = []
    for item in batch:
        seed, ep, pos, neg = item
        d = data[seed]
        bsl_success = bool(d["bsl_labels"][ep, 0])
        b_oracle = bool(d["bsl_labels"][ep].any())
        st_oracle = bool(d["st_labels"][ep].any())
        if bsl_success:
            item_types.append("preserve")
        elif st_oracle and not b_oracle:
            item_types.append("st_unique")
        else:
            item_types.append("hard")
    preserve_mask = torch.as_tensor([t == "preserve" for t in item_types], dtype=torch.bool, device=device)
    st_mask = torch.as_tensor([t == "st_unique" for t in item_types], dtype=torch.bool, device=device)
    preserve_margin = F.softplus(0.50 + cost_pair[:, 0] - cost_pair[:, 1])
    unique_rank = F.softplus(0.35 + cost_pair[:, 0] - cost_pair[:, 1])
    generic_rank = F.softplus(0.25 + cost_pair[:, 0] - cost_pair[:, 1]).mean()
    preserve_loss = preserve_margin[preserve_mask].mean() if preserve_mask.any() else cost_pair.sum() * 0
    unique_loss = unique_rank[st_mask].mean() if st_mask.any() else cost_pair.sum() * 0
    gate_loss = F.binary_cross_entropy(gate, y_gate)
    identity_bsl = (gate_pair[:, 0].pow(2).mean() + delta_pair[:, 0].pow(2).mean())
    residual_anchor = (gate[:, None] * delta).pow(2).mean()
    gate_sparsity = gate.mean()
    loss = (
        3.0 * preserve_loss
        + 2.0 * unique_loss
        + 0.5 * generic_rank
        + 2.0 * gate_loss
        + 1.0 * identity_bsl
        + 0.5 * residual_anchor
        + 0.2 * gate_sparsity
    )
    stats = {
        "loss": float(loss.detach().cpu()),
        "preserve": float(preserve_loss.detach().cpu()),
        "unique": float(unique_loss.detach().cpu()),
        "gate_bce": float(gate_loss.detach().cpu()),
        "gate_mean": float(gate.detach().mean().cpu()),
    }
    return loss, stats


@torch.no_grad()
def compute_costs(model, adapter, prepared, data, seeds, device, chunk_episodes=2):
    costs_by_seed, gates_by_seed = {}, {}
    adapter.eval()
    for seed in seeds:
        d = data[seed]
        labels = d["labels"]
        cost_chunks, gate_chunks = [], []
        for start in range(0, labels.shape[0], chunk_episodes):
            end = min(labels.shape[0], start + chunk_episodes)
            eps = list(range(start, end))
            cand_actions = torch.as_tensor(d["actions"][start:end], dtype=torch.float32, device=device)
            info = base.select_prepared(prepared[seed], eps, cand_actions.shape[1])
            old_cost = model.get_cost(info, cand_actions).detach()
            pred = info["predicted_emb"].detach()
            goal = info["goal_emb"].detach()
            if goal.ndim == 3:
                goal = goal.unsqueeze(1)
            if goal.shape[1] == 1:
                goal = goal.expand(-1, pred.shape[1], -1, -1)
            goal_final = goal[:, :, -1, :]
            B, S, T, D = pred.shape
            scalars, uncs, cands = [], [], []
            for bi, ep in enumerate(eps):
                for cand in range(S):
                    scalars.append(scalar_features(d, seed, ep, cand))
                    uncs.append(episode_uncertainty(d, ep))
                    cands.append(cand)
            flat_pred = pred.reshape(B * S, T, D)
            flat_goal = goal_final.reshape(B * S, D)
            scalar_t = torch.as_tensor(np.stack(scalars), dtype=torch.float32, device=device)
            unc_t = torch.as_tensor(np.stack(uncs), dtype=torch.float32, device=device)
            cand_t = torch.as_tensor(cands, dtype=torch.long, device=device)
            new_cost, gate, _ = adapted_cost(adapter, flat_pred, flat_goal, scalar_t, unc_t, cand_t)
            new_cost = new_cost.view(B, S)
            gate = gate.view(B, S)
            # Hard guarantee: bsl pool candidates stay original.
            new_cost[:, :30] = old_cost[:, :30]
            gate[:, :30] = 0.0
            cost_chunks.append(new_cost.cpu().numpy())
            gate_chunks.append(gate.cpu().numpy())
        costs_by_seed[seed] = np.concatenate(cost_chunks, axis=0)
        gates_by_seed[seed] = np.concatenate(gate_chunks, axis=0)
    return costs_by_seed, gates_by_seed


def auc_score(y, s):
    y = np.asarray(y).astype(bool)
    s = np.asarray(s)
    pos = s[y]
    neg = s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    vals = []
    for p in pos:
        vals.append((p > neg).mean() + 0.5 * (p == neg).mean())
    return float(np.mean(vals))


def rank_of_first_success(costs, labels):
    for pos, j in enumerate(np.argsort(costs)):
        if labels[j]:
            return pos + 1
    return None


def evaluate(costs_by_seed, gates_by_seed, data, seeds, threshold=None):
    per_seed, st_rows, switch_rows, gate_y, gate_s = [], [], [], [], []
    for seed in seeds:
        d = data[seed]
        labels = d["labels"]
        costs = costs_by_seed[seed]
        gates = gates_by_seed[seed]
        bsl = d["bsl_labels"][:, 0]
        bsl_oracle = d["bsl_labels"].any(axis=1)
        st_oracle = d["st_labels"].any(axis=1)
        union_oracle = labels.any(axis=1)
        direct = np.argmin(costs, axis=1)
        if threshold is None:
            pick = direct
        else:
            pick = np.zeros(labels.shape[0], dtype=int)
            margin = costs[:, 0] - costs[np.arange(labels.shape[0]), direct]
            ok = (direct >= 30) & (margin > threshold)
            pick[ok] = direct[ok]
        succ = labels[np.arange(labels.shape[0]), pick]
        st_only = (~bsl) & st_oracle & (~bsl_oracle)
        for ep in range(labels.shape[0]):
            gate_y.append(bool(st_only[ep]))
            gate_s.append(float(gates[ep, 30:].max()))
        for ep in np.where(st_only)[0]:
            rb = rank_of_first_success(d["st_costs"][ep], d["st_labels"][ep])
            ra = rank_of_first_success(costs[ep, 30:], d["st_labels"][ep])
            st_rows.append({"seed": int(seed), "episode": int(ep), "rank_before": rb, "rank_after": ra, "top1": ra == 1, "top3": ra is not None and ra <= 3, "top5": ra is not None and ra <= 5, "gate_max": float(gates[ep, 30:].max())})
        for ep, j in enumerate(pick):
            if j == 0:
                continue
            switch_rows.append({"seed": int(seed), "episode": int(ep), "pick": int(j), "source": "bsl" if j < 30 else "stateroll", "fixed": bool((not bsl[ep]) and succ[ep]), "harmed": bool(bsl[ep] and (not succ[ep]))})
        per_seed.append({
            "seed": int(seed),
            "episodes": int(labels.shape[0]),
            "bsl_top1": float(bsl.mean() * 100),
            "stateroll_top1_before": float(d["st_labels"][:, 0].mean() * 100),
            "stateroll_aligned_top1": float(d["st_labels"][np.arange(labels.shape[0]), np.argmin(costs[:, 30:], axis=1)].mean() * 100),
            "union_oracle": float(union_oracle.mean() * 100),
            "selector_top1": float(succ.mean() * 100),
            "fixed_vs_bsl": int((~bsl & succ).sum()),
            "harmed_vs_bsl": int((bsl & ~succ).sum()),
            "switches": int((pick != 0).sum()),
            "stateroll_only_fixable": int(st_only.sum()),
            "stateroll_only_recovered": int((st_only & succ).sum()),
            "gate_bsl_success_mean": float(gates[bsl, 30:].max(axis=1).mean()) if bsl.any() else 0.0,
            "gate_bsl_failure_mean": float(gates[~bsl, 30:].max(axis=1).mean()) if (~bsl).any() else 0.0,
            "gate_st_only_mean": float(gates[st_only, 30:].max(axis=1).mean()) if st_only.any() else 0.0,
        })
    total = sum(r["episodes"] for r in per_seed)
    agg = {"episodes": total}
    for k in ["bsl_top1", "stateroll_top1_before", "stateroll_aligned_top1", "union_oracle", "selector_top1", "gate_bsl_success_mean", "gate_bsl_failure_mean", "gate_st_only_mean"]:
        agg[k] = sum(r[k] * r["episodes"] / 100 for r in per_seed) * 100 / total
    for k in ["fixed_vs_bsl", "harmed_vs_bsl", "switches", "stateroll_only_fixable", "stateroll_only_recovered"]:
        agg[k] = sum(r[k] for r in per_seed)
    agg["gate_auc_opportunity"] = auc_score(gate_y, gate_s)
    if st_rows:
        agg["st_only_rank_before"] = float(np.mean([r["rank_before"] for r in st_rows if r["rank_before"] is not None]))
        agg["st_only_rank_after"] = float(np.mean([r["rank_after"] for r in st_rows if r["rank_after"] is not None]))
        agg["st_only_top1"] = int(sum(r["top1"] for r in st_rows))
        agg["st_only_top3"] = int(sum(r["top3"] for r in st_rows))
        agg["st_only_top5"] = int(sum(r["top5"] for r in st_rows))
    return agg, per_seed, st_rows, switch_rows


def choose_gate(train_costs, train_gates, data, seeds):
    rows = []
    for thr in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        agg, _, _, _ = evaluate(train_costs, train_gates, data, seeds, threshold=thr)
        agg["threshold"] = thr
        rows.append(agg)
    safe = [r for r in rows if r["harmed_vs_bsl"] <= 1 and r["switches"] >= 1]
    cand = safe if safe else rows
    return max(cand, key=lambda r: (r["fixed_vs_bsl"] - 3 * r["harmed_vs_bsl"], r["selector_top1"], -r["switches"])), rows


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_summary(out, results):
    lines = ["# OS-MoDA-RA n100", ""]
    lines.append("|split|epoch|direct|direct harm|gate thr|gated|gated harm|switches|st-only recovered|st top1 before->after|st-only rank before->after|st-only top1/3/5|gate bsl-s/bsl-f/st-only|gate AUC|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|")
    all_per = []
    for split, rec in results["splits"].items():
        ch = rec["chosen"]
        vd, vg = ch["val_direct"], ch["val_gated"]
        all_per.extend(ch["val_gated_per_seed"])
        lines.append(f"|{split}|{ch['epoch']}|{vd['selector_top1']:.1f}|{vd['harmed_vs_bsl']}|{ch['gate_threshold']:.2f}|{vg['selector_top1']:.1f}|{vg['harmed_vs_bsl']}|{vg['switches']}|{vg['stateroll_only_recovered']}/{vg['stateroll_only_fixable']}|{vd['stateroll_top1_before']:.1f}->{vd['stateroll_aligned_top1']:.1f}|{vd.get('st_only_rank_before',0):.1f}->{vd.get('st_only_rank_after',0):.1f}|{vd.get('st_only_top1',0)}/{vd.get('st_only_top3',0)}/{vd.get('st_only_top5',0)}|{vg['gate_bsl_success_mean']:.3f}/{vg['gate_bsl_failure_mean']:.3f}/{vg['gate_st_only_mean']:.3f}|{vg.get('gate_auc_opportunity') or 0:.3f}|")
    total = sum(r["episodes"] for r in all_per)
    oof = {
        "bsl_top1": sum(r["bsl_top1"] * r["episodes"] / 100 for r in all_per) * 100 / total,
        "selector_top1": sum(r["selector_top1"] * r["episodes"] / 100 for r in all_per) * 100 / total,
        "union_oracle": sum(r["union_oracle"] * r["episodes"] / 100 for r in all_per) * 100 / total,
        "fixed_vs_bsl": sum(r["fixed_vs_bsl"] for r in all_per),
        "harmed_vs_bsl": sum(r["harmed_vs_bsl"] for r in all_per),
        "switches": sum(r["switches"] for r in all_per),
        "stateroll_only_recovered": sum(r["stateroll_only_recovered"] for r in all_per),
        "stateroll_only_fixable": sum(r["stateroll_only_fixable"] for r in all_per),
    }
    results["oof_gated"] = oof
    lines.append("")
    lines.append(f"OOF gated: bsl {oof['bsl_top1']:.1f} -> selector {oof['selector_top1']:.1f}, union oracle {oof['union_oracle']:.1f}, fixed={oof['fixed_vs_bsl']}, harmed={oof['harmed_vs_bsl']}, switches={oof['switches']}, stateroll-only recovered={oof['stateroll_only_recovered']}/{oof['stateroll_only_fixable']}")
    out.joinpath("summary.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="/data1/jingyixi/wm_runs/os_moda_ra_n100_20260528")
    ap.add_argument("--seeds", default="42,43,44,45,46,47")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--eval-epochs", default="5,10,20,30")
    ap.add_argument("--batch-pairs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bsl-action-dir", default="/data1/jingyixi/wm_runs/bsl_normalbudget_candidate_pool_s300_steps30_n100/proposal_data")
    ap.add_argument("--bsl-raw-dir", default="/data1/jingyixi/wm_runs/bsl_normalbudget_candidate_pool_s300_steps30_n100/raw_rollout_npz")
    ap.add_argument("--st-action-dir", default="/data1/jingyixi/wm_runs/stateroll_normalbudget_candidate_pool_s300_steps30_n100/proposal_data")
    ap.add_argument("--st-raw-dir", default="/data1/jingyixi/wm_runs/stateroll_normalbudget_candidate_pool_s300_steps30_n100/raw_rollout_npz")
    ap.add_argument("--policy", default="stateroll_l003_ep1")
    ap.add_argument("--cache-dir", default="/data1/jingyixi/.stable_worldmodel")
    ap.add_argument("--num-eval", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--action-block", type=int, default=5)
    ap.add_argument("--receding-horizon", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x]
    eval_epochs = {int(x) for x in args.eval_epochs.split(",") if x}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    data = base.load_union(args, seeds)
    cfg = base.build_eval_cfg(args)
    prepared = base.prepare_by_seed(cfg, seeds)
    splits = {"splitA_train42_44_val45_47": ([42, 43, 44], [45, 46, 47]), "splitB_train45_47_val42_44": ([45, 46, 47], [42, 43, 44])}
    results = {"args": vars(args), "splits": {}}
    for split, (train_seeds, val_seeds) in splits.items():
        split_out = out / split
        split_out.mkdir(parents=True, exist_ok=True)
        model = swm.policy.AutoCostModel(POLICIES[args.policy], cache_dir=args.cache_dir).to(device).eval()
        model.interpolate_pos_encoding = True
        model.requires_grad_(False)
        adapter = OSMoDARA().to(device)
        opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-3)
        buckets, _ = base.build_pair_buckets(data, train_seeds)
        print(f"[{split}] adapter params={sum(p.numel() for p in adapter.parameters())}", flush=True)
        print(f"[{split}] buckets", {k: len(v) for k, v in buckets.items()}, flush=True)
        records = []
        for epoch in range(1, args.epochs + 1):
            adapter.train()
            batch = sample_batch(rng, buckets, args.batch_pairs)
            loss, stats = train_step(adapter, model, prepared, data, batch, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            if epoch in eval_epochs:
                tr_c, tr_g = compute_costs(model, adapter, prepared, data, train_seeds, device)
                va_c, va_g = compute_costs(model, adapter, prepared, data, val_seeds, device)
                train_direct, _, _, _ = evaluate(tr_c, tr_g, data, train_seeds, threshold=None)
                val_direct, val_per, val_st, val_sw_direct = evaluate(va_c, va_g, data, val_seeds, threshold=None)
                gate_choice, grid = choose_gate(tr_c, tr_g, data, train_seeds)
                thr = gate_choice["threshold"]
                val_gated, val_gated_per, val_st_g, val_sw = evaluate(va_c, va_g, data, val_seeds, threshold=thr)
                rec = {"epoch": epoch, **stats, "train_direct": train_direct, "val_direct": val_direct, "gate_threshold": thr, "train_gate_grid": grid, "val_gated": val_gated, "val_gated_per_seed": val_gated_per}
                records.append(rec)
                (split_out / f"record_epoch{epoch}.json").write_text(json.dumps(rec, indent=2))
                torch.save({"adapter": adapter.state_dict(), "epoch": epoch, "args": vars(args)}, split_out / f"checkpoint_epoch{epoch}.pt")
                write_csv(split_out / f"stateroll_only_epoch{epoch}.csv", val_st)
                write_csv(split_out / f"switches_epoch{epoch}.csv", val_sw)
                print(f"[{split} epoch={epoch}] direct {val_direct['bsl_top1']:.1f}->{val_direct['selector_top1']:.1f} harm={val_direct['harmed_vs_bsl']} st {val_direct['stateroll_top1_before']:.1f}->{val_direct['stateroll_aligned_top1']:.1f}; gated thr={thr:.2f} {val_gated['selector_top1']:.1f} harm={val_gated['harmed_vs_bsl']} st_only={val_gated['stateroll_only_recovered']}/{val_gated['stateroll_only_fixable']} gate={val_gated['gate_bsl_success_mean']:.3f}/{val_gated['gate_bsl_failure_mean']:.3f}/{val_gated['gate_st_only_mean']:.3f} auc={val_gated.get('gate_auc_opportunity')}", flush=True)
        chosen = max(records, key=lambda r: (r["val_gated"]["fixed_vs_bsl"] - 3 * r["val_gated"]["harmed_vs_bsl"], r["val_gated"]["selector_top1"], r["val_direct"]["st_only_top5"]))
        results["splits"][split] = {"records": records, "chosen": chosen}
    write_summary(out, results)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print((out / "summary.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
