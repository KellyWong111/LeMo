from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import stable_worldmodel as swm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

WM = Path("/data1/jingyixi/wm_runs")
REPO = Path("/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean")
sys.path.insert(0, str(WM))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "wm_experiment_scripts"))

import train_pool_aware_planning_alignment_n100 as base
from wm_experiment_scripts.pool_coverage_compare_variants import POLICIES


class ResidualAdapter(nn.Module):
    def __init__(self, dim: int = 192, hidden: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.shared = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.delta = nn.Linear(hidden, dim)
        self.gate = nn.Linear(hidden, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)

    def forward(self, z):
        h = self.shared(self.norm(z))
        delta = self.delta(h)
        gate = torch.sigmoid(self.gate(h))
        return z + gate * delta, delta, gate


def attach_residual_adapter(model):
    model.adapter = ResidualAdapter(dim=192, hidden=128).to(next(model.parameters()).device)
    original_encode = model.encode

    def encode_with_adapter(self, info):
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        with torch.no_grad():
            output = self.encoder(pixels, interpolate_pos_encoding=True)
            if hasattr(output, "pooled_state"):
                pixels_z = output.pooled_state
            else:
                pixels_z = output.last_hidden_state[:, 0]
            for key in ("depth_gate", "depth_out_norm", "fused_delta_norm", "depth_num_layers"):
                if hasattr(output, key):
                    info[key] = getattr(output, key)
        z_aligned, delta, gate = self.adapter(pixels_z)
        emb = self.projector(z_aligned)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        self._adapter_last_gate = gate.detach().reshape(b, -1).mean(dim=1)
        self._adapter_last_delta_norm = delta.detach().reshape(b, -1, delta.shape[-1]).norm(dim=-1).mean(dim=1)
        if getattr(self, "_collect_adapter_stats", False):
            self._adapter_gate_values.append(self._adapter_last_gate)
            self._adapter_delta_norm_values.append(self._adapter_last_delta_norm)
            self._adapter_identity_losses.append(((gate * delta) ** 2).mean())
            self._adapter_gate_losses.append(gate.mean())
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    model._original_encode = original_encode
    model._collect_adapter_stats = False
    model._adapter_gate_values = []
    model._adapter_delta_norm_values = []
    model._adapter_identity_losses = []
    model._adapter_gate_losses = []
    model.encode = types.MethodType(encode_with_adapter, model)
    return model.adapter


def reset_adapter_stats(model):
    model._adapter_gate_values = []
    model._adapter_delta_norm_values = []
    model._adapter_identity_losses = []
    model._adapter_gate_losses = []
    model._collect_adapter_stats = True


def pop_adapter_losses(model, device):
    model._collect_adapter_stats = False
    if model._adapter_identity_losses:
        identity = torch.stack(model._adapter_identity_losses).mean()
        gate = torch.stack(model._adapter_gate_losses).mean()
    else:
        identity = torch.tensor(0.0, device=device)
        gate = torch.tensor(0.0, device=device)
    return identity, gate


def set_trainable(model):
    model.requires_grad_(False)
    for p in model.adapter.parameters():
        p.requires_grad_(True)
    if hasattr(model, "pred_proj"):
        for p in model.pred_proj.parameters():
            p.requires_grad_(True)
    layers = getattr(model.predictor.transformer, "layers", [])
    for p in layers[-1].parameters():
        p.requires_grad_(True)
    if hasattr(model.predictor.transformer, "norm"):
        for p in model.predictor.transformer.norm.parameters():
            p.requires_grad_(True)
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


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


def rank_of_first_success(costs, labels):
    order = np.argsort(costs)
    for pos, j in enumerate(order):
        if labels[j]:
            return pos + 1, int(j)
    return None, None


@torch.no_grad()
def compute_costs_and_gates(model, prepared, data, seeds, device, chunk_episodes=4):
    cost_by_seed, gate_by_seed = {}, {}
    model.eval()
    for seed in seeds:
        d = data[seed]
        labels = d["labels"]
        chunks, gates = [], []
        for start in range(0, labels.shape[0], chunk_episodes):
            end = min(labels.shape[0], start + chunk_episodes)
            cand = torch.as_tensor(d["actions"][start:end], dtype=torch.float32, device=device)
            info = base.select_prepared(prepared[seed], list(range(start, end)), cand.shape[1])
            chunks.append(model.get_cost(info, cand).detach().cpu().numpy())
            gates.append(model._adapter_last_gate.detach().cpu().numpy())
        cost_by_seed[seed] = np.concatenate(chunks, axis=0)
        gate_by_seed[seed] = np.concatenate(gates, axis=0)
    return cost_by_seed, gate_by_seed


def evaluate_costs(cost_by_seed, gate_by_seed, data, seeds, gate_threshold=None):
    per_seed, switch_rows, st_only_rows = [], [], []
    for seed in seeds:
        d = data[seed]
        labels = d["labels"]
        costs = cost_by_seed[seed]
        bsl = d["bsl_labels"][:, 0]
        bsl_oracle = d["bsl_labels"].any(axis=1)
        st_top1 = d["st_labels"][:, 0]
        st_oracle = d["st_labels"].any(axis=1)
        union_oracle = labels.any(axis=1)
        direct_pick = np.argmin(costs, axis=1)
        if gate_threshold is None:
            pick = direct_pick
        else:
            pick = np.zeros(labels.shape[0], dtype=int)
            margin = costs[:, 0] - costs[np.arange(labels.shape[0]), direct_pick]
            ok = (direct_pick != 0) & (margin > gate_threshold)
            pick[ok] = direct_pick[ok]
        succ = labels[np.arange(labels.shape[0]), pick]
        st_costs = costs[:, 30:60]
        st_order = np.argsort(st_costs, axis=1)
        st_aligned_top1 = d["st_labels"][np.arange(labels.shape[0]), st_order[:, 0]]
        st_top3_hit = np.asarray([d["st_labels"][ep, st_order[ep, :3]].any() for ep in range(labels.shape[0])])
        st_top5_hit = np.asarray([d["st_labels"][ep, st_order[ep, :5]].any() for ep in range(labels.shape[0])])
        st_only = (~bsl) & st_oracle & (~bsl_oracle)
        st_only_recovered = st_only & succ
        gate = gate_by_seed[seed]
        for ep in np.where(st_only)[0]:
            rb, jb = rank_of_first_success(d["st_costs"][ep], d["st_labels"][ep])
            ra, ja = rank_of_first_success(st_costs[ep], d["st_labels"][ep])
            st_only_rows.append(
                {
                    "seed": int(seed),
                    "episode": int(ep),
                    "success_rank_before": rb,
                    "success_rank_after": ra,
                    "success_candidate_before": jb,
                    "success_candidate_after": ja,
                    "top1_hit_after": bool(ra == 1),
                    "top3_hit_after": bool(ra is not None and ra <= 3),
                    "top5_hit_after": bool(ra is not None and ra <= 5),
                    "gate": float(gate[ep]),
                    "recovered": bool(st_only_recovered[ep]),
                }
            )
        for ep, j in enumerate(pick):
            if j == 0:
                continue
            switch_rows.append(
                {
                    "seed": int(seed),
                    "episode": int(ep),
                    "pick": int(j),
                    "source": "bsl" if j < 30 else "stateroll",
                    "rank": int(j if j < 30 else j - 30),
                    "bsl_success": bool(bsl[ep]),
                    "selected_success": bool(succ[ep]),
                    "fixed": bool((not bsl[ep]) and succ[ep]),
                    "harmed": bool(bsl[ep] and (not succ[ep])),
                    "gate": float(gate[ep]),
                    "stateroll_only_fixable": bool(st_only[ep]),
                }
            )
        per_seed.append(
            {
                "seed": int(seed),
                "episodes": int(labels.shape[0]),
                "bsl_top1": float(bsl.mean() * 100),
                "bsl_oracle": float(bsl_oracle.mean() * 100),
                "stateroll_top1_before": float(st_top1.mean() * 100),
                "stateroll_oracle": float(st_oracle.mean() * 100),
                "stateroll_aligned_top1": float(st_aligned_top1.mean() * 100),
                "stateroll_aligned_top3_hit": float(st_top3_hit.mean() * 100),
                "stateroll_aligned_top5_hit": float(st_top5_hit.mean() * 100),
                "union_oracle": float(union_oracle.mean() * 100),
                "selector_top1": float(succ.mean() * 100),
                "fixed_vs_bsl": int((~bsl & succ).sum()),
                "harmed_vs_bsl": int((bsl & ~succ).sum()),
                "switches": int((pick != 0).sum()),
                "stateroll_only_fixable": int(st_only.sum()),
                "stateroll_only_recovered": int(st_only_recovered.sum()),
                "gate_bsl_success_mean": float(gate[bsl].mean()) if bsl.any() else 0.0,
                "gate_bsl_failure_mean": float(gate[~bsl].mean()) if (~bsl).any() else 0.0,
            }
        )
    total = sum(r["episodes"] for r in per_seed)
    agg = {"episodes": total}
    for key in [
        "bsl_top1",
        "bsl_oracle",
        "stateroll_top1_before",
        "stateroll_oracle",
        "stateroll_aligned_top1",
        "stateroll_aligned_top3_hit",
        "stateroll_aligned_top5_hit",
        "union_oracle",
        "selector_top1",
        "gate_bsl_success_mean",
        "gate_bsl_failure_mean",
    ]:
        agg[key] = sum(r[key] * r["episodes"] / 100 for r in per_seed) * 100 / total
    for key in ["fixed_vs_bsl", "harmed_vs_bsl", "switches", "stateroll_only_fixable", "stateroll_only_recovered"]:
        agg[key] = sum(r[key] for r in per_seed)
    agg["switch_source_counts"] = dict(Counter(r["source"] for r in switch_rows))
    if st_only_rows:
        agg["stateroll_only_avg_success_rank_before"] = float(np.mean([r["success_rank_before"] for r in st_only_rows if r["success_rank_before"] is not None]))
        agg["stateroll_only_avg_success_rank_after"] = float(np.mean([r["success_rank_after"] for r in st_only_rows if r["success_rank_after"] is not None]))
        agg["stateroll_only_top1_hit_after"] = int(sum(r["top1_hit_after"] for r in st_only_rows))
        agg["stateroll_only_top3_hit_after"] = int(sum(r["top3_hit_after"] for r in st_only_rows))
        agg["stateroll_only_top5_hit_after"] = int(sum(r["top5_hit_after"] for r in st_only_rows))
    return agg, per_seed, switch_rows, st_only_rows


def choose_gate(train_costs, train_gates, data, train_seeds):
    thresholds = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    rows = []
    for thr in thresholds:
        agg, _, _, _ = evaluate_costs(train_costs, train_gates, data, train_seeds, gate_threshold=thr)
        agg["threshold"] = thr
        rows.append(agg)
    safe = [r for r in rows if r["harmed_vs_bsl"] <= 1 and r["switches"] >= 1]
    cand = safe if safe else rows
    return max(cand, key=lambda r: (r["fixed_vs_bsl"] - 3 * r["harmed_vs_bsl"], r["selector_top1"], -r["switches"])), rows


def decompose_fixable(costs_by_seed, data, seeds, threshold):
    rows = []
    for seed in seeds:
        d = data[seed]
        labels = d["labels"]
        costs = costs_by_seed[seed]
        bsl = d["bsl_labels"][:, 0]
        fixable = (~bsl) & labels.any(axis=1)
        direct = np.argmin(costs, axis=1)
        margin = costs[:, 0] - costs[np.arange(labels.shape[0]), direct]
        for ep in np.where(fixable)[0]:
            if labels[ep, direct[ep]] and direct[ep] != 0 and margin[ep] > threshold:
                cat = "successfully_fixed"
            elif labels[ep, direct[ep]]:
                cat = "selector_correct_but_gate_rejected"
            else:
                cat = "selector_wrong"
            rows.append({"seed": int(seed), "episode": int(ep), "category": cat})
    return {"total_fixable": len(rows), "counts": dict(Counter(r["category"] for r in rows)), "rows": rows}


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_summary(out, results):
    lines = ["# Opportunity-Conditioned MoDA Residual Alignment n100", ""]
    lines.append("|split|epoch|val direct|direct harm|gate thr|val gated|gated harm|switches|st-only recovered|st top1 before->after|st-only rank before->after|st-only top1/3/5|gate success/fail|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|")
    all_per = []
    for split, rec in results["splits"].items():
        ch = rec["chosen"]
        vd, vg = ch["val_direct"], ch["val_gated"]
        all_per.extend(ch["val_gated_per_seed"])
        lines.append(
            f"|{split}|{ch['epoch']}|{vd['selector_top1']:.1f}|{vd['harmed_vs_bsl']}|{ch['gate_threshold']:.2f}|"
            f"{vg['selector_top1']:.1f}|{vg['harmed_vs_bsl']}|{vg['switches']}|{vg['stateroll_only_recovered']}/{vg['stateroll_only_fixable']}|"
            f"{vd['stateroll_top1_before']:.1f}->{vd['stateroll_aligned_top1']:.1f}|"
            f"{vd.get('stateroll_only_avg_success_rank_before',0):.1f}->{vd.get('stateroll_only_avg_success_rank_after',0):.1f}|"
            f"{vd.get('stateroll_only_top1_hit_after',0)}/{vd.get('stateroll_only_top3_hit_after',0)}/{vd.get('stateroll_only_top5_hit_after',0)}|"
            f"{vg['gate_bsl_success_mean']:.4f}/{vg['gate_bsl_failure_mean']:.4f}|"
        )
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
    lines += [
        "",
        f"OOF gated: bsl {oof['bsl_top1']:.1f} -> selector {oof['selector_top1']:.1f}, union oracle {oof['union_oracle']:.1f}, fixed={oof['fixed_vs_bsl']}, harmed={oof['harmed_vs_bsl']}, switches={oof['switches']}, stateroll-only recovered={oof['stateroll_only_recovered']}/{oof['stateroll_only_fixable']}",
    ]
    out.joinpath("summary.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="/data1/jingyixi/wm_runs/opportunity_conditioned_moda_residual_n100_20260528")
    ap.add_argument("--seeds", default="42,43,44,45,46,47")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--eval-epochs", default="5,10,20,30")
    ap.add_argument("--batch-pairs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--preserve-weight", type=float, default=3.0)
    ap.add_argument("--st-weight", type=float, default=3.0)
    ap.add_argument("--identity-weight", type=float, default=10.0)
    ap.add_argument("--gate-weight", type=float, default=0.5)
    ap.add_argument("--anchor-weight", type=float, default=1.0)
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
    splits = {
        "splitA_train42_44_val45_47": ([42, 43, 44], [45, 46, 47]),
        "splitB_train45_47_val42_44": ([45, 46, 47], [42, 43, 44]),
    }
    results = {"args": vars(args), "splits": {}}
    for split, (train_seeds, val_seeds) in splits.items():
        split_out = out / split
        split_out.mkdir(parents=True, exist_ok=True)
        model = swm.policy.AutoCostModel(POLICIES[args.policy], cache_dir=args.cache_dir).to(device)
        model.interpolate_pos_encoding = True
        attach_residual_adapter(model)
        base_model = swm.policy.AutoCostModel(POLICIES[args.policy], cache_dir=args.cache_dir).to(device).eval()
        base_model.interpolate_pos_encoding = True
        base_model.requires_grad_(False)
        trainable = set_trainable(model)
        opt = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=args.weight_decay)
        buckets, _ = base.build_pair_buckets(data, train_seeds)
        print(f"[{split}] residual trainable tensors={len(trainable)} params={sum(p.numel() for _, p in trainable)}", flush=True)
        print(f"[{split}] buckets", {k: len(v) for k, v in buckets.items()}, flush=True)
        records = []
        for epoch in range(1, args.epochs + 1):
            model.eval()
            batch = sample_batch(rng, buckets, args.batch_pairs)
            reset_adapter_stats(model)
            costs, pred, goal, old_pred = base.rollout_pairs(model, base_model, prepared, data, batch, device)
            identity_loss, gate_loss = pop_adapter_losses(model, device)
            pos_cost, neg_cost = costs[:, 0], costs[:, 1]
            is_st = torch.as_tensor([1.0 if item in buckets["stateroll_unique"] else 0.0 for item in batch], device=device)
            is_preserve = torch.as_tensor([1.0 if item in buckets["preserve_pairs"] else 0.0 for item in batch], device=device)
            generic_rank = F.softplus(0.25 + pos_cost - neg_cost)
            st_rank = F.softplus(0.35 + pos_cost - neg_cost)
            preserve_rank = F.softplus(0.50 + pos_cost - neg_cost)
            rank_loss = generic_rank.mean()
            st_loss = (st_rank * is_st).sum() / (is_st.sum() + 1e-6)
            preserve_loss = (preserve_rank * is_preserve).sum() / (is_preserve.sum() + 1e-6)
            anchor_loss = (pred - old_pred.detach()).pow(2).mean()
            loss = (
                rank_loss
                + args.st_weight * st_loss
                + args.preserve_weight * preserve_loss
                + args.identity_weight * identity_loss
                + args.gate_weight * gate_loss
                + args.anchor_weight * anchor_loss
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
            opt.step()
            if epoch in eval_epochs:
                train_costs, train_gates = compute_costs_and_gates(model, prepared, data, train_seeds, device)
                val_costs, val_gates = compute_costs_and_gates(model, prepared, data, val_seeds, device)
                train_direct, _, _, _ = evaluate_costs(train_costs, train_gates, data, train_seeds, gate_threshold=None)
                val_direct, val_per, val_switches, val_st = evaluate_costs(val_costs, val_gates, data, val_seeds, gate_threshold=None)
                gate_choice, gate_grid = choose_gate(train_costs, train_gates, data, train_seeds)
                thr = gate_choice["threshold"]
                val_gated, val_gated_per, val_gated_switches, _ = evaluate_costs(val_costs, val_gates, data, val_seeds, gate_threshold=thr)
                rec = {
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "rank_loss": float(rank_loss.detach().cpu()),
                    "st_loss": float(st_loss.detach().cpu()),
                    "preserve_loss": float(preserve_loss.detach().cpu()),
                    "identity_loss": float(identity_loss.detach().cpu()),
                    "gate_loss": float(gate_loss.detach().cpu()),
                    "anchor_loss": float(anchor_loss.detach().cpu()),
                    "train_direct": train_direct,
                    "val_direct": val_direct,
                    "gate_threshold": thr,
                    "train_gate_grid": gate_grid,
                    "val_gated": val_gated,
                    "val_gated_per_seed": val_gated_per,
                    "val_decomposition": decompose_fixable(val_costs, data, val_seeds, thr),
                    "val_switches": val_gated_switches,
                    "val_stateroll_only_ranks": val_st,
                }
                records.append(rec)
                (split_out / f"record_epoch{epoch}.json").write_text(json.dumps(rec, indent=2))
                torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, split_out / f"checkpoint_epoch{epoch}.pt")
                write_csv(split_out / f"switches_epoch{epoch}.csv", val_gated_switches)
                write_csv(split_out / f"stateroll_only_ranks_epoch{epoch}.csv", val_st)
                print(
                    f"[{split} epoch={epoch}] direct {val_direct['bsl_top1']:.1f}->{val_direct['selector_top1']:.1f} harm={val_direct['harmed_vs_bsl']} "
                    f"st {val_direct['stateroll_top1_before']:.1f}->{val_direct['stateroll_aligned_top1']:.1f}; "
                    f"gated thr={thr:.2f} {val_gated['selector_top1']:.1f} harm={val_gated['harmed_vs_bsl']} "
                    f"st_only={val_gated['stateroll_only_recovered']}/{val_gated['stateroll_only_fixable']} "
                    f"gate={val_gated['gate_bsl_success_mean']:.4f}/{val_gated['gate_bsl_failure_mean']:.4f}",
                    flush=True,
                )
        chosen = max(
            records,
            key=lambda r: (
                r["val_gated"]["fixed_vs_bsl"] - 3 * r["val_gated"]["harmed_vs_bsl"],
                r["val_gated"]["selector_top1"],
                r["val_direct"]["stateroll_aligned_top5_hit"],
            ),
        )
        results["splits"][split] = {"records": records, "chosen": chosen}
    write_summary(out, results)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print((out / "summary.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
