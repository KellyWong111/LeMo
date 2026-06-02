from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
import torch.nn.functional as F

WM = Path("/data1/jingyixi/wm_runs")
sys.path.insert(0, str(WM))
import train_pool_aware_planning_alignment_n100 as base

REPO = Path("/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "wm_experiment_scripts"))
from wm_experiment_scripts.pool_coverage_compare_variants import POLICIES


def patch_moda_attention_kernel():
    """Use smaller MoDA Triton backward tiles for 5090 shared-memory limits.

    This patch is local to this training script. It does not edit the repo's
    moda_module.py, and it keeps the forward path identical except for passing
    conservative backward kernel parameters exposed by the v14 API.
    """
    import moda_module

    def _parallel_moda_small_bwd(self, q, k, v, cached_k=None, cached_v=None):
        moda_kernel = moda_module.parallel_moda_chunk_visible if self.chunk_visible else moda_module.parallel_moda
        scale_multiplier = float(getattr(self, "attention_scale_multiplier", 1.0))
        kwargs = dict(
            scale=self.scale * scale_multiplier,
            moda_group_num=1,
            head_first=False,
            need_lse=False,
            warn_shape=False,
        )
        if not self.chunk_visible:
            kwargs.update(
                customized_BT_backward=32,
                customized_BS_backward=16,
                group_bs=16,
                group_warps=1,
                depth_bs=16,
                depth_warps=1,
            )
        return moda_kernel(q, k, v, cached_k=cached_k, cached_v=cached_v, **kwargs)

    moda_module.MoDAAttention._parallel_moda = _parallel_moda_small_bwd


def set_trainable_c_strict(model):
    """C_strict: late MoDAAttention/gate_proj/norm + predictor last1 + pred_proj."""
    model.requires_grad_(False)
    late_layers = (10, 11)
    for name, p in model.named_parameters():
        train = False
        if name.startswith("pred_proj."):
            train = True
        if name.startswith("predictor.transformer.layers.5.") or name.startswith("predictor.transformer.norm."):
            train = True
        if name == "encoder.transformer.depth_cache_gate":
            train = True
        for i in late_layers:
            if name.startswith(f"encoder.transformer.layers.{i}.attn."):
                train = True
            if name.startswith(f"encoder.transformer.layers.{i}.mlp.gate_proj"):
                train = True
            if name.startswith(f"encoder.transformer.layers.{i}.attn_norm") or name.startswith(f"encoder.transformer.layers.{i}.mlp_norm"):
                train = True
        if name.startswith("encoder.transformer.norm."):
            train = True
        p.requires_grad_(train)
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


@torch.no_grad()
def compute_costs(model, prepared, data, seeds, device, chunk_episodes=4):
    out = {}
    for seed in seeds:
        d = data[seed]
        labels = d["labels"]
        chunks = []
        for start in range(0, labels.shape[0], chunk_episodes):
            end = min(labels.shape[0], start + chunk_episodes)
            cand = torch.as_tensor(d["actions"][start:end], dtype=torch.float32, device=device)
            info = base.select_prepared(prepared[seed], list(range(start, end)), cand.shape[1])
            chunks.append(model.get_cost(info, cand).detach().cpu().numpy())
        out[seed] = np.concatenate(chunks, axis=0)
    return out


def rank_of_first_success(costs, labels):
    order = np.argsort(costs)
    for pos, j in enumerate(order):
        if labels[j]:
            return pos + 1, int(j)
    return None, None


def evaluate_costs(cost_by_seed, data, seeds, gate_threshold=None):
    per_seed = []
    switch_rows = []
    st_only_rows = []
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

        for ep in np.where(st_only)[0]:
            r_before, j_before = rank_of_first_success(d["st_costs"][ep], d["st_labels"][ep])
            r_after, j_after_rel = rank_of_first_success(st_costs[ep], d["st_labels"][ep])
            st_only_rows.append(
                {
                    "seed": int(seed),
                    "episode": int(ep),
                    "success_rank_before": r_before,
                    "success_rank_after": r_after,
                    "success_candidate_before": j_before,
                    "success_candidate_after": int(j_after_rel) if j_after_rel is not None else None,
                    "top1_hit_after": bool(r_after == 1),
                    "top3_hit_after": bool(r_after is not None and r_after <= 3),
                    "top5_hit_after": bool(r_after is not None and r_after <= 5),
                    "direct_or_gated_recovered": bool(st_only_recovered[ep]),
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
            }
        )
    total = sum(r["episodes"] for r in per_seed)
    agg = {}
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


def choose_gate(train_costs, data, train_seeds):
    thresholds = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    rows = []
    for thr in thresholds:
        agg, _, _, _ = evaluate_costs(train_costs, data, train_seeds, gate_threshold=thr)
        agg["threshold"] = thr
        rows.append(agg)
    safe = [r for r in rows if r["harmed_vs_bsl"] <= 1 and r["switches"] >= 1]
    cand = safe if safe else rows
    return max(cand, key=lambda r: (r["selector_top1"], r["fixed_vs_bsl"] - 3 * r["harmed_vs_bsl"], -r["switches"])), rows


def decompose_fixable_gated(costs, data, seeds, threshold):
    rows = []
    for seed in seeds:
        d = data[seed]
        labels = d["labels"]
        c = costs[seed]
        bsl = d["bsl_labels"][:, 0]
        fixable = (~bsl) & labels.any(axis=1)
        direct = np.argmin(c, axis=1)
        margin = c[:, 0] - c[np.arange(labels.shape[0]), direct]
        for ep in np.where(fixable)[0]:
            if labels[ep, direct[ep]] and direct[ep] != 0 and margin[ep] > threshold:
                cat = "successfully_fixed"
            elif labels[ep, direct[ep]]:
                cat = "selector_correct_but_gate_rejected"
            else:
                cat = "selector_wrong"
            rows.append({"seed": int(seed), "episode": int(ep), "category": cat})
    counts = Counter(r["category"] for r in rows)
    return {"total_fixable": len(rows), "counts": dict(counts), "rows": rows}


def write_summary(out, results):
    lines = ["# C_strict MoDA-Aware Planning Alignment n100", ""]
    lines.append("|split|epoch|train direct|train harm|val direct|val direct harm|gate thr|val gated|val gated harm|st-only recovered gated|st top1 after|st-only top1/3/5|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    per_all = []
    for split, rec in results["splits"].items():
        ch = rec["chosen"]
        va_d = ch["val_direct"]
        va_g = ch["val_gated"]
        per_all.extend(ch["val_gated_per_seed"])
        lines.append(
            f"|{split}|{ch['epoch']}|{ch['train_direct']['selector_top1']:.1f}|{ch['train_direct']['harmed_vs_bsl']}|"
            f"{va_d['selector_top1']:.1f}|{va_d['harmed_vs_bsl']}|{ch['gate_threshold']:.2f}|{va_g['selector_top1']:.1f}|{va_g['harmed_vs_bsl']}|"
            f"{va_g['stateroll_only_recovered']}/{va_g['stateroll_only_fixable']}|{va_d['stateroll_aligned_top1']:.1f}|"
            f"{va_d.get('stateroll_only_top1_hit_after',0)}/{va_d.get('stateroll_only_top3_hit_after',0)}/{va_d.get('stateroll_only_top5_hit_after',0)}|"
        )
    total = sum(r["episodes"] for r in per_all)
    oof = {
        "bsl_top1": sum(r["bsl_top1"] * r["episodes"] / 100 for r in per_all) * 100 / total,
        "selector_top1": sum(r["selector_top1"] * r["episodes"] / 100 for r in per_all) * 100 / total,
        "union_oracle": sum(r["union_oracle"] * r["episodes"] / 100 for r in per_all) * 100 / total,
        "fixed_vs_bsl": sum(r["fixed_vs_bsl"] for r in per_all),
        "harmed_vs_bsl": sum(r["harmed_vs_bsl"] for r in per_all),
        "switches": sum(r["switches"] for r in per_all),
        "stateroll_only_recovered": sum(r["stateroll_only_recovered"] for r in per_all),
        "stateroll_only_fixable": sum(r["stateroll_only_fixable"] for r in per_all),
    }
    results["oof_gated"] = oof
    lines += [
        "",
        f"OOF gated: bsl {oof['bsl_top1']:.1f} -> selector {oof['selector_top1']:.1f}, union oracle {oof['union_oracle']:.1f}, fixed={oof['fixed_vs_bsl']}, harmed={oof['harmed_vs_bsl']}, switches={oof['switches']}, stateroll-only recovered={oof['stateroll_only_recovered']}/{oof['stateroll_only_fixable']}",
    ]
    out.joinpath("summary.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="/data1/jingyixi/wm_runs/moda_c_strict_alignment_n100_20260528")
    ap.add_argument("--seeds", default="42,43,44,45,46,47")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--eval-epochs", default="5,10,20,30")
    ap.add_argument("--batch-pairs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--anchor-weight", type=float, default=5.0)
    ap.add_argument("--st-weight", type=float, default=2.0)
    # inherited data/config defaults
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
    patch_moda_attention_kernel()
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
        base_model = swm.policy.AutoCostModel(POLICIES[args.policy], cache_dir=args.cache_dir).to(device).eval()
        base_model.interpolate_pos_encoding = True
        base_model.requires_grad_(False)
        trainable = set_trainable_c_strict(model)
        opt = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=args.weight_decay)
        buckets, _ = base.build_pair_buckets(data, train_seeds)
        print(f"[{split}] C_strict trainable tensors={len(trainable)} params={sum(p.numel() for _,p in trainable)}", flush=True)
        print(f"[{split}] buckets", {k: len(v) for k, v in buckets.items()}, flush=True)
        records = []
        for epoch in range(1, args.epochs + 1):
            # Keep frozen BatchNorm/Dropout modules in eval mode. Gradients still flow to C_strict trainable params.
            model.eval()
            batch = []
            batch += base.sample_items(rng, buckets["stateroll_unique"], args.batch_pairs // 2)
            batch += base.sample_items(rng, buckets["preserve_pairs"], args.batch_pairs // 4)
            batch += base.sample_items(rng, buckets["generic"], args.batch_pairs - len(batch))
            costs, pred, goal, old_pred = base.rollout_pairs(model, base_model, prepared, data, batch, device)
            pos_cost, neg_cost = costs[:, 0], costs[:, 1]
            rank_loss = F.softplus(0.25 + pos_cost - neg_cost).mean()
            goal_t = base.align_goal(goal, pred)
            curve = (pred - goal_t).pow(2).sum(-1).sqrt()
            goal_loss = F.softplus(0.05 + curve[:, 0, -1] - curve[:, 1, -1]).mean()
            anchor_loss = (pred - old_pred.detach()).pow(2).mean()
            st_mask = torch.as_tensor([1.0 if item in buckets["stateroll_unique"] else 0.0 for item in batch], device=device)
            st_loss = (F.softplus(0.35 + pos_cost - neg_cost) * st_mask).sum() / (st_mask.sum() + 1e-6)
            loss = rank_loss + 0.5 * goal_loss + args.anchor_weight * anchor_loss + args.st_weight * st_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 0.5)
            opt.step()
            model.eval()
            if epoch in eval_epochs:
                train_costs = compute_costs(model, prepared, data, train_seeds, device)
                val_costs = compute_costs(model, prepared, data, val_seeds, device)
                train_direct, train_per, _, train_st = evaluate_costs(train_costs, data, train_seeds, gate_threshold=None)
                val_direct, val_per, val_switches, val_st = evaluate_costs(val_costs, data, val_seeds, gate_threshold=None)
                gate_choice, gate_grid = choose_gate(train_costs, data, train_seeds)
                thr = gate_choice["threshold"]
                val_gated, val_gated_per, val_gated_switches, _ = evaluate_costs(val_costs, data, val_seeds, gate_threshold=thr)
                decomp = decompose_fixable_gated(val_costs, data, val_seeds, thr)
                rec = {
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "rank_loss": float(rank_loss.detach().cpu()),
                    "goal_loss": float(goal_loss.detach().cpu()),
                    "anchor_loss": float(anchor_loss.detach().cpu()),
                    "st_loss": float(st_loss.detach().cpu()),
                    "train_direct": train_direct,
                    "val_direct": val_direct,
                    "gate_threshold": thr,
                    "train_gate_grid": gate_grid,
                    "val_gated": val_gated,
                    "val_gated_per_seed": val_gated_per,
                    "val_decomposition": decomp,
                    "val_stateroll_only_ranks": val_st,
                }
                records.append(rec)
                (split_out / f"record_epoch{epoch}.json").write_text(json.dumps(rec, indent=2))
                torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, split_out / f"checkpoint_epoch{epoch}.pt")
                print(f"[{split} epoch={epoch}] direct val {val_direct['bsl_top1']:.1f}->{val_direct['selector_top1']:.1f} harm={val_direct['harmed_vs_bsl']} st_top1_after={val_direct['stateroll_aligned_top1']:.1f}; gated thr={thr:.2f} val {val_gated['selector_top1']:.1f} harm={val_gated['harmed_vs_bsl']}", flush=True)
        chosen = max(records, key=lambda r: (r["train_direct"]["stateroll_aligned_top5_hit"], r["val_gated"]["selector_top1"], -r["val_gated"]["harmed_vs_bsl"]))
        results["splits"][split] = {"records": records, "chosen": chosen}
    write_summary(out, results)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print((out / "summary.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
