from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO = Path("/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "wm_experiment_scripts"))

import stable_worldmodel as swm
import analyze_cem_margin as base
from pool_coverage_compare_variants import POLICIES, clone_prepared
from topk_oracle_pilot import eval_fixed_plans


class ResidualProposalNet(nn.Module):
    def __init__(self, context_dim: int, action_flat_dim: int, hidden: int = 512):
        super().__init__()
        inp = context_dim + action_flat_dim + 2
        self.net = nn.Sequential(
            nn.LayerNorm(inp),
            nn.Linear(inp, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_flat_dim),
        )

    def forward(self, context, action, cost, rank):
        x = torch.cat([context, action, cost[:, None], rank[:, None]], dim=-1)
        return self.net(x)


def cfg_for(policy_name: str, args):
    cfg = OmegaConf.load(str(REPO / "config/eval/pusht.yaml"))
    cfg.policy = POLICIES[policy_name]
    cfg.eval.num_eval = args.num_eval
    cfg.cache_dir = args.cache_dir
    OmegaConf.update(cfg, "world.max_episode_steps", 2 * cfg.eval.eval_budget, merge=True)
    OmegaConf.update(cfg, "plan_config.horizon", args.horizon, merge=True)
    OmegaConf.update(cfg, "plan_config.action_block", args.action_block, merge=True)
    OmegaConf.update(cfg, "plan_config.receding_horizon", args.receding_horizon, merge=True)
    return cfg


def prepare_eval_context(cfg, seed: int):
    dataset = base.get_dataset(cfg)
    process = base.get_process(cfg, dataset)
    valid_indices = base.get_valid_indices(cfg, dataset)
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False))
    indices = valid_indices[picked]
    raw_info = base.build_info_dict(cfg, dataset, process, indices)
    prepared = base.make_eval_like_info(
        raw_info,
        {"pixels": base.img_transform(cfg), "goal": base.img_transform(cfg)},
        process,
    )
    swm.World(**OmegaConf.to_container(cfg.world, resolve=True), image_shape=(224, 224))
    return dataset, process, indices, prepared


def load_data(args):
    data = {}
    for variant in [x for x in args.variants.split(",") if x]:
        for seed in [int(x) for x in args.seeds.split(",") if x]:
            p = Path(args.teacher_dir) / f"{variant}_seed{seed}.npz"
            d = np.load(p, allow_pickle=True)
            data[(variant, seed)] = {k: d[k] for k in d.files}
    return data


def build_pairs(data, keys, max_success_rank):
    pairs = []
    preserve = []
    for key in keys:
        d = data[key]
        labels = d["labels"].astype(bool)
        for ep in range(labels.shape[0]):
            if labels[ep, 0]:
                preserve.append((key, ep, 0))
                continue
            hits = np.where(labels[ep])[0]
            if len(hits) and int(hits[0]) <= max_success_rank:
                succ = int(hits[0])
                pairs.append((key, ep, 0, succ))
    return pairs, preserve


def train_one(args, data, train_keys, ckpt_path):
    first = next(iter(data.values()))
    context_dim = int(first["context"].shape[-1])
    action_flat_dim = int(np.prod(first["actions"].shape[2:]))
    net = ResidualProposalNet(context_dim, action_flat_dim, args.hidden).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pairs, preserve = build_pairs(data, train_keys, args.max_success_rank)
    if not pairs:
        raise RuntimeError("No near-miss pairs for residual proposal training.")
    rng = np.random.default_rng(args.init_seed)
    curves = []
    for epoch in range(1, args.epochs + 1):
        idx = rng.choice(len(pairs), size=min(args.batch_size, len(pairs)), replace=len(pairs) < args.batch_size)
        ctxs, uf, us, costs, ranks = [], [], [], [], []
        for i in idx:
            key, ep, f_rank, s_rank = pairs[int(i)]
            d = data[key]
            ctxs.append(d["context"][ep])
            uf.append(d["actions"][ep, f_rank].reshape(-1))
            us.append(d["actions"][ep, s_rank].reshape(-1))
            costs.append(float(d["costs"][ep, f_rank]))
            ranks.append(float(f_rank) / max(1, d["actions"].shape[1] - 1))
        ctx = torch.tensor(np.stack(ctxs), dtype=torch.float32, device="cuda")
        uf_t = torch.tensor(np.stack(uf), dtype=torch.float32, device="cuda")
        us_t = torch.tensor(np.stack(us), dtype=torch.float32, device="cuda")
        cost_t = torch.tensor(costs, dtype=torch.float32, device="cuda")
        rank_t = torch.tensor(ranks, dtype=torch.float32, device="cuda")
        delta = net(ctx, uf_t, cost_t, rank_t)
        pred = (uf_t + delta).clamp(-1.0, 1.0)
        loss = F.mse_loss(pred, us_t) + args.lambda_mag * delta.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if epoch % args.log_every == 0 or epoch == 1:
            rec = {"epoch": epoch, "loss": float(loss.item()), "pairs": len(pairs)}
            curves.append(rec)
            print("[TRAIN]", ckpt_path.name, rec, flush=True)
    torch.save({"model": net.state_dict(), "context_dim": context_dim, "action_flat_dim": action_flat_dim, "args": vars(args)}, ckpt_path)
    return net, curves


@torch.inference_mode()
def score_candidates(cfg, prepared_base, candidates_np):
    model = base.load_model(cfg, cache_dir=cfg.cache_dir)
    cand = torch.tensor(candidates_np, dtype=torch.float32, device="cuda")
    bsz, ns = cand.shape[:2]
    prepared = clone_prepared(prepared_base)
    prepared = base.expand_info_for_candidates(prepared, bsz, ns)
    costs = model.get_cost(prepared, cand).detach().float().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return costs


def eval_fixed(cfg, dataset, process, indices, candidates_np):
    labels = []
    for rank in range(candidates_np.shape[1]):
        metrics = eval_fixed_plans(cfg, dataset, process, indices, candidates_np[:, rank])
        labels.append(np.asarray(metrics["episode_successes"], dtype=np.bool_))
    return np.stack(labels, axis=1)


def eval_one(args, net, data, variant, seed, mode_tag):
    cfg = cfg_for(args.policy, args)
    d = data[(variant, seed)]
    dataset, process, indices, prepared = prepare_eval_context(cfg, seed)
    labels = d["labels"].astype(bool)
    actions = d["actions"].astype(np.float32)
    costs = d["costs"].astype(np.float32)
    refined = []
    meta = []
    for ep in range(actions.shape[0]):
        ranks = [r for r in range(min(args.refine_topk, actions.shape[1])) if not labels[ep, r]]
        for r in ranks:
            ctx = torch.tensor(d["context"][ep:ep+1], dtype=torch.float32, device="cuda")
            uf = torch.tensor(actions[ep, r].reshape(1, -1), dtype=torch.float32, device="cuda")
            c = torch.tensor([float(costs[ep, r])], dtype=torch.float32, device="cuda")
            rk = torch.tensor([float(r) / max(1, actions.shape[1] - 1)], dtype=torch.float32, device="cuda")
            with torch.no_grad():
                delta = net(ctx, uf, c, rk)
            out = (uf + delta).clamp(-1.0, 1.0).cpu().numpy().reshape(actions.shape[2:])
            refined.append((ep, out))
            meta.append((ep, r))
    n_ref = max(1, args.refine_topk)
    refined_by_ep = [[] for _ in range(actions.shape[0])]
    for ep, act in refined:
        refined_by_ep[ep].append(act)
    refined_arr = np.zeros((actions.shape[0], n_ref, *actions.shape[2:]), dtype=np.float32)
    for ep in range(actions.shape[0]):
        vals = refined_by_ep[ep][:n_ref]
        if not vals:
            vals = [actions[ep, 0]] * n_ref
        while len(vals) < n_ref:
            vals.append(vals[-1])
        refined_arr[ep] = np.stack(vals[:n_ref])
    refined_cost = score_candidates(cfg, prepared, refined_arr)
    refined_order = np.argsort(refined_cost, axis=1)
    refined_sorted = refined_arr[np.arange(refined_arr.shape[0])[:, None], refined_order]
    refined_labels = eval_fixed(cfg, dataset, process, indices, refined_sorted)

    combined_actions = np.concatenate([actions, refined_sorted], axis=1)
    combined_costs = np.concatenate([costs, np.take_along_axis(refined_cost, refined_order, axis=1)], axis=1)
    order = np.argsort(combined_costs, axis=1)[:, :args.topk]
    mixed_actions = combined_actions[np.arange(combined_actions.shape[0])[:, None], order]
    mixed_labels = eval_fixed(cfg, dataset, process, indices, mixed_actions)

    fixed = harmed = 0
    for ep in range(labels.shape[0]):
        fixed += (not labels[ep, 0]) and bool(mixed_labels[ep, 0])
        harmed += bool(labels[ep, 0]) and (not bool(mixed_labels[ep, 0]))
    return {
        "mode": mode_tag,
        "variant": variant,
        "seed": seed,
        "old_top1": float(labels[:, 0].mean() * 100.0),
        "old_oracle": float(labels.any(axis=1).mean() * 100.0),
        "residual_gen_top1": float(refined_labels[:, 0].mean() * 100.0),
        "residual_gen_oracle": float(refined_labels.any(axis=1).mean() * 100.0),
        "mixed_top1": float(mixed_labels[:, 0].mean() * 100.0),
        "mixed_oracle": float(mixed_labels.any(axis=1).mean() * 100.0),
        "fixed_wrong": int(fixed),
        "harmed_correct": int(harmed),
    }


def summarize(rows):
    by = {}
    for r in rows:
        by.setdefault((r["mode"], r["variant"]), []).append(r)
    lines = [
        "# Residual Near-Miss Proposal",
        "",
        "|mode|variant|n|old_top1|old_oracle|res_gen_top1|res_gen_oracle|mixed_top1|mixed_oracle|fixed|harmed|net_gain|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (mode, variant), items in sorted(by.items()):
        def mean(k):
            return float(np.mean([x[k] for x in items]))
        fixed = int(np.sum([x["fixed_wrong"] for x in items]))
        harmed = int(np.sum([x["harmed_correct"] for x in items]))
        lines.append(
            f"|{mode}|{variant}|{len(items)}|{mean('old_top1'):.1f}|{mean('old_oracle'):.1f}|"
            f"{mean('residual_gen_top1'):.1f}|{mean('residual_gen_oracle'):.1f}|{mean('mixed_top1'):.1f}|{mean('mixed_oracle'):.1f}|"
            f"{fixed}|{harmed}|{mean('mixed_top1') - mean('old_top1'):.1f}|"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/data1/jingyixi/wm_runs/rpn_residual_proposal")
    parser.add_argument("--teacher-dir", default="/data1/jingyixi/wm_runs/action_proposal_rpn/proposal_data")
    parser.add_argument("--cache-dir", default=os.environ.get("STABLEWM_HOME", "/data1/jingyixi/.stable_worldmodel"))
    parser.add_argument("--policy", default="stateroll_l003_ep1")
    parser.add_argument("--variants", default="vf05_mix20,vf05,vf03_mix20")
    parser.add_argument("--seeds", default="42,43,44,45,46,47")
    parser.add_argument("--num-eval", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=4)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--refine-topk", type=int, default=5)
    parser.add_argument("--max-success-rank", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--lambda-mag", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--mode", choices=["combined", "per_variant"], default="combined")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(args)
    variants = [x for x in args.variants.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    rows, curves = [], {}
    jobs = [("combined", None, held) for held in seeds] if args.mode == "combined" else [(v, v, held) for v in variants for held in seeds]
    for tag, train_variant, held_seed in jobs:
        train_keys = [(v, s) for v in variants for s in seeds if s != held_seed and (train_variant is None or v == train_variant)]
        ckpt = out / f"{tag}_holdout_seed{held_seed}.pt"
        net, curve = train_one(args, data, train_keys, ckpt)
        curves[f"{tag}_holdout_seed{held_seed}"] = curve
        eval_variants = variants if train_variant is None else [train_variant]
        for variant in eval_variants:
            print("[EVAL]", tag, variant, held_seed, flush=True)
            rows.append(eval_one(args, net, data, variant, held_seed, args.mode))
            (out / f"results_{args.mode}.json").write_text(json.dumps(rows, indent=2))
            (out / f"summary_{args.mode}.md").write_text(summarize(rows))
    (out / f"curves_{args.mode}.json").write_text(json.dumps(curves, indent=2))
    print(summarize(rows), flush=True)


if __name__ == "__main__":
    main()
