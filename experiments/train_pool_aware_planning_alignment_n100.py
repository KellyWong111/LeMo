from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO = Path("/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "wm_experiment_scripts"))

import analyze_cem_margin as base
from wm_experiment_scripts.pool_coverage_compare_variants import POLICIES


def zscore(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return (x - x.mean(axis=axis, keepdims=True)) / (x.std(axis=axis, keepdims=True) + 1e-6)


def load_src(action_dir: str, raw_dir: str, variant: str, seed: int) -> dict:
    a = np.load(Path(action_dir) / f"{variant}_seed{seed}.npz", allow_pickle=True)
    r = np.load(Path(raw_dir) / f"{variant}_seed{seed}.npz", allow_pickle=True)
    return {
        "actions": a["actions"].astype(np.float32),
        "costs": a["costs"].astype(np.float32),
        "labels": a["labels"].astype(bool),
        "pred": r["pred"].astype(np.float32),
        "goal": r["goal"].astype(np.float32),
        "indices": a["indices"],
    }


def load_union(args, seeds):
    data = {}
    for seed in seeds:
        b = load_src(args.bsl_action_dir, args.bsl_raw_dir, "baseline", seed)
        st = load_src(args.st_action_dir, args.st_raw_dir, "vf05_mix20", seed)
        assert np.all(b["indices"] == st["indices"])
        data[seed] = {
            "actions": np.concatenate([b["actions"], st["actions"]], axis=1),
            "costs": np.concatenate([b["costs"], st["costs"]], axis=1),
            "labels": np.concatenate([b["labels"], st["labels"]], axis=1),
            "bsl_labels": b["labels"],
            "st_labels": st["labels"],
            "bsl_costs": b["costs"],
            "st_costs": st["costs"],
            "indices": b["indices"],
        }
    return data


def build_eval_cfg(args):
    cfg = OmegaConf.load(str(REPO / "config/eval/pusht.yaml"))
    cfg.policy = POLICIES[args.policy]
    cfg.eval.num_eval = args.num_eval
    cfg.cache_dir = args.cache_dir
    OmegaConf.update(cfg, "world.max_episode_steps", 2 * cfg.eval.eval_budget, merge=True)
    OmegaConf.update(cfg, "plan_config.horizon", args.horizon, merge=True)
    OmegaConf.update(cfg, "plan_config.action_block", args.action_block, merge=True)
    OmegaConf.update(cfg, "plan_config.receding_horizon", args.receding_horizon, merge=True)
    return cfg


def prepare_by_seed(cfg, seeds):
    dataset = base.get_dataset(cfg)
    process = base.get_process(cfg, dataset)
    valid_indices = base.get_valid_indices(cfg, dataset)
    transform = {"pixels": base.img_transform(cfg), "goal": base.img_transform(cfg)}
    out = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        picked = np.sort(rng.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False))
        indices = valid_indices[picked]
        raw_info = base.build_info_dict(cfg, dataset, process, indices)
        out[seed] = base.make_eval_like_info(raw_info, transform, process)
    return out


def select_prepared(prepared, eps, topk):
    ep_idx = torch.as_tensor(eps, dtype=torch.long)
    selected = {}
    for k, v in prepared.items():
        selected[k] = v.index_select(0, ep_idx) if torch.is_tensor(v) else v
    return base.expand_info_for_candidates(selected, len(eps), topk)


def set_trainable(model, last_blocks: int):
    model.requires_grad_(False)
    if hasattr(model, "pred_proj"):
        for p in model.pred_proj.parameters():
            p.requires_grad_(True)
    layers = getattr(model.predictor.transformer, "layers", [])
    if last_blocks > 0:
        for block in layers[-last_blocks:]:
            for p in block.parameters():
                p.requires_grad_(True)
        if hasattr(model.predictor.transformer, "norm"):
            for p in model.predictor.transformer.norm.parameters():
                p.requires_grad_(True)
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def build_pair_buckets(data, train_seeds):
    buckets = defaultdict(list)
    preserve = defaultdict(list)
    for seed in train_seeds:
        d = data[seed]
        labels = d["labels"]
        costs = d["costs"]
        for ep in range(labels.shape[0]):
            lab = labels[ep]
            succ = np.where(lab)[0]
            fail = np.where(~lab)[0]
            if d["bsl_labels"][ep, 0]:
                preserve[seed].append((seed, ep))
            if succ.size and fail.size:
                low_fail = [int(j) for j in np.argsort(costs[ep]) if not lab[j]][:5]
                pos_choices = [int(j) for j in succ]
                for pos in pos_choices[: min(4, len(pos_choices))]:
                    for neg in low_fail[:3]:
                        buckets["generic"].append((seed, ep, pos, neg))
                if not d["bsl_labels"][ep, 0]:
                    for pos in pos_choices[: min(4, len(pos_choices))]:
                        buckets["bsl_fail"].append((seed, ep, pos, 0))
            b_oracle = bool(d["bsl_labels"][ep].any())
            st_oracle = bool(d["st_labels"][ep].any())
            if (not d["bsl_labels"][ep, 0]) and st_oracle and (not b_oracle):
                st_succ = np.where(d["st_labels"][ep])[0] + 30
                for pos in st_succ[: min(4, len(st_succ))]:
                    buckets["stateroll_unique"].append((seed, ep, int(pos), 0))
            if d["bsl_labels"][ep, 0] and fail.size:
                low_fail = [int(j) for j in np.argsort(costs[ep]) if not lab[j]][:5]
                for neg in low_fail[:3]:
                    buckets["preserve_pairs"].append((seed, ep, 0, neg))
    return buckets, preserve


def sample_items(rng, items, n):
    if not items or n <= 0:
        return []
    idx = rng.choice(len(items), size=n, replace=len(items) < n)
    return [items[int(i)] for i in idx]


def rollout_pairs(model, base_model, prepared, data, batch, device):
    by_seed = defaultdict(list)
    for item in batch:
        by_seed[item[0]].append(item)
    all_costs, all_pred, all_goal, all_old_pred = [], [], [], []
    for seed, items in by_seed.items():
        eps = [x[1] for x in items]
        cand = np.stack([data[seed]["actions"][ep, [pos, neg]] for _, ep, pos, neg in items])
        cand_t = torch.as_tensor(cand, dtype=torch.float32, device=device)
        info = select_prepared(prepared[seed], eps, 2)
        costs = model.get_cost(info, cand_t)
        pred = info["predicted_emb"]
        goal = info["goal_emb"]
        with torch.no_grad():
            info0 = select_prepared(prepared[seed], eps, 2)
            _ = base_model.get_cost(info0, cand_t)
            old_pred = info0["predicted_emb"].detach()
        all_costs.append(costs)
        all_pred.append(pred)
        all_goal.append(goal)
        all_old_pred.append(old_pred)
    return torch.cat(all_costs), torch.cat(all_pred), torch.cat(all_goal), torch.cat(all_old_pred)


def align_goal(goal, pred):
    if goal.ndim == 3:
        goal = goal.unsqueeze(1)
    if goal.shape[1] == 1:
        goal = goal.expand(-1, pred.shape[1], -1, -1)
    if goal.shape[2] == 1:
        goal = goal.expand(-1, -1, pred.shape[2], -1)
    elif goal.shape[2] != pred.shape[2]:
        goal = goal[:, :, -pred.shape[2] :, :]
    return goal


@torch.no_grad()
def eval_cost_model(model, prepared, data, seeds, device, chunk_episodes=4):
    per_seed = []
    switch_rows = []
    for seed in seeds:
        d = data[seed]
        labels = d["labels"]
        cost_chunks = []
        for start in range(0, labels.shape[0], chunk_episodes):
            end = min(labels.shape[0], start + chunk_episodes)
            cand = torch.as_tensor(d["actions"][start:end], dtype=torch.float32, device=device)
            info = select_prepared(prepared[seed], list(range(start, end)), cand.shape[1])
            cost_chunks.append(model.get_cost(info, cand).detach().cpu().numpy())
        new_cost = np.concatenate(cost_chunks, axis=0)
        picks = np.argmin(new_cost, axis=1)
        succ = labels[np.arange(labels.shape[0]), picks]
        bsl = d["bsl_labels"][:, 0]
        bsl_oracle = d["bsl_labels"].any(axis=1)
        st_oracle = d["st_labels"].any(axis=1)
        union_oracle = labels.any(axis=1)
        st_only_fixable = (~bsl) & st_oracle & (~bsl_oracle)
        recovered_st_only = st_only_fixable & succ
        per_seed.append(
            {
                "seed": seed,
                "episodes": int(labels.shape[0]),
                "bsl_top1": float(bsl.mean() * 100),
                "bsl_oracle": float(bsl_oracle.mean() * 100),
                "stateroll_top1": float(d["st_labels"][:, 0].mean() * 100),
                "stateroll_oracle": float(st_oracle.mean() * 100),
                "union_oracle": float(union_oracle.mean() * 100),
                "aligned_top1": float(succ.mean() * 100),
                "fixed_vs_bsl": int((~bsl & succ).sum()),
                "harmed_vs_bsl": int((bsl & ~succ).sum()),
                "stateroll_only_fixable": int(st_only_fixable.sum()),
                "stateroll_only_recovered": int(recovered_st_only.sum()),
            }
        )
        for ep, pick in enumerate(picks):
            if pick == 0:
                continue
            switch_rows.append(
                {
                    "seed": seed,
                    "episode": ep,
                    "pick": int(pick),
                    "source": "bsl" if pick < 30 else "stateroll",
                    "rank": int(pick if pick < 30 else pick - 30),
                    "bsl_success": bool(bsl[ep]),
                    "selected_success": bool(succ[ep]),
                    "stateroll_only_fixable": bool(st_only_fixable[ep]),
                }
            )
    total = sum(r["episodes"] for r in per_seed)
    agg = {
        "episodes": total,
        "bsl_top1": sum(r["bsl_top1"] * r["episodes"] / 100 for r in per_seed) * 100 / total,
        "bsl_oracle": sum(r["bsl_oracle"] * r["episodes"] / 100 for r in per_seed) * 100 / total,
        "stateroll_top1": sum(r["stateroll_top1"] * r["episodes"] / 100 for r in per_seed) * 100 / total,
        "stateroll_oracle": sum(r["stateroll_oracle"] * r["episodes"] / 100 for r in per_seed) * 100 / total,
        "union_oracle": sum(r["union_oracle"] * r["episodes"] / 100 for r in per_seed) * 100 / total,
        "aligned_top1": sum(r["aligned_top1"] * r["episodes"] / 100 for r in per_seed) * 100 / total,
        "fixed_vs_bsl": sum(r["fixed_vs_bsl"] for r in per_seed),
        "harmed_vs_bsl": sum(r["harmed_vs_bsl"] for r in per_seed),
        "stateroll_only_fixable": sum(r["stateroll_only_fixable"] for r in per_seed),
        "stateroll_only_recovered": sum(r["stateroll_only_recovered"] for r in per_seed),
        "switch_source_counts": dict(Counter(r["source"] for r in switch_rows)),
    }
    return agg, per_seed, switch_rows


def choose_best(train_records):
    safe = [r for r in train_records if r["train"]["harmed_vs_bsl"] <= 1]
    cand = safe if safe else train_records
    return max(cand, key=lambda r: (r["train"]["aligned_top1"], r["train"]["fixed_vs_bsl"] - r["train"]["harmed_vs_bsl"], -r["train"]["harmed_vs_bsl"]))


def write_summary(out: Path, results: dict):
    lines = ["# Pool-Aware Planning Alignment n100", ""]
    lines.append("|split|chosen epoch|train bsl|train aligned|train fix|train harm|val bsl|val aligned|val oracle|val fix|val harm|st-only recovered|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    per_all = []
    for split, rec in results["splits"].items():
        tr = rec["chosen"]["train"]
        va = rec["chosen"]["val"]
        per_all.extend(rec["chosen"]["val_per_seed"])
        lines.append(
            f"|{split}|{rec['chosen']['epoch']}|{tr['bsl_top1']:.1f}|{tr['aligned_top1']:.1f}|{tr['fixed_vs_bsl']}|{tr['harmed_vs_bsl']}|"
            f"{va['bsl_top1']:.1f}|{va['aligned_top1']:.1f}|{va['union_oracle']:.1f}|{va['fixed_vs_bsl']}|{va['harmed_vs_bsl']}|{va['stateroll_only_recovered']}/{va['stateroll_only_fixable']}|"
        )
    total = sum(r["episodes"] for r in per_all)
    oof = {
        "bsl_top1": sum(r["bsl_top1"] * r["episodes"] / 100 for r in per_all) * 100 / total,
        "aligned_top1": sum(r["aligned_top1"] * r["episodes"] / 100 for r in per_all) * 100 / total,
        "union_oracle": sum(r["union_oracle"] * r["episodes"] / 100 for r in per_all) * 100 / total,
        "fixed_vs_bsl": sum(r["fixed_vs_bsl"] for r in per_all),
        "harmed_vs_bsl": sum(r["harmed_vs_bsl"] for r in per_all),
        "stateroll_only_recovered": sum(r["stateroll_only_recovered"] for r in per_all),
        "stateroll_only_fixable": sum(r["stateroll_only_fixable"] for r in per_all),
    }
    results["oof"] = oof
    lines += [
        "",
        f"OOF: bsl {oof['bsl_top1']:.1f} -> aligned-cost {oof['aligned_top1']:.1f}, union oracle {oof['union_oracle']:.1f}, fixed={oof['fixed_vs_bsl']}, harmed={oof['harmed_vs_bsl']}, stateroll-only recovered={oof['stateroll_only_recovered']}/{oof['stateroll_only_fixable']}",
    ]
    out.joinpath("summary.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="/data1/jingyixi/wm_runs/pool_aware_planning_alignment_n100")
    ap.add_argument("--bsl-action-dir", default="/data1/jingyixi/wm_runs/bsl_normalbudget_candidate_pool_s300_steps30_n100/proposal_data")
    ap.add_argument("--bsl-raw-dir", default="/data1/jingyixi/wm_runs/bsl_normalbudget_candidate_pool_s300_steps30_n100/raw_rollout_npz")
    ap.add_argument("--st-action-dir", default="/data1/jingyixi/wm_runs/stateroll_normalbudget_candidate_pool_s300_steps30_n100/proposal_data")
    ap.add_argument("--st-raw-dir", default="/data1/jingyixi/wm_runs/stateroll_normalbudget_candidate_pool_s300_steps30_n100/raw_rollout_npz")
    ap.add_argument("--seeds", default="42,43,44,45,46,47")
    ap.add_argument("--policy", default="stateroll_l003_ep1")
    ap.add_argument("--cache-dir", default="/data1/jingyixi/.stable_worldmodel")
    ap.add_argument("--num-eval", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--action-block", type=int, default=5)
    ap.add_argument("--receding-horizon", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--eval-epochs", default="5,10,20,30,50")
    ap.add_argument("--batch-pairs", type=int, default=24)
    ap.add_argument("--last-blocks", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x]
    eval_epochs = {int(x) for x in args.eval_epochs.split(",") if x}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    data = load_union(args, seeds)
    cfg = build_eval_cfg(args)
    prepared = prepare_by_seed(cfg, seeds)
    splits = {
        "splitA_train42_44_val45_47": ([42, 43, 44], [45, 46, 47]),
        "splitB_train45_47_val42_44": ([45, 46, 47], [42, 43, 44]),
    }
    results = {"args": vars(args), "splits": {}}
    for split_name, (train_seeds, val_seeds) in splits.items():
        split_out = out / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        model = swm.policy.AutoCostModel(POLICIES[args.policy], cache_dir=args.cache_dir).to(device)
        model.interpolate_pos_encoding = True
        base_model = swm.policy.AutoCostModel(POLICIES[args.policy], cache_dir=args.cache_dir).to(device).eval()
        base_model.interpolate_pos_encoding = True
        base_model.requires_grad_(False)
        trainable = set_trainable(model, args.last_blocks)
        opt = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=args.weight_decay)
        buckets, preserve = build_pair_buckets(data, train_seeds)
        print(f"[{split_name}] trainable tensors={len(trainable)} params={sum(p.numel() for _, p in trainable)}", flush=True)
        print(f"[{split_name}] buckets", {k: len(v) for k, v in buckets.items()}, "preserve", {k: len(v) for k, v in preserve.items()}, flush=True)
        records = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            batch = []
            batch += sample_items(rng, buckets["generic"], args.batch_pairs // 4)
            batch += sample_items(rng, buckets["bsl_fail"], args.batch_pairs // 4)
            batch += sample_items(rng, buckets["stateroll_unique"], args.batch_pairs // 3)
            batch += sample_items(rng, buckets["preserve_pairs"], args.batch_pairs - len(batch))
            if not batch:
                raise RuntimeError("empty training batch")
            costs, pred, goal, old_pred = rollout_pairs(model, base_model, prepared, data, batch, device)
            pos_cost = costs[:, 0]
            neg_cost = costs[:, 1]
            rank_loss = F.softplus(0.2 + pos_cost - neg_cost).mean()
            goal_t = align_goal(goal, pred)
            curve = (pred - goal_t).pow(2).sum(-1).sqrt()
            goal_loss = F.softplus(0.05 + curve[:, 0, -1] - curve[:, 1, -1]).mean()
            prog_loss = F.softplus(0.02 + curve[:, 0].mean(-1) - curve[:, 1].mean(-1)).mean()
            anchor_loss = (pred - old_pred.detach()).pow(2).mean()
            # Extra weight for explicitly stateroll-only pairs inside the sampled batch.
            st_mask = torch.as_tensor([1.0 if item in buckets["stateroll_unique"] else 0.0 for item in batch], device=device)
            st_loss = (F.softplus(0.3 + pos_cost - neg_cost) * st_mask).sum() / (st_mask.sum() + 1e-6)
            loss = rank_loss + 0.5 * goal_loss + 0.5 * prog_loss + 1.0 * anchor_loss + 2.0 * st_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
            opt.step()
            model.eval()
            if epoch in eval_epochs:
                train_agg, train_per, _ = eval_cost_model(model, prepared, data, train_seeds, device)
                val_agg, val_per, val_switches = eval_cost_model(model, prepared, data, val_seeds, device)
                rec = {
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "rank_loss": float(rank_loss.detach().cpu()),
                    "goal_loss": float(goal_loss.detach().cpu()),
                    "prog_loss": float(prog_loss.detach().cpu()),
                    "anchor_loss": float(anchor_loss.detach().cpu()),
                    "st_loss": float(st_loss.detach().cpu()),
                    "train": train_agg,
                    "val": val_agg,
                    "train_per_seed": train_per,
                    "val_per_seed": val_per,
                    "val_switches": val_switches,
                }
                records.append(rec)
                (split_out / f"record_epoch{epoch}.json").write_text(json.dumps(rec, indent=2))
                torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, split_out / f"checkpoint_epoch{epoch}.pt")
                print(f"[{split_name} epoch={epoch}] train {train_agg['bsl_top1']:.1f}->{train_agg['aligned_top1']:.1f} harm={train_agg['harmed_vs_bsl']} val {val_agg['bsl_top1']:.1f}->{val_agg['aligned_top1']:.1f} harm={val_agg['harmed_vs_bsl']}", flush=True)
        chosen = choose_best(records)
        results["splits"][split_name] = {"records": records, "chosen": chosen}
        (split_out / "all_records.json").write_text(json.dumps(records, indent=2))
    write_summary(out, results)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print((out / "summary.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
