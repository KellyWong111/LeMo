from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


def zscore(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return (x - x.mean(axis=axis, keepdims=True)) / (x.std(axis=axis, keepdims=True) + 1e-6)


def entropy_from_cost(costs: np.ndarray) -> float:
    x = -costs.astype(np.float64)
    x = x - x.max()
    p = np.exp(x)
    p = p / (p.sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())


def load_env(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def union_env(b: dict, s: dict) -> dict:
    out = {}
    cat1 = [
        "labels", "replay_success", "costs", "actions", "agent_xy", "object_xy", "object_angle",
        "goal_xy", "goal_angle", "distance_curve", "angle_error_curve", "progress_curve",
        "contact_curve", "final_distance", "final_angle_error", "min_distance", "final_progress",
        "max_progress", "trajectory_smoothness", "object_path_len", "contact_proxy",
        "agent_object_min_dist",
    ]
    for k in cat1:
        out[k] = np.concatenate([b[k], s[k]], axis=1)
    out["indices"] = b["indices"]
    out["source"] = np.concatenate([np.zeros(30, dtype=np.float32), np.ones(30, dtype=np.float32)])
    out["bsl_labels"] = b["labels"].astype(bool)
    return out


def build_episode_context(b: dict, s: dict, ep: int) -> np.ndarray:
    def one(pool):
        c = pool["costs"][ep].astype(np.float32)
        dc = pool["distance_curve"][ep]
        pc = pool["progress_curve"][ep]
        return [
            float(c[0]),
            float(c[1] - c[0]),
            float(c[4] - c[0]),
            float(c[9] - c[0]),
            float(np.std(c)),
            entropy_from_cost(c),
            float(pool["final_distance"][ep, 0]),
            float(pool["final_angle_error"][ep, 0]),
            float(pool["min_distance"][ep, 0]),
            float(pool["final_progress"][ep, 0]),
            float(pool["max_progress"][ep, 0]),
            float(pool["contact_proxy"][ep, 0]),
            float(pool["agent_object_min_dist"][ep, 0]),
            float(dc[0, -1] - dc[0, 0]),
            float(pc[0].max()),
        ]
    bx = one(b)
    sx = one(s)
    return np.asarray(bx + sx + [sx[1] - bx[1], sx[6] - bx[6], sx[9] - bx[9], sx[11] - bx[11]], dtype=np.float32)


def curve_sample(x: np.ndarray, n: int = 16) -> np.ndarray:
    idx = np.linspace(0, x.shape[-1] - 1, n).round().astype(int)
    return x[..., idx]


def build_dataset(args, seeds: list[int]) -> dict:
    episodes = []
    for seed in seeds:
        b = load_env(Path(args.env_dir) / f"baseline_seed{seed}.npz")
        s = load_env(Path(args.env_dir) / f"vf05_mix20_seed{seed}.npz")
        assert np.all(b["indices"] == s["indices"])
        u = union_env(b, s)
        n_ep, n_c = u["labels"].shape
        costs = u["costs"].astype(np.float32)
        zc_all = zscore(costs, axis=1)
        zc_src = np.zeros_like(costs)
        zc_src[:, :30] = zscore(costs[:, :30], axis=1)
        zc_src[:, 30:] = zscore(costs[:, 30:], axis=1)
        source = np.tile(u["source"][None, :], (n_ep, 1))
        rank = np.tile(np.arange(n_c, dtype=np.float32)[None, :], (n_ep, 1)) / float(n_c - 1)
        action_norm = np.sqrt((u["actions"] ** 2).sum(axis=(2, 3)))
        action_smooth = np.sqrt((np.diff(u["actions"], axis=2) ** 2).sum(axis=(2, 3)))
        for ep in range(n_ep):
            labels = u["labels"][ep].astype(bool)
            bsl_success = bool(u["bsl_labels"][ep, 0])
            bsl_scalar = np.asarray([
                source[ep, 0], rank[ep, 0], costs[ep, 0], zc_all[ep, 0], zc_src[ep, 0],
                u["final_distance"][ep, 0], u["final_angle_error"][ep, 0], u["min_distance"][ep, 0],
                u["final_progress"][ep, 0], u["max_progress"][ep, 0], u["trajectory_smoothness"][ep, 0],
                u["object_path_len"][ep, 0], u["contact_proxy"][ep, 0], u["agent_object_min_dist"][ep, 0],
                action_norm[ep, 0], action_smooth[ep, 0],
            ], dtype=np.float32)
            ep_ctx = build_episode_context(b, s, ep)
            scalars, seqs = [], []
            b_obj0 = u["object_xy"][ep, 0]
            b_dist0 = curve_sample(u["distance_curve"][ep, 0])
            b_ang0 = curve_sample(u["angle_error_curve"][ep, 0])
            b_prog0 = curve_sample(u["progress_curve"][ep, 0])
            b_contact0 = curve_sample(u["contact_curve"][ep, 0])
            for j in range(n_c):
                c = np.asarray([
                    source[ep, j], rank[ep, j], costs[ep, j], zc_all[ep, j], zc_src[ep, j],
                    u["final_distance"][ep, j], u["final_angle_error"][ep, j], u["min_distance"][ep, j],
                    u["final_progress"][ep, j], u["max_progress"][ep, j], u["trajectory_smoothness"][ep, j],
                    u["object_path_len"][ep, j], u["contact_proxy"][ep, j], u["agent_object_min_dist"][ep, j],
                    action_norm[ep, j], action_smooth[ep, j],
                ], dtype=np.float32)
                rel = c - bsl_scalar
                goal_xy = u["goal_xy"][ep, j]
                obj = curve_sample(u["object_xy"][ep, j].transpose(1, 0)).transpose(1, 0)
                agent = curve_sample(u["agent_xy"][ep, j].transpose(1, 0)).transpose(1, 0)
                obj_rel_goal = obj - goal_xy[None, :]
                b_obj_rel_goal = curve_sample((b_obj0 - u["goal_xy"][ep, 0]).transpose(1, 0)).transpose(1, 0)
                disp = obj[-1] - obj[0]
                bdisp = b_obj0[-1] - b_obj0[0]
                seq = np.concatenate([
                    obj_rel_goal.reshape(-1),
                    b_obj_rel_goal.reshape(-1),
                    (obj_rel_goal - b_obj_rel_goal).reshape(-1),
                    (agent - obj).reshape(-1),
                    curve_sample(u["distance_curve"][ep, j]),
                    b_dist0,
                    curve_sample(u["distance_curve"][ep, j]) - b_dist0,
                    curve_sample(u["angle_error_curve"][ep, j]),
                    b_ang0,
                    curve_sample(u["progress_curve"][ep, j]),
                    b_prog0,
                    curve_sample(u["progress_curve"][ep, j]) - b_prog0,
                    curve_sample(u["contact_curve"][ep, j]),
                    b_contact0,
                    disp,
                    bdisp,
                    disp - bdisp,
                ]).astype(np.float32)
                scalars.append(np.concatenate([c, bsl_scalar, rel, ep_ctx]).astype(np.float32))
                seqs.append(seq)
            episodes.append({
                "seed": seed,
                "episode": ep,
                "scalar": np.stack(scalars).astype(np.float32),
                "seq": np.stack(seqs).astype(np.float32),
                "labels": labels,
                "bsl_success": bsl_success,
                "fixable": (not bsl_success) and bool(labels.any()),
                "unfixable": (not bsl_success) and not bool(labels.any()),
            })
    return {"episodes": episodes, "scalar_dim": int(episodes[0]["scalar"].shape[1]), "seq_dim": int(episodes[0]["seq"].shape[1])}


def split_indices(dataset, train_seeds, val_seeds):
    train, val = [], []
    for i, ep in enumerate(dataset["episodes"]):
        if ep["seed"] in train_seeds:
            train.append(i)
        elif ep["seed"] in val_seeds:
            val.append(i)
    return train, val


def detector_xy(dataset, indices):
    xs, ys = [], []
    for i in indices:
        ep = dataset["episodes"][i]
        xs.append(np.concatenate([ep["scalar"][0, 48:], ep["seq"][0, :96]]))
        ys.append(1 if ep["fixable"] else 0)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def train_detector(dataset, train_idx, method, seed):
    x, y = detector_xy(dataset, train_idx)
    if method == "logreg":
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, random_state=seed))
    else:
        clf = ExtraTreesClassifier(n_estimators=600, max_depth=4, min_samples_leaf=4, class_weight="balanced", random_state=seed, n_jobs=-1)
    clf.fit(x, y)
    return clf


def detector_scores(clf, dataset, indices):
    x, _ = detector_xy(dataset, indices)
    return clf.predict_proba(x)[:, 1]


class EnvReplacementScorer(nn.Module):
    def __init__(self, scalar_dim: int, seq_dim: int, dropout: float):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(seq_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.scalar = nn.Sequential(
            nn.Linear(scalar_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, scalar, seq):
        return self.head(torch.cat([self.scalar(scalar), self.seq(seq)], dim=-1)).squeeze(-1)


def train_one(args, dataset, train_idx, seed: int, out: Path):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = EnvReplacementScorer(dataset["scalar_dim"], dataset["seq_dim"], args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    repair = [i for i in train_idx if dataset["episodes"][i]["fixable"]]
    preserve = [i for i in train_idx if dataset["episodes"][i]["bsl_success"]]
    unfix = [i for i in train_idx if dataset["episodes"][i]["unfixable"]]
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    eval_epochs = {int(x) for x in args.eval_epochs.split(",") if x}
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        steps = max(1, math.ceil(len(train_idx) / args.batch_episodes))
        for _ in range(steps):
            batch = []
            nr = max(1, int(args.batch_episodes * 0.55))
            npres = max(1, int(args.batch_episodes * 0.35))
            nun = max(0, args.batch_episodes - nr - npres)
            batch += rng.choice(repair, nr, replace=len(repair) < nr).tolist()
            batch += rng.choice(preserve, npres, replace=len(preserve) < npres).tolist()
            if unfix and nun:
                batch += rng.choice(unfix, nun, replace=len(unfix) < nun).tolist()
            rng.shuffle(batch)
            terms = []
            for i in batch:
                ep = dataset["episodes"][i]
                scalar = torch.as_tensor(ep["scalar"], dtype=torch.float32, device=device)
                seq = torch.as_tensor(ep["seq"], dtype=torch.float32, device=device)
                scores = model(scalar, seq)
                if ep["fixable"]:
                    pos = torch.as_tensor(np.where(ep["labels"])[0], dtype=torch.long, device=device)
                    neg_np = np.where(~ep["labels"])[0]
                    if len(neg_np) == 0:
                        continue
                    hard = [0] + [int(x) for x in neg_np[: min(10, len(neg_np))] if int(x) != 0]
                    neg = torch.as_tensor(hard, dtype=torch.long, device=device)
                    pos_score = torch.logsumexp(scores[pos] / args.temp, dim=0) * args.temp
                    neg_score = torch.logsumexp(scores[neg] / args.temp, dim=0) * args.temp
                    terms.append(args.repair_weight * F.softplus(args.margin + neg_score - pos_score))
                elif ep["bsl_success"]:
                    # Preserve only as a light safety term; detector and threshold handle conservativeness.
                    target = torch.zeros_like(scores[1:])
                    terms.append(args.preserve_weight * F.binary_cross_entropy_with_logits(scores[1:], target))
                else:
                    target = torch.zeros_like(scores[1:])
                    terms.append(args.unfixable_weight * F.binary_cross_entropy_with_logits(scores[1:], target))
            if not terms:
                continue
            loss = torch.stack(terms).mean() + args.l2_score * (scores ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch in eval_epochs:
            ckpt = {"model": model.state_dict(), "scalar_dim": dataset["scalar_dim"], "seq_dim": dataset["seq_dim"], "epoch": epoch, "seed": seed, "args": vars(args)}
            torch.save(ckpt, out / f"checkpoint_epoch{epoch}.pt")
            hist.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else None})
            (out / "history.json").write_text(json.dumps(hist, indent=2))
            print(f"seed={seed} epoch={epoch} loss={hist[-1]['loss']}", flush=True)


@torch.no_grad()
def score_checkpoint(path: Path, dataset, indices, args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(path, map_location=device)
    model = EnvReplacementScorer(ckpt["scalar_dim"], ckpt["seq_dim"], args.dropout).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    rows = []
    for i in indices:
        ep = dataset["episodes"][i]
        scalar = torch.as_tensor(ep["scalar"], dtype=torch.float32, device=device)
        seq = torch.as_tensor(ep["seq"], dtype=torch.float32, device=device)
        rows.append(model(scalar, seq).cpu().numpy())
    return np.stack(rows)


def eval_grid(dataset, indices, det, score_sets, topks, thresholds, vote_ks, max_switch_fracs):
    labels = np.stack([dataset["episodes"][i]["labels"] for i in indices])
    bsl = np.asarray([dataset["episodes"][i]["bsl_success"] for i in indices], dtype=bool)
    seeds = np.asarray([dataset["episodes"][i]["seed"] for i in indices], dtype=int)
    oracle = labels.any(axis=1)
    score_mean = np.mean(score_sets, axis=0)
    order = np.argsort(-det)
    rows = []
    for topk in topks:
        allowed = np.zeros(len(indices), dtype=bool)
        allowed[order[: min(topk, len(indices))]] = True
        for thr in thresholds:
            votes = np.stack([(s[:, 1:].max(axis=1) > thr) for s in score_sets], axis=0).sum(axis=0)
            for vk in vote_ks:
                for msf in max_switch_fracs:
                    cand = []
                    for ei in range(len(indices)):
                        if not allowed[ei] or votes[ei] < vk:
                            continue
                        j = int(score_mean[ei, 1:].argmax() + 1)
                        sc = float(score_mean[ei, j])
                        if sc > thr:
                            cand.append((sc, ei, j))
                    cand.sort(reverse=True)
                    max_sw = max(1, int(round(len(indices) * msf)))
                    pick = np.zeros(len(indices), dtype=int)
                    for _, ei, j in cand[:max_sw]:
                        pick[ei] = j
                    succ = labels[np.arange(len(indices)), pick]
                    per = []
                    for sd in sorted(set(seeds.tolist())):
                        m = seeds == sd
                        per.append({
                            "seed": int(sd),
                            "episodes": int(m.sum()),
                            "bsl_top1": float(bsl[m].mean() * 100),
                            "selector_top1": float(succ[m].mean() * 100),
                            "oracle": float(oracle[m].mean() * 100),
                            "fixed_vs_bsl": int((~bsl[m] & succ[m]).sum()),
                            "harmed_vs_bsl": int((bsl[m] & ~succ[m]).sum()),
                            "switches": int((pick[m] != 0).sum()),
                        })
                    rows.append({
                        "detector_topk": int(topk),
                        "score_threshold": float(thr),
                        "vote_k": int(vk),
                        "max_switch_frac": float(msf),
                        "bsl_top1": float(bsl.mean() * 100),
                        "selector_top1": float(succ.mean() * 100),
                        "oracle": float(oracle.mean() * 100),
                        "fixed_vs_bsl": int((~bsl & succ).sum()),
                        "harmed_vs_bsl": int((bsl & ~succ).sum()),
                        "switches": int((pick != 0).sum()),
                        "detector_fixable_captured": int(sum(dataset["episodes"][indices[i]]["fixable"] for i in np.where(allowed)[0])),
                        "per_seed": per,
                    })
    return rows


def choose(rows, harmed_limit, min_switches):
    safe = [r for r in rows if r["harmed_vs_bsl"] <= harmed_limit and r["switches"] >= min_switches]
    if not safe:
        safe = [r for r in rows if r["harmed_vs_bsl"] <= harmed_limit]
    cand = safe if safe else rows
    return max(cand, key=lambda r: (r["fixed_vs_bsl"] - 3 * r["harmed_vs_bsl"], r["selector_top1"], -r["switches"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-dir", default="/data1/jingyixi/wm_runs/env_traj_features_n100")
    ap.add_argument("--output-dir", default="/data1/jingyixi/wm_runs/env_traj_replacement_n100_20260527")
    ap.add_argument("--seeds", default="42,43,44,45,46,47")
    ap.add_argument("--model-seeds", default="0,1,2,3")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--eval-epochs", default="5,10,20,30")
    ap.add_argument("--batch-episodes", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--repair-weight", type=float, default=2.0)
    ap.add_argument("--preserve-weight", type=float, default=0.5)
    ap.add_argument("--unfixable-weight", type=float, default=0.25)
    ap.add_argument("--l2-score", type=float, default=1e-4)
    ap.add_argument("--topks", default="10,20,30,40,50,70,90")
    ap.add_argument("--thresholds", default="-1.0,-0.5,0.0,0.5,1.0,1.5,2.0")
    ap.add_argument("--vote-ks", default="1,2,3,4")
    ap.add_argument("--max-switch-fracs", default="0.05,0.10,0.15,0.20")
    ap.add_argument("--harmed-limit", type=int, default=2)
    ap.add_argument("--min-switches", type=int, default=1)
    ap.add_argument("--detector", default="extratrees", choices=["extratrees", "logreg"])
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(args, [int(x) for x in args.seeds.split(",") if x])
    splits = {
        "splitA_train42_44_val45_47": ([42, 43, 44], [45, 46, 47]),
        "splitB_train45_47_val42_44": ([45, 46, 47], [42, 43, 44]),
    }
    topks = [int(x) for x in args.topks.split(",") if x]
    thresholds = [float(x) for x in args.thresholds.split(",") if x]
    vote_ks = [int(x) for x in args.vote_ks.split(",") if x]
    max_switch_fracs = [float(x) for x in args.max_switch_fracs.split(",") if x]
    model_seeds = [int(x) for x in args.model_seeds.split(",") if x]
    results = {}
    for split, (tr_seeds, va_seeds) in splits.items():
        split_out = out / split
        tr_idx, va_idx = split_indices(dataset, tr_seeds, va_seeds)
        clf = train_detector(dataset, tr_idx, args.detector, seed=0)
        det_tr = detector_scores(clf, dataset, tr_idx)
        det_va = detector_scores(clf, dataset, va_idx)
        try:
            dx, dy = detector_xy(dataset, va_idx)
            auc = float(roc_auc_score(dy, det_va))
        except Exception:
            auc = None
        for ms in model_seeds:
            train_one(args, dataset, tr_idx, ms, split_out / f"critic_seed{ms}")
        best = None
        for epoch in [int(x) for x in args.eval_epochs.split(",") if x]:
            paths = [split_out / f"critic_seed{ms}" / f"checkpoint_epoch{epoch}.pt" for ms in model_seeds]
            tr_scores = [score_checkpoint(p, dataset, tr_idx, args) for p in paths]
            va_scores = [score_checkpoint(p, dataset, va_idx, args) for p in paths]
            tr_rows = eval_grid(dataset, tr_idx, det_tr, tr_scores, topks, thresholds, vote_ks, max_switch_fracs)
            va_rows = eval_grid(dataset, va_idx, det_va, va_scores, topks, thresholds, vote_ks, max_switch_fracs)
            chosen_tr = choose(tr_rows, args.harmed_limit, args.min_switches)
            chosen_va = [r for r in va_rows if all([
                r["detector_topk"] == chosen_tr["detector_topk"],
                r["score_threshold"] == chosen_tr["score_threshold"],
                r["vote_k"] == chosen_tr["vote_k"],
                r["max_switch_frac"] == chosen_tr["max_switch_frac"],
            ])][0]
            rec = {"epoch": epoch, "train": chosen_tr, "val": chosen_va}
            if best is None or (chosen_tr["fixed_vs_bsl"] - 3 * chosen_tr["harmed_vs_bsl"], chosen_tr["selector_top1"]) > (best["train"]["fixed_vs_bsl"] - 3 * best["train"]["harmed_vs_bsl"], best["train"]["selector_top1"]):
                best = rec
            (split_out / f"train_grid_epoch{epoch}.json").write_text(json.dumps(tr_rows, indent=2))
            (split_out / f"val_grid_epoch{epoch}.json").write_text(json.dumps(va_rows, indent=2))
        results[split] = {"train_seeds": tr_seeds, "val_seeds": va_seeds, "detector_val_auc": auc, "best": best}
        print(split, json.dumps({"auc": auc, "best_train": {k: v for k, v in best["train"].items() if k != "per_seed"}, "best_val": {k: v for k, v in best["val"].items() if k != "per_seed"}}, indent=2), flush=True)

    (out / "results.json").write_text(json.dumps(results, indent=2))
    per_all = []
    lines = ["# Env-Trajectory Replacement n100", ""]
    lines.append("|split|auc|epoch|train bsl|train sel|train fix|train harm|train sw|topk|thr|vote|maxsw|val bsl|val sel|val oracle|val fix|val harm|val sw|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split, rec in results.items():
        tr, va = rec["best"]["train"], rec["best"]["val"]
        per_all.extend(va["per_seed"])
        lines.append(
            f"|{split}|{rec['detector_val_auc'] if rec['detector_val_auc'] is not None else 'NA'}|{rec['best']['epoch']}|"
            f"{tr['bsl_top1']:.1f}|{tr['selector_top1']:.1f}|{tr['fixed_vs_bsl']}|{tr['harmed_vs_bsl']}|{tr['switches']}|"
            f"{tr['detector_topk']}|{tr['score_threshold']:.2f}|{tr['vote_k']}|{tr['max_switch_frac']:.2f}|"
            f"{va['bsl_top1']:.1f}|{va['selector_top1']:.1f}|{va['oracle']:.1f}|{va['fixed_vs_bsl']}|{va['harmed_vs_bsl']}|{va['switches']}|"
        )
    total = sum(r["episodes"] for r in per_all)
    bsl = sum(r["bsl_top1"] * r["episodes"] / 100 for r in per_all)
    sel = sum(r["selector_top1"] * r["episodes"] / 100 for r in per_all)
    oracle = sum(r["oracle"] * r["episodes"] / 100 for r in per_all)
    fixed = sum(r["fixed_vs_bsl"] for r in per_all)
    harmed = sum(r["harmed_vs_bsl"] for r in per_all)
    switches = sum(r["switches"] for r in per_all)
    lines.append("")
    lines.append(f"OOF aggregate: bsl {100*bsl/total:.1f} -> selector {100*sel/total:.1f}, oracle {100*oracle/total:.1f}, fixed={fixed}, harmed={harmed}, switches={switches}")
    lines.append("")
    lines.append("Per seed:")
    lines.append("```json")
    lines.append(json.dumps(per_all, indent=2))
    lines.append("```")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
