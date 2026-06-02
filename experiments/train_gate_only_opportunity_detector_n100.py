from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def entropy_from_cost(costs):
    x = -costs.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    p = np.exp(x)
    p = p / (p.sum(axis=-1, keepdims=True) + 1e-12)
    return -(p * np.log(p + 1e-12)).sum(axis=-1)


def load_src(action_dir, raw_dir, variant, seed):
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


def goal_for_pred(goal, pred):
    g = goal
    if g.ndim == 2:
        g = g[:, None, :]
    if g.ndim == 3 and pred.ndim == 4:
        # [E,T,D] or [E,1,D] -> [E,1,T,D] / [E,1,1,D]
        g = g[:, None, :, :]
    if g.shape[1] == 1:
        g = np.repeat(g, pred.shape[1], axis=1)
    if g.shape[2] == 1:
        g = np.repeat(g, pred.shape[2], axis=2)
    elif g.shape[2] != pred.shape[2]:
        g = g[:, :, -pred.shape[2] :, :]
    return g


def traj_stats(pred, goal):
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


def action_stats(actions):
    norm_t = np.sqrt((actions**2).sum(axis=-1))
    return {
        "norm": norm_t.mean(axis=2),
        "std": norm_t.std(axis=2),
        "final_norm": norm_t[:, :, -1],
    }


def safe_margin(sorted_cost, k):
    k = min(k, len(sorted_cost) - 1)
    return float(sorted_cost[k] - sorted_cost[0])


def build_rows(args, seeds):
    rows = []
    for seed in seeds:
        b = load_src(args.bsl_action_dir, args.bsl_raw_dir, "baseline", seed)
        st = load_src(args.st_action_dir, args.st_raw_dir, "vf05_mix20", seed)
        assert np.all(b["indices"] == st["indices"])
        b_tr = traj_stats(b["pred"], b["goal"])
        st_tr = traj_stats(st["pred"], st["goal"])
        b_act = action_stats(b["actions"])
        st_act = action_stats(st["actions"])
        n = b["labels"].shape[0]
        for ep in range(n):
            bc = b["costs"][ep]
            sc = st["costs"][ep]
            bs = np.sort(bc)
            ss = np.sort(sc)
            b_best = int(np.argmin(bc))
            st_best = int(np.argmin(sc))
            bsl_success = bool(b["labels"][ep, 0])
            bsl_oracle = bool(b["labels"][ep].any())
            st_oracle = bool(st["labels"][ep].any())
            union_oracle = bool(bsl_oracle or st_oracle)
            opportunity = (not bsl_success) and st_oracle
            st_only = (not bsl_success) and st_oracle and (not bsl_oracle)
            feat = {
                "bsl_margin_top2": safe_margin(bs, 1),
                "bsl_margin_top5": safe_margin(bs, 4),
                "bsl_margin_top10": safe_margin(bs, 9),
                "bsl_cost_std": float(bc.std()),
                "bsl_cost_entropy": float(entropy_from_cost(bc[None])[0]),
                "bsl_top1_cost": float(bc[0]),
                "bsl_best_cost": float(bs[0]),
                "st_best_cost": float(ss[0]),
                "st_minus_bsl_best_cost": float(ss[0] - bs[0]),
                "st_mean5_minus_bsl_mean5_cost": float(ss[:5].mean() - bs[:5].mean()),
                "bsl_rank0_minus_st_best_final_dist": float(b_tr["final"][ep, 0] - st_tr["final"][ep, st_best]),
                "bsl_best_minus_st_best_final_dist": float(b_tr["final"][ep, b_best] - st_tr["final"][ep, st_best]),
                "bsl_rank0_minus_st_best_mean_dist": float(b_tr["mean"][ep, 0] - st_tr["mean"][ep, st_best]),
                "bsl_rank0_minus_st_best_min_dist": float(b_tr["min"][ep, 0] - st_tr["min"][ep, st_best]),
                "st_best_minus_bsl_rank0_progress": float(st_tr["progress"][ep, st_best] - b_tr["progress"][ep, 0]),
                "bsl_rank0_final_dist": float(b_tr["final"][ep, 0]),
                "bsl_rank0_mean_dist": float(b_tr["mean"][ep, 0]),
                "bsl_rank0_min_dist": float(b_tr["min"][ep, 0]),
                "bsl_rank0_progress": float(b_tr["progress"][ep, 0]),
                "st_best_final_dist": float(st_tr["final"][ep, st_best]),
                "st_best_mean_dist": float(st_tr["mean"][ep, st_best]),
                "st_best_min_dist": float(st_tr["min"][ep, st_best]),
                "st_best_progress": float(st_tr["progress"][ep, st_best]),
                "bsl_action_norm": float(b_act["norm"][ep, 0]),
                "bsl_action_std": float(b_act["std"][ep, 0]),
                "st_best_action_norm": float(st_act["norm"][ep, st_best]),
                "st_best_action_std": float(st_act["std"][ep, st_best]),
                "st_best_rank": float(st_best) / 29.0,
                "bsl_best_rank": float(b_best) / 29.0,
                "bsl_latent_mean": float(b_tr["latent_mean"][ep, 0]),
                "bsl_latent_std": float(b_tr["latent_std"][ep, 0]),
                "st_latent_mean": float(st_tr["latent_mean"][ep, st_best]),
                "st_latent_std": float(st_tr["latent_std"][ep, st_best]),
            }
            rows.append(
                {
                    "seed": int(seed),
                    "episode": int(ep),
                    "bsl_success": bsl_success,
                    "bsl_oracle": bsl_oracle,
                    "st_oracle": st_oracle,
                    "union_oracle": union_oracle,
                    "opportunity": opportunity,
                    "stateroll_only_fixable": st_only,
                    **feat,
                }
            )
    return rows


def make_xy(rows, feature_names):
    x = np.asarray([[r[k] for k in feature_names] for r in rows], dtype=np.float32)
    y = np.asarray([r["opportunity"] for r in rows], dtype=np.int64)
    w = np.ones(len(rows), dtype=np.float32)
    for i, r in enumerate(rows):
        if r["stateroll_only_fixable"]:
            w[i] = 2.5
        elif r["opportunity"]:
            w[i] = 1.5
    return x, y, w


def topk_metrics(rows, scores, ks=(10, 20, 50)):
    order = np.argsort(-scores)
    out = {}
    total_st_only = sum(r["stateroll_only_fixable"] for r in rows)
    total_opp = sum(r["opportunity"] for r in rows)
    for k in ks:
        idx = order[: min(k, len(order))]
        out[f"top{k}_st_only_capture"] = int(sum(rows[i]["stateroll_only_fixable"] for i in idx))
        out[f"top{k}_opportunity_capture"] = int(sum(rows[i]["opportunity"] for i in idx))
        out[f"top{k}_false_positive_bsl_success"] = int(sum(rows[i]["bsl_success"] for i in idx))
        out[f"top{k}_st_only_total"] = int(total_st_only)
        out[f"top{k}_opportunity_total"] = int(total_opp)
    return out


def per_seed_capture(rows, scores, k=50):
    order = np.argsort(-scores)[: min(k, len(scores))]
    selected = set(int(i) for i in order)
    by_seed = {}
    for seed in sorted({r["seed"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["seed"] == seed]
        total = sum(rows[i]["stateroll_only_fixable"] for i in idx)
        cap = sum(rows[i]["stateroll_only_fixable"] for i in idx if i in selected)
        opp_total = sum(rows[i]["opportunity"] for i in idx)
        opp_cap = sum(rows[i]["opportunity"] for i in idx if i in selected)
        by_seed[str(seed)] = {
            "st_only_capture_top50": int(cap),
            "st_only_total": int(total),
            "opportunity_capture_top50": int(opp_cap),
            "opportunity_total": int(opp_total),
        }
    return by_seed


def model_defs(seed=0):
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        "extratrees": ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1),
        "randomforest": RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 64), alpha=1e-3, learning_rate_init=1e-3, max_iter=1000, random_state=seed, early_stopping=True)),
    }


def fit_predict(model, xtr, ytr, wtr, xva):
    try:
        model.fit(xtr, ytr, **({"sample_weight": wtr} if "Pipeline" not in type(model).__name__ else {}))
    except TypeError:
        model.fit(xtr, ytr)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(xva)[:, 1]
    return model.decision_function(xva)


def evaluate_split(name, train_rows, val_rows, feature_names, seed=0):
    xtr, ytr, wtr = make_xy(train_rows, feature_names)
    xva, yva, _ = make_xy(val_rows, feature_names)
    recs = []
    for model_name, model in model_defs(seed).items():
        scores = fit_predict(model, xtr, ytr, wtr, xva)
        auc = float(roc_auc_score(yva, scores)) if len(np.unique(yva)) > 1 else None
        rec = {
            "split": name,
            "model": model_name,
            "auc": auc,
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "pos_train": int(ytr.sum()),
            "pos_val": int(yva.sum()),
            "st_only_val": int(sum(r["stateroll_only_fixable"] for r in val_rows)),
            **topk_metrics(val_rows, scores),
            "per_seed_top50": per_seed_capture(val_rows, scores, k=50),
        }
        recs.append(rec)
    return recs


def flatten_for_csv(records):
    rows = []
    for r in records:
        d = {k: v for k, v in r.items() if k != "per_seed_top50"}
        rows.append(d)
    return rows


def write_csv(path, rows):
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


def write_summary(path, records):
    lines = ["# Gate-Only Opportunity Detector n100", ""]
    lines.append("Previous OS-MoDA-RA learned gate AUC was about 0.41-0.48, with nearly identical gate means across bsl-success/bsl-failure/stateroll-only cases.")
    lines.append("")
    lines.append("|split|model|AUC|top10 st-only|top20 st-only|top50 st-only|top50 opp|top50 FP bsl-success|")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in records:
        auc = "NA" if r["auc"] is None else f"{r['auc']:.3f}"
        lines.append(
            f"|{r['split']}|{r['model']}|{auc}|"
            f"{r['top10_st_only_capture']}/{r['top10_st_only_total']}|"
            f"{r['top20_st_only_capture']}/{r['top20_st_only_total']}|"
            f"{r['top50_st_only_capture']}/{r['top50_st_only_total']}|"
            f"{r['top50_opportunity_capture']}/{r['top50_opportunity_total']}|"
            f"{r['top50_false_positive_bsl_success']}|"
        )
    best = sorted([r for r in records if r["auc"] is not None], key=lambda r: (r["auc"], r["top50_st_only_capture"]), reverse=True)[:5]
    lines.append("")
    lines.append("Best by AUC:")
    for r in best:
        lines.append(f"- {r['split']} {r['model']}: AUC={r['auc']:.3f}, top50 st-only={r['top50_st_only_capture']}/{r['top50_st_only_total']}, FP bsl-success={r['top50_false_positive_bsl_success']}")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="/data1/jingyixi/wm_runs/gate_only_opportunity_detector_n100_20260528")
    ap.add_argument("--seeds", default="42,43,44,45,46,47")
    ap.add_argument("--bsl-action-dir", default="/data1/jingyixi/wm_runs/bsl_normalbudget_candidate_pool_s300_steps30_n100/proposal_data")
    ap.add_argument("--bsl-raw-dir", default="/data1/jingyixi/wm_runs/bsl_normalbudget_candidate_pool_s300_steps30_n100/raw_rollout_npz")
    ap.add_argument("--st-action-dir", default="/data1/jingyixi/wm_runs/stateroll_normalbudget_candidate_pool_s300_steps30_n100/proposal_data")
    ap.add_argument("--st-raw-dir", default="/data1/jingyixi/wm_runs/stateroll_normalbudget_candidate_pool_s300_steps30_n100/raw_rollout_npz")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x]
    rows = build_rows(args, seeds)
    meta = {"total": len(rows), "positive_opportunity": int(sum(r["opportunity"] for r in rows)), "stateroll_only_fixable": int(sum(r["stateroll_only_fixable"] for r in rows)), "bsl_success": int(sum(r["bsl_success"] for r in rows))}
    feature_names = [k for k in rows[0].keys() if k not in {"seed", "episode", "bsl_success", "bsl_oracle", "st_oracle", "union_oracle", "opportunity", "stateroll_only_fixable"}]
    splits = {
        "splitA_train42_44_val45_47": ([42, 43, 44], [45, 46, 47]),
        "splitB_train45_47_val42_44": ([45, 46, 47], [42, 43, 44]),
    }
    records = []
    for name, (tr, va) in splits.items():
        train_rows = [r for r in rows if r["seed"] in tr]
        val_rows = [r for r in rows if r["seed"] in va]
        records.extend(evaluate_split(name, train_rows, val_rows, feature_names, seed=0))
    for held in seeds:
        train_rows = [r for r in rows if r["seed"] != held]
        val_rows = [r for r in rows if r["seed"] == held]
        records.extend(evaluate_split(f"loso_hold{held}", train_rows, val_rows, feature_names, seed=held))
    (out / "gate_only_opportunity_detector_n100.json").write_text(json.dumps({"meta": meta, "features": feature_names, "records": records}, indent=2))
    write_csv(out / "gate_only_opportunity_detector_n100.csv", flatten_for_csv(records))
    write_summary(out / "gate_only_opportunity_detector_n100.md", records)
    print((out / "gate_only_opportunity_detector_n100.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
