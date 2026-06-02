from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("/data1/jingyixi/wm_runs")
OUT = ROOT / "risk_controlled_moda_integration_20260530"
BSL_ACTION = ROOT / "bsl_normalbudget_candidate_pool_s300_steps30_n100" / "proposal_data"
BSL_RAW = ROOT / "bsl_normalbudget_candidate_pool_s300_steps30_n100" / "raw_rollout_npz"
ST_ACTION = ROOT / "stateroll_normalbudget_candidate_pool_s300_steps30_n100" / "proposal_data"
ST_RAW = ROOT / "stateroll_normalbudget_candidate_pool_s300_steps30_n100" / "raw_rollout_npz"
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


def entropy_from_cost(costs: np.ndarray) -> float:
    x = -costs.astype(np.float64)
    x = x - x.max()
    p = np.exp(x)
    p = p / (p.sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())


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
    norm = np.sqrt((actions**2).sum(axis=-1))
    flat = actions.reshape(actions.shape[0], actions.shape[1], -1)
    return {
        "norm_mean": norm.mean(axis=2),
        "norm_std": norm.std(axis=2),
        "diversity": flat.std(axis=1).mean(axis=1),
    }


def margin(sorted_costs: np.ndarray, k: int) -> float:
    return float(sorted_costs[min(k, len(sorted_costs) - 1)] - sorted_costs[0])


def build_rows() -> list[dict]:
    rows = []
    for seed in SEEDS:
        b = load_src(BSL_ACTION, BSL_RAW, "baseline", seed)
        st = load_src(ST_ACTION, ST_RAW, "vf05_mix20", seed)
        assert np.all(b["indices"] == st["indices"])
        btr, strj = traj_stats(b["pred"], b["goal"]), traj_stats(st["pred"], st["goal"])
        bact, sact = action_stats(b["actions"]), action_stats(st["actions"])
        for ep in range(b["labels"].shape[0]):
            bc, sc = b["costs"][ep], st["costs"][ep]
            bo, so = np.argsort(bc, kind="stable"), np.argsort(sc, kind="stable")
            bs, ss = bc[bo], sc[so]
            b0, s0 = int(bo[0]), int(so[0])
            bsl_success = bool(b["labels"][ep, 0])
            st_top1_success = bool(st["labels"][ep, s0])
            st_top3_success = bool(st["labels"][ep, so[:3]].any())
            st_top5_success = bool(st["labels"][ep, so[:5]].any())
            st_top10_success = bool(st["labels"][ep, so[:10]].any())
            st_top30_success = bool(st["labels"][ep].any())
            row = {
                "seed": seed,
                "episode": ep,
                "dataset_index": int(b["indices"][ep]),
                "bsl_success": bsl_success,
                "st_top1_success": st_top1_success,
                "st_top3_success": st_top3_success,
                "st_top5_success": st_top5_success,
                "st_top10_success": st_top10_success,
                "st_top30_success": st_top30_success,
                "fix_raw_top1": (not bsl_success) and st_top1_success,
                "fix_top10_opportunity": (not bsl_success) and st_top10_success,
                "harm_raw_top1": bsl_success and (not st_top1_success),
                "bsl_top1_cost": float(bc[0]),
                "st_top1_cost": float(sc[s0]),
                "best_moda_minus_bsl_cost": float(sc[s0] - bc[0]),
                "bsl_margin_top2": margin(bs, 1),
                "bsl_margin_top5": margin(bs, 4),
                "bsl_margin_top10": margin(bs, 9),
                "st_margin_top2": margin(ss, 1),
                "st_margin_top5": margin(ss, 4),
                "st_margin_top10": margin(ss, 9),
                "margin5_disagreement": margin(ss, 4) - margin(bs, 4),
                "margin10_disagreement": margin(ss, 9) - margin(bs, 9),
                "bsl_cost_mean": float(bc.mean()),
                "bsl_cost_std": float(bc.std()),
                "bsl_cost_entropy": entropy_from_cost(bc),
                "st_cost_mean": float(sc.mean()),
                "st_cost_std": float(sc.std()),
                "st_cost_entropy": entropy_from_cost(sc),
                "cost_mean_disagreement": float(sc.mean() - bc.mean()),
                "cost_std_disagreement": float(sc.std() - bc.std()),
                "entropy_disagreement": entropy_from_cost(sc) - entropy_from_cost(bc),
                "bsl_candidate_diversity": float(bact["diversity"][ep]),
                "st_candidate_diversity": float(sact["diversity"][ep]),
                "diversity_disagreement": float(sact["diversity"][ep] - bact["diversity"][ep]),
                "bsl_rank0_final_dist": float(btr["final"][ep, 0]),
                "st_best_final_dist": float(strj["final"][ep, s0]),
                "final_dist_improvement_proxy": float(btr["final"][ep, 0] - strj["final"][ep, s0]),
                "bsl_rank0_mean_dist": float(btr["mean"][ep, 0]),
                "st_best_mean_dist": float(strj["mean"][ep, s0]),
                "mean_dist_improvement_proxy": float(btr["mean"][ep, 0] - strj["mean"][ep, s0]),
                "bsl_rank0_min_dist": float(btr["min"][ep, 0]),
                "st_best_min_dist": float(strj["min"][ep, s0]),
                "min_dist_improvement_proxy": float(btr["min"][ep, 0] - strj["min"][ep, s0]),
                "bsl_rank0_progress": float(btr["progress"][ep, 0]),
                "st_best_progress": float(strj["progress"][ep, s0]),
                "progress_improvement_proxy": float(strj["progress"][ep, s0] - btr["progress"][ep, 0]),
                "bsl_latent_std": float(btr["latent_std"][ep, 0]),
                "st_latent_std": float(strj["latent_std"][ep, s0]),
                "latent_std_disagreement": float(strj["latent_std"][ep, s0] - btr["latent_std"][ep, 0]),
                "bsl_action_norm": float(bact["norm_mean"][ep, 0]),
                "st_action_norm": float(sact["norm_mean"][ep, s0]),
                "action_norm_disagreement": float(sact["norm_mean"][ep, s0] - bact["norm_mean"][ep, 0]),
            }
            rows.append(row)
    return rows


def feature_names(rows: list[dict]) -> list[str]:
    excluded = {
        "seed",
        "episode",
        "dataset_index",
        "bsl_success",
        "st_top1_success",
        "st_top3_success",
        "st_top5_success",
        "st_top10_success",
        "st_top30_success",
        "fix_raw_top1",
        "fix_top10_opportunity",
        "harm_raw_top1",
    }
    return [k for k in rows[0] if k not in excluded]


def make_x(rows: list[dict], feats: list[str]) -> np.ndarray:
    return np.asarray([[r[f] for f in feats] for r in rows], dtype=np.float64)


def fit_logistic(xtr: np.ndarray, ytr: np.ndarray, sample_weight: np.ndarray, xva: np.ndarray) -> np.ndarray:
    mean, std = xtr.mean(axis=0), xtr.std(axis=0) + 1e-6
    xs, xv = (xtr - mean) / std, (xva - mean) / std
    xb = np.concatenate([xs, np.ones((xs.shape[0], 1))], axis=1)
    vb = np.concatenate([xv, np.ones((xv.shape[0], 1))], axis=1)
    w = np.zeros(xb.shape[1])
    sw = sample_weight / (sample_weight.mean() + 1e-12)
    for _ in range(3000):
        z = np.clip(xb @ w, -40, 40)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = (xb.T @ ((p - ytr) * sw)) / len(ytr)
        grad[:-1] += 1e-3 * w[:-1]
        w -= 0.04 * grad
    return 1.0 / (1.0 + np.exp(-np.clip(vb @ w, -40, 40)))


def fit_models(xtr: np.ndarray, ytr: np.ndarray, sample_weight: np.ndarray, xva: np.ndarray) -> dict[str, np.ndarray]:
    scores = {"logistic": fit_logistic(xtr, ytr, sample_weight, xva)}
    try:
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        models = {
            "extratrees": (
                ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, random_state=0, class_weight="balanced"),
                "extratreesclassifier__sample_weight",
            ),
            "rf": (
                RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=1, class_weight="balanced"),
                "randomforestclassifier__sample_weight",
            ),
        }
        for name, (clf, weight_key) in models.items():
            pipe = make_pipeline(StandardScaler(), clf)
            pipe.fit(xtr, ytr, **{weight_key: sample_weight})
            scores[name] = pipe.predict_proba(xva)[:, 1]
    except Exception as exc:
        scores["sklearn_error"] = np.asarray([np.nan])
        print(f"[warn] sklearn models skipped: {exc}")
    return scores


def evaluate_threshold(rows: list[dict], scores: np.ndarray, threshold: float, mode: str) -> dict:
    selected = scores >= threshold
    bsl = np.asarray([r["bsl_success"] for r in rows], dtype=bool)
    st_top1 = np.asarray([r["st_top1_success"] for r in rows], dtype=bool)
    st_top10 = np.asarray([r["st_top10_success"] for r in rows], dtype=bool)
    if mode == "deploy_raw_top1":
        chosen_success = np.where(selected, st_top1, bsl)
    elif mode == "oracle_top10_bound":
        chosen_success = np.where(selected, st_top10, bsl)
    else:
        raise ValueError(mode)
    fixed = int(((~bsl) & chosen_success).sum())
    harmed = int((bsl & (~chosen_success)).sum())
    switches = int(selected.sum())
    return {
        "mode": mode,
        "threshold": float(threshold),
        "top1": float(chosen_success.mean() * 100.0),
        "bsl_top1": float(bsl.mean() * 100.0),
        "raw_moda_top1": float(st_top1.mean() * 100.0),
        "fixed": fixed,
        "harmed": harmed,
        "net": fixed - harmed,
        "switches": switches,
        "selected_moda_rate": float(selected.mean() * 100.0),
        "top10_opportunity_selected": int((selected & (~bsl) & st_top10).sum()),
        "raw_top1_fixable_selected": int((selected & (~bsl) & st_top1).sum()),
        "bsl_success_selected": int((selected & bsl).sum()),
    }


def threshold_grid(scores: np.ndarray) -> np.ndarray:
    qs = np.linspace(0, 1, 61)
    vals = np.unique(np.quantile(scores, qs).round(8))
    return np.r_[vals.max() + 1e-6, vals[::-1], vals.min() - 1e-6]


def choose_threshold(train_rows: list[dict], train_scores: np.ndarray, harm_budget: int, mode: str) -> tuple[float, list[dict]]:
    curve = []
    for thr in threshold_grid(train_scores):
        r = evaluate_threshold(train_rows, train_scores, float(thr), mode)
        r["harm_budget"] = harm_budget
        curve.append(r)
    safe = [r for r in curve if r["harmed"] <= harm_budget]
    cand = safe if safe else curve
    best = max(cand, key=lambda r: (r["top1"], r["net"], r["fixed"], -r["harmed"], -r["switches"]))
    return float(best["threshold"]), curve


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = build_rows()
    feats = feature_names(all_rows)
    deployment_rows = []
    sweep_rows = []
    case_rows = []
    for split, (train_seeds, val_seeds) in SPLITS.items():
        train = [r for r in all_rows if r["seed"] in train_seeds]
        val = [r for r in all_rows if r["seed"] in val_seeds]
        xtr, xva = make_x(train, feats), make_x(val, feats)
        targets = {
            "raw_top1_fix_gate": [r["fix_raw_top1"] for r in train],
            "top10_opportunity_gate": [r["fix_top10_opportunity"] for r in train],
        }
        for target_name, target_values in targets.items():
            ytr = np.asarray(target_values, dtype=bool)
            sw = np.ones(len(train), dtype=np.float64)
            for i, r in enumerate(train):
                if target_name == "raw_top1_fix_gate" and r["fix_raw_top1"]:
                    sw[i] = 5.0
                elif target_name == "top10_opportunity_gate" and r["fix_top10_opportunity"]:
                    sw[i] = 4.0
                elif r["harm_raw_top1"]:
                    sw[i] = 5.0
                elif r["bsl_success"]:
                    sw[i] = 2.5
            train_score_bank = fit_models(xtr, ytr, sw, xtr)
            val_score_bank = fit_models(xtr, ytr, sw, xva)
            for model_name, scores in val_score_bank.items():
                if model_name.endswith("error"):
                    continue
                for mode in ["deploy_raw_top1", "oracle_top10_bound"]:
                    for hb in [0, 1, 2, 3, 5, 8]:
                        train_scores = train_score_bank[model_name]
                        thr, train_curve = choose_threshold(train, train_scores, hb, mode)
                        for tr in train_curve:
                            tr.update({"split": split, "model": model_name, "target": target_name, "eval_split": "train"})
                            sweep_rows.append(tr)
                        out = evaluate_threshold(val, scores, thr, mode)
                        out.update({"split": split, "model": model_name, "target": target_name, "harm_budget": hb, "threshold_source": "train_only"})
                        deployment_rows.append(out)
                        selected = scores >= thr
                        for i, (r, sel) in enumerate(zip(val, selected)):
                            if not sel:
                                continue
                            fixed = (not r["bsl_success"]) and r["st_top1_success"]
                            harmed = r["bsl_success"] and (not r["st_top1_success"])
                            if fixed or harmed or ((not r["bsl_success"]) and r["st_top10_success"]):
                                case_rows.append(
                                    {
                                        "split": split,
                                        "model": model_name,
                                        "target": target_name,
                                        "mode": mode,
                                        "harm_budget": hb,
                                        "seed": r["seed"],
                                        "episode": r["episode"],
                                        "dataset_index": r["dataset_index"],
                                        "gate_score": float(scores[i]),
                                        "threshold": float(thr),
                                        "bsl_success": r["bsl_success"],
                                        "st_top1_success": r["st_top1_success"],
                                        "st_top10_success": r["st_top10_success"],
                                        "fixed": bool(fixed),
                                        "harmed": bool(harmed),
                                        "best_moda_minus_bsl_cost": r["best_moda_minus_bsl_cost"],
                                        "final_dist_improvement_proxy": r["final_dist_improvement_proxy"],
                                        "progress_improvement_proxy": r["progress_improvement_proxy"],
                                    }
                                )
    # Aggregate OOF by method/mode/budget.
    aggregate = []
    keys = sorted({(r["model"], r["target"], r["mode"], r["harm_budget"]) for r in deployment_rows})
    for model, target, mode, hb in keys:
        rows = [r for r in deployment_rows if r["model"] == model and r["target"] == target and r["mode"] == mode and r["harm_budget"] == hb]
        if len(rows) != len(SPLITS):
            continue
        total_eps = 300 * len(SPLITS)
        bsl_success = sum(r["bsl_top1"] / 100.0 * 300 for r in rows)
        moda_success = sum(r["raw_moda_top1"] / 100.0 * 300 for r in rows)
        fixed = sum(r["fixed"] for r in rows)
        harmed = sum(r["harmed"] for r in rows)
        aggregate.append(
            {
                "model": model,
                "target": target,
                "mode": mode,
                "harm_budget": hb,
                "top1": float((bsl_success + fixed - harmed) / total_eps * 100.0) if mode == "deploy_raw_top1" else float(np.mean([r["top1"] for r in rows])),
                "bsl_top1": float(bsl_success / total_eps * 100.0),
                "raw_moda_top1": float(moda_success / total_eps * 100.0),
                "fixed": int(fixed),
                "harmed": int(harmed),
                "net": int(fixed - harmed),
                "switches": int(sum(r["switches"] for r in rows)),
                "selected_moda_rate": float(np.mean([r["selected_moda_rate"] for r in rows])),
                "top10_opportunity_selected": int(sum(r["top10_opportunity_selected"] for r in rows)),
                "raw_top1_fixable_selected": int(sum(r["raw_top1_fixable_selected"] for r in rows)),
                "bsl_success_selected": int(sum(r["bsl_success_selected"] for r in rows)),
            }
        )
    best_deploy = max(
        [r for r in aggregate if r["mode"] == "deploy_raw_top1"],
        key=lambda r: (r["top1"], r["net"], -r["harmed"], r["fixed"]),
    )
    write_csv(OUT / "risk_controlled_moda_integration.csv", deployment_rows + aggregate)
    write_csv(OUT / "gate_threshold_sweep.csv", sweep_rows)
    write_csv(OUT / "case_studies_fix_harm.csv", case_rows)
    (OUT / "risk_controlled_moda_integration.json").write_text(
        json.dumps({"features": feats, "deployment_rows": deployment_rows, "aggregate": aggregate, "best_deploy": best_deploy}, indent=2)
        + "\n"
    )
    md = [
        "# Risk-Controlled MoDA Integration",
        "",
        "Baseline remains the fallback planner. MoDA is used only when an episode-level opportunity gate fires. This is not a MoDA-only standalone planner.",
        "",
        "## Deployable Raw-Top1 MoDA Integration",
        "",
        "|model|target|harm budget|top1|bsl top1|raw MoDA top1|fixed|harmed|net|switches|selected %|top10 opp selected|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in aggregate:
        if r["mode"] != "deploy_raw_top1":
            continue
        md.append(
            f"|{r['model']}|{r['target']}|{r['harm_budget']}|{r['top1']:.2f}|{r['bsl_top1']:.2f}|{r['raw_moda_top1']:.2f}|"
            f"{r['fixed']}|{r['harmed']}|{r['net']}|{r['switches']}|{r['selected_moda_rate']:.2f}|{r['top10_opportunity_selected']}|"
        )
    md += [
        "",
        "## Top10 Opportunity Upper Bound",
        "",
        "This section is diagnostic only: it assumes a perfect verifier can choose a successful MoDA candidate from top10 when present.",
        "",
        "|model|target|harm budget|top1 bound|fixed|harmed|net|switches|selected %|top10 opp selected|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in aggregate:
        if r["mode"] != "oracle_top10_bound":
            continue
        md.append(
            f"|{r['model']}|{r['target']}|{r['harm_budget']}|{r['top1']:.2f}|{r['fixed']}|{r['harmed']}|{r['net']}|"
            f"{r['switches']}|{r['selected_moda_rate']:.2f}|{r['top10_opportunity_selected']}|"
        )
    md += [
        "",
        "## Verdict",
        "",
        f"Best deployable raw-top1 integration: {best_deploy['model']} target={best_deploy['target']} harm_budget={best_deploy['harm_budget']} top1={best_deploy['top1']:.2f}, fixed={best_deploy['fixed']}, harmed={best_deploy['harmed']}, net={best_deploy['net']}.",
    ]
    (OUT / "risk_controlled_moda_integration.md").write_text("\n".join(md) + "\n")
    print((OUT / "risk_controlled_moda_integration.md").read_text())


if __name__ == "__main__":
    main()
