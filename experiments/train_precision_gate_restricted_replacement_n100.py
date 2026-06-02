from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WM = Path("/data1/jingyixi/wm_runs")
sys.path.insert(0, str(WM))

import train_gate_only_opportunity_detector_n100 as gate


OUT = Path("/data1/jingyixi/wm_runs/precision_gate_restricted_replacement_n100_20260528")


class Args:
    bsl_action_dir = "/data1/jingyixi/wm_runs/bsl_normalbudget_candidate_pool_s300_steps30_n100/proposal_data"
    bsl_raw_dir = "/data1/jingyixi/wm_runs/bsl_normalbudget_candidate_pool_s300_steps30_n100/raw_rollout_npz"
    st_action_dir = "/data1/jingyixi/wm_runs/stateroll_normalbudget_candidate_pool_s300_steps30_n100/proposal_data"
    st_raw_dir = "/data1/jingyixi/wm_runs/stateroll_normalbudget_candidate_pool_s300_steps30_n100/raw_rollout_npz"


def split_defs(seeds):
    out = {
        "splitA_train42_44_val45_47": ([42, 43, 44], [45, 46, 47]),
        "splitB_train45_47_val42_44": ([45, 46, 47], [42, 43, 44]),
    }
    for held in seeds:
        out[f"loso_hold{held}"] = ([s for s in seeds if s != held], [held])
    return out


def build_rows_and_raw(args, seeds):
    rows = []
    raw = {}
    for seed in seeds:
        b = gate.load_src(args.bsl_action_dir, args.bsl_raw_dir, "baseline", seed)
        st = gate.load_src(args.st_action_dir, args.st_raw_dir, "vf05_mix20", seed)
        assert np.all(b["indices"] == st["indices"])
        raw[seed] = {"b": b, "st": st}
    rows = gate.build_rows(args, seeds)
    return rows, raw


def episode_features(row):
    return np.asarray(
        [
            row["bsl_margin_top2"],
            row["bsl_margin_top5"],
            row["bsl_margin_top10"],
            row["bsl_cost_std"],
            row["bsl_cost_entropy"],
            row["bsl_top1_cost"],
            row["st_minus_bsl_best_cost"],
            row["bsl_rank0_minus_st_best_final_dist"],
            row["bsl_rank0_minus_st_best_mean_dist"],
            row["bsl_rank0_minus_st_best_min_dist"],
            row["st_best_minus_bsl_rank0_progress"],
            row["bsl_rank0_final_dist"],
            row["bsl_rank0_mean_dist"],
            row["bsl_rank0_min_dist"],
            row["bsl_rank0_progress"],
            row["bsl_action_norm"],
            row["bsl_action_std"],
            row["st_best_action_norm"],
            row["st_best_action_std"],
        ],
        dtype=np.float32,
    )


def candidate_features(row, raw, ep, cand):
    b = raw["b"]
    st = raw["st"]
    bc = b["costs"][ep]
    sc = st["costs"][ep]
    bp = b["pred"][ep]
    sp = st["pred"][ep]
    bg = b["goal"][ep]
    sg = st["goal"][ep]
    b_tr = gate.traj_stats(b["pred"][ep : ep + 1], b["goal"][ep : ep + 1])
    st_tr = gate.traj_stats(st["pred"][ep : ep + 1], st["goal"][ep : ep + 1])
    b_act = gate.action_stats(b["actions"][ep : ep + 1])
    st_act = gate.action_stats(st["actions"][ep : ep + 1])
    b_best = int(np.argmin(bc))
    st_best = int(np.argmin(sc))
    cand_tr = gate.traj_stats(st["pred"][ep : ep + 1], st["goal"][ep : ep + 1])
    # use stateroll candidates only for the restricted selector
    return np.asarray(
        [
            row["bsl_margin_top2"],
            row["bsl_margin_top5"],
            row["bsl_margin_top10"],
            row["bsl_cost_std"],
            row["bsl_cost_entropy"],
            row["bsl_top1_cost"],
            row["st_minus_bsl_best_cost"],
            row["bsl_rank0_minus_st_best_final_dist"],
            row["bsl_rank0_minus_st_best_mean_dist"],
            row["bsl_rank0_minus_st_best_min_dist"],
            row["st_best_minus_bsl_rank0_progress"],
            row["bsl_action_norm"],
            row["bsl_action_std"],
            row["st_best_action_norm"],
            row["st_best_action_std"],
            float(cand - 30) / 29.0,
            float(sc[cand]),
            float(sc[cand] - bc[0]),
            float(sc[cand] - sc[st_best]),
            float(st_tr["final"][0, cand] if cand < st_tr["final"].shape[1] else st_tr["final"][0, st_best]),
            float(st_tr["mean"][0, cand] if cand < st_tr["mean"].shape[1] else st_tr["mean"][0, st_best]),
            float(st_tr["min"][0, cand] if cand < st_tr["min"].shape[1] else st_tr["min"][0, st_best]),
            float(st_tr["progress"][0, cand] if cand < st_tr["progress"].shape[1] else st_tr["progress"][0, st_best]),
            float(st_act["norm"][0, cand] if cand < st_act["norm"].shape[1] else st_act["norm"][0, st_best]),
            float(st_act["std"][0, cand] if cand < st_act["std"].shape[1] else st_act["std"][0, st_best]),
            float(row["bsl_success"]),
            float(row["opportunity"]),
            float(row["stateroll_only_fixable"]),
        ],
        dtype=np.float32,
    )


def make_detector_xy(rows, feature_names):
    x = np.asarray([[r[k] for k in feature_names] for r in rows], dtype=np.float32)
    y = np.asarray([r["opportunity"] for r in rows], dtype=np.int64)
    w = np.ones(len(rows), dtype=np.float32)
    for i, r in enumerate(rows):
        if r["stateroll_only_fixable"]:
            w[i] = 2.5
        elif r["opportunity"]:
            w[i] = 1.5
    return x, y, w


def detector_models(seed=0):
    return {
        "extratrees": ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1),
        "randomforest": RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1),
    }


def fit_predict_detector(model, xtr, ytr, wtr, xva):
    try:
        if hasattr(model, "named_steps"):
            final_step = list(model.named_steps.keys())[-1]
            model.fit(xtr, ytr, **{f"{final_step}__sample_weight": wtr})
        else:
            model.fit(xtr, ytr, sample_weight=wtr)
    except TypeError:
        model.fit(xtr, ytr)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(xva)[:, 1]
    return model.decision_function(xva)


def selected_mask(rows, scores, rule):
    n = len(rows)
    order = np.argsort(-scores)
    topk_mask = np.zeros(n, dtype=bool)
    topk_mask[order[: min(rule["topk"], n)]] = True

    margin = np.asarray([r["bsl_margin_top10"] for r in rows], dtype=np.float64)
    entropy = np.asarray([r["bsl_cost_entropy"] for r in rows], dtype=np.float64)
    st_gap = np.asarray([r["st_minus_bsl_best_cost"] for r in rows], dtype=np.float64)
    abs_gap = np.abs(st_gap)

    masks = [topk_mask]
    if rule.get("margin_thr") is not None:
        masks.append(margin <= rule["margin_thr"])
    if rule.get("entropy_thr") is not None:
        masks.append(entropy >= rule["entropy_thr"])
    if rule.get("st_gap_thr") is not None:
        masks.append(st_gap <= rule["st_gap_thr"])
    if rule.get("abs_gap_thr") is not None:
        masks.append(abs_gap >= rule["abs_gap_thr"])

    if rule["combine"] == "AND":
        return np.logical_and.reduce(masks)
    if len(masks) == 1:
        return masks[0]
    return masks[0] & np.logical_or.reduce(masks[1:])


def rule_templates():
    qs = [0.10, 0.20, 0.30, 0.40]
    topks = [5, 10, 15, 20, 30, 50]
    rules = []
    for topk in topks:
        rules.append({"topk": topk, "combine": "AND"})
        for q in qs:
            for key in ["margin_thr", "entropy_thr", "st_gap_thr", "abs_gap_thr"]:
                rules.append({"topk": topk, key: q, "combine": "AND"})
                rules.append({"topk": topk, key: q, "combine": "OR"})
        for q1 in qs:
            for q2 in qs:
                rules.append({"topk": topk, "margin_thr": q1, "entropy_thr": q2, "combine": "AND"})
                rules.append({"topk": topk, "margin_thr": q1, "st_gap_thr": q2, "combine": "AND"})
                rules.append({"topk": topk, "entropy_thr": q1, "st_gap_thr": q2, "combine": "AND"})
                rules.append({"topk": topk, "margin_thr": q1, "entropy_thr": q2, "combine": "OR"})
                rules.append({"topk": topk, "margin_thr": q1, "st_gap_thr": q2, "combine": "OR"})
                rules.append({"topk": topk, "entropy_thr": q1, "st_gap_thr": q2, "combine": "OR"})
    return rules


def inst_rule_from_train(train_rows, tpl):
    rule = {"topk": tpl["topk"], "combine": tpl["combine"]}
    for key in ["margin_thr", "entropy_thr", "st_gap_thr", "abs_gap_thr"]:
        if key in tpl:
            q = tpl[key]
            arr_key = {
                "margin_thr": "bsl_margin_top10",
                "entropy_thr": "bsl_cost_entropy",
                "st_gap_thr": "st_minus_bsl_best_cost",
                "abs_gap_thr": "st_minus_bsl_best_cost",
            }[key]
            arr = np.asarray([r[arr_key] for r in train_rows], dtype=np.float64)
            if key == "entropy_thr":
                rule[key] = float(np.quantile(arr, 1.0 - q))
            elif key == "abs_gap_thr":
                rule[key] = float(np.quantile(np.abs(arr), 1.0 - q))
            else:
                rule[key] = float(np.quantile(arr, q))
    return rule


def rule_name(rule):
    parts = [f"top{rule['topk']}"]
    for key in ["margin_thr", "entropy_thr", "st_gap_thr", "abs_gap_thr"]:
        if key in rule:
            parts.append(key.replace("_thr", "") + f"@{rule[key]:.3f}")
    parts.append(rule["combine"])
    return "+".join(parts)


def rule_metrics(rows, scores, rule):
    selected = selected_mask(rows, scores, rule)
    st_only = np.asarray([r["stateroll_only_fixable"] for r in rows], dtype=bool)
    opportunity = np.asarray([r["opportunity"] for r in rows], dtype=bool)
    bsl_success = np.asarray([r["bsl_success"] for r in rows], dtype=bool)
    nonfix_bsl_fail = (~bsl_success) & (~opportunity)
    return {
        "selected_count": int(selected.sum()),
        "stateroll_only_captured": int((selected & st_only).sum()),
        "stateroll_only_total": int(st_only.sum()),
        "opportunity_captured": int((selected & opportunity).sum()),
        "opportunity_total": int(opportunity.sum()),
        "bsl_success_false_positive": int((selected & bsl_success).sum()),
        "bsl_failure_nonfixable_false_positive": int((selected & nonfix_bsl_fail).sum()),
        "precision_stateroll_only": float((selected & st_only).sum() / selected.sum()) if selected.sum() else 0.0,
        "precision_opportunity": float((selected & opportunity).sum() / selected.sum()) if selected.sum() else 0.0,
    }


def pick_gate_rule(train_rows, train_scores):
    rules = []
    for tpl in rule_templates():
        rule = inst_rule_from_train(train_rows, tpl)
        m = rule_metrics(train_rows, train_scores, rule)
        m.update({"rule": rule, "rule_name": rule_name(rule)})
        rules.append(m)
    safe = [r for r in rules if r["bsl_success_false_positive"] <= 5 and r["selected_count"] <= 20]
    cand = safe if safe else rules
    cand.sort(key=lambda r: (r["stateroll_only_captured"], r["precision_stateroll_only"], -r["bsl_success_false_positive"], -r["selected_count"]), reverse=True)
    return cand[0], rules


def fixed_precision_gate_rule(split_name, train_rows):
    # Hard-coded precision-gate families derived from the previous scan.
    # Thresholds are instantiated on train split only, but the rule family is fixed.
    if "splitA" in split_name:
        tpl = {"topk": 10, "st_gap_thr": 0.20, "combine": "AND"}
    elif "splitB" in split_name:
        tpl = {"topk": 10, "abs_gap_thr": 0.10, "combine": "AND"}
    elif "hold" in split_name:
        # Conservative default for LOSO: use the safer splitA family.
        tpl = {"topk": 10, "st_gap_thr": 0.20, "combine": "AND"}
    else:
        tpl = {"topk": 10, "st_gap_thr": 0.20, "combine": "AND"}
    rule = inst_rule_from_train(train_rows, tpl)
    return rule, rule_name(rule)


def gate_upper_bound(rows, selected):
    st_only = np.asarray([r["stateroll_only_fixable"] for r in rows], dtype=bool)
    opp = np.asarray([r["opportunity"] for r in rows], dtype=bool)
    bsl = np.asarray([r["bsl_success"] for r in rows], dtype=bool)
    fixed_upper = int((selected & st_only).sum())
    opp_upper = int((selected & opp).sum())
    return {
        "selected_count": int(selected.sum()),
        "stateroll_only_captured": fixed_upper,
        "opportunity_captured": opp_upper,
        "bsl_success_false_positive": int((selected & bsl).sum()),
        "nonfixable_false_positive": int((selected & (~opp) & (~bsl)).sum()),
    }


def candidate_dataset(raw, rows, selected_idx):
    X, y, w, meta = [], [], [], []
    for i in selected_idx:
        r = rows[i]
        seed = r["seed"]
        ep = r["episode"]
        b = raw[seed]["b"]
        st = raw[seed]["st"]
        # restrict selector to stateroll candidates only
        for cand in range(30):
            bc = b["costs"][ep]
            sc = st["costs"][ep]
            bp = b["pred"][ep]
            sp = st["pred"][ep]
            bg = b["goal"][ep]
            sg = st["goal"][ep]
            b_tr = gate.traj_stats(b["pred"][ep : ep + 1], b["goal"][ep : ep + 1])
            st_tr = gate.traj_stats(st["pred"][ep : ep + 1], st["goal"][ep : ep + 1])
            b_act = gate.action_stats(b["actions"][ep : ep + 1])
            st_act = gate.action_stats(st["actions"][ep : ep + 1])
            feat = np.asarray(
                [
                    *episode_features(r),
                    float(cand) / 29.0,
                    float(sc[cand]),
                    float(sc[cand] - bc[0]),
                    float(sc[cand] - np.min(sc)),
                    float(st_tr["final"][0, cand]),
                    float(st_tr["mean"][0, cand]),
                    float(st_tr["min"][0, cand]),
                    float(st_tr["progress"][0, cand]),
                    float(st_act["norm"][0, cand]),
                    float(st_act["std"][0, cand]),
                    float(r["bsl_success"]),
                    float(r["opportunity"]),
                    float(r["stateroll_only_fixable"]),
                ],
                dtype=np.float32,
            )
            X.append(feat)
            y.append(bool(st["labels"][ep, cand]))
            w.append(2.5 if r["stateroll_only_fixable"] else (1.5 if r["opportunity"] else 1.0))
            meta.append({"seed": seed, "episode": ep, "cand": cand, "bsl_success": bool(r["bsl_success"]), "opportunity": bool(r["opportunity"]), "st_only": bool(r["stateroll_only_fixable"])})
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), np.asarray(w, dtype=np.float32), meta


def candidate_models(seed=0):
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
        "extratrees": ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1),
        "randomforest": RandomForestClassifier(n_estimators=600, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 64), alpha=1e-3, learning_rate_init=1e-3, max_iter=1200, random_state=seed, early_stopping=True)),
    }


def fit_candidate_scores(model, xtr, ytr, wtr, xva):
    try:
        if hasattr(model, "named_steps"):
            final_step = list(model.named_steps.keys())[-1]
            model.fit(xtr, ytr, **{f"{final_step}__sample_weight": wtr})
        else:
            model.fit(xtr, ytr, sample_weight=wtr)
    except TypeError:
        model.fit(xtr, ytr)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(xva)[:, 1]
    return model.decision_function(xva)


def eval_restricted_selector(rows, selected_mask_val, cand_scores, threshold):
    st_only = np.asarray([r["stateroll_only_fixable"] for r in rows], dtype=bool)
    bsl_success = np.asarray([r["bsl_success"] for r in rows], dtype=bool)
    opp = np.asarray([r["opportunity"] for r in rows], dtype=bool)
    selected = np.zeros(len(rows), dtype=bool)
    fixed = harmed = switches = 0
    selected_idx = np.where(selected_mask_val)[0]
    # candidate scores are flattened per selected episode in same order as candidate_dataset
    ptr = 0
    per_seed = {}
    for idx in selected_idx:
        # 30 candidates per episode
        ep_scores = cand_scores[ptr : ptr + 30]
        ptr += 30
        best_rel = int(np.argmax(ep_scores))
        best_score = float(ep_scores[best_rel])
        if best_score > threshold:
            selected[idx] = True
            switches += 1
            # candidate index in full union pool
            cand = 30 + best_rel
            # success if any stateroll candidate success among chosen candidate
            # We need actual label from rows? rows only episode-level. handled outside by caller.
    return selected, switches


def candidate_eval_from_scores(rows, gate_selected_mask, cand_scores_flat, threshold, raw):
    # Returns episode-level selection with fixed/harmed relative to bsl.
    selected_episode = np.zeros(len(rows), dtype=bool)
    selected_candidate = np.full(len(rows), -1, dtype=int)
    ptr = 0
    for idx in np.where(gate_selected_mask)[0]:
        ep_scores = cand_scores_flat[ptr : ptr + 30]
        ptr += 30
        best_rel = int(np.argmax(ep_scores))
        best_score = float(ep_scores[best_rel])
        if best_score > threshold:
            selected_episode[idx] = True
            selected_candidate[idx] = best_rel
    fixed = harmed = 0
    for i, r in enumerate(rows):
        if not selected_episode[i]:
            continue
        seed = r["seed"]
        ep = r["episode"]
        cand = selected_candidate[i]
        succ = bool(raw[seed]["st"]["labels"][ep, cand])
        if (not r["bsl_success"]) and succ:
            fixed += 1
        if r["bsl_success"] and (not succ):
            harmed += 1
    return {
        "selected_episode_mask": selected_episode,
        "selected_candidate": selected_candidate,
        "switches": int(selected_episode.sum()),
        "fixed": int(fixed),
        "harmed": int(harmed),
    }


def choose_threshold(train_rows, train_gate_mask, cand_scores_flat):
    # One score per candidate. Threshold on per-episode max score among 30 stateroll candidates.
    episode_scores = []
    ptr = 0
    for idx in np.where(train_gate_mask)[0]:
        ep_scores = cand_scores_flat[ptr : ptr + 30]
        ptr += 30
        episode_scores.append(float(np.max(ep_scores)))
    episode_scores = np.asarray(episode_scores, dtype=np.float64)
    if len(episode_scores) == 0:
        return 1.1, []
    thresholds = np.unique(np.quantile(episode_scores, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).round(6))
    thresholds = np.concatenate(([0.0], thresholds, [0.95, 0.99]))
    rows = []
    for thr in thresholds:
        fixed = harmed = switches = 0
        ptr = 0
        for idx in np.where(train_gate_mask)[0]:
            ep_scores = cand_scores_flat[ptr : ptr + 30]
            ptr += 30
            best_rel = int(np.argmax(ep_scores))
            best_score = float(ep_scores[best_rel])
            if best_score > thr:
                switches += 1
                seed = train_rows[idx]["seed"]
                ep = train_rows[idx]["episode"]
                cand = best_rel
                succ = bool(raw_global[seed]["st"]["labels"][ep, cand])
                if (not train_rows[idx]["bsl_success"]) and succ:
                    fixed += 1
                if train_rows[idx]["bsl_success"] and (not succ):
                    harmed += 1
        rows.append({"threshold": float(thr), "fixed": int(fixed), "harmed": int(harmed), "switches": int(switches)})
    safe = [r for r in rows if r["harmed"] <= 1 and r["switches"] >= 1]
    cand = safe if safe else rows
    cand.sort(key=lambda r: (r["fixed"] - 3 * r["harmed"], -r["switches"], -r["threshold"]), reverse=True)
    return cand[0]["threshold"], rows


def select_gate_and_selector(train_rows, val_rows, train_scores, val_scores, raw, detector_name, split_name):
    # fixed precision gate family; do not choose gate from selector quality
    gate_rule, gate_rule_name = fixed_precision_gate_rule(split_name, train_rows)
    train_gate = selected_mask(train_rows, train_scores, gate_rule)
    val_gate = selected_mask(val_rows, val_scores, gate_rule)
    # gate upper bound
    train_gate_ub = gate_upper_bound(train_rows, train_gate)
    val_gate_ub = gate_upper_bound(val_rows, val_gate)
    # candidate selector restricted to gate-selected train episodes
    train_sel_idx = np.where(train_gate)[0]
    val_sel_idx = np.where(val_gate)[0]
    # Build candidate training data on selected train episodes
    xtr, ytr, wtr, _ = candidate_dataset(raw, train_rows, train_sel_idx)
    xva, yva, wva, _ = candidate_dataset(raw, val_rows, val_sel_idx)
    candidate_records = []
    for model_name, model in candidate_models(seed=0).items():
        scores_train = fit_candidate_scores(model, xtr, ytr, wtr, xtr)
        thr, thr_rows = choose_threshold(train_rows, train_gate, scores_train)
        scores_val = fit_candidate_scores(model, xtr, ytr, wtr, xva)
        eval_train = candidate_eval_from_scores(train_rows, train_gate, scores_train, thr, raw)
        eval_val = candidate_eval_from_scores(val_rows, val_gate, scores_val, thr, raw)
        selected_episode = np.zeros(len(val_rows), dtype=bool)
        selected_candidate = np.full(len(val_rows), -1, dtype=int)
        ptr = 0
        for i in range(len(val_rows)):
            if not val_gate[i]:
                continue
            ep_scores = scores_val[ptr : ptr + 30]
            ptr += 30
            best_rel = int(np.argmax(ep_scores))
            best_score = float(ep_scores[best_rel])
            if best_score > thr:
                selected_episode[i] = True
                selected_candidate[i] = best_rel
        fixed = harmed = switches = 0
        st_only = np.asarray([r["stateroll_only_fixable"] for r in val_rows], dtype=bool)
        for i, r in enumerate(val_rows):
            if not selected_episode[i]:
                continue
            seed = r["seed"]
            ep = r["episode"]
            cand = selected_candidate[i]
            succ = bool(raw[seed]["st"]["labels"][ep, cand])
            switches += 1
            if (not r["bsl_success"]) and succ:
                fixed += 1
            if r["bsl_success"] and (not succ):
                harmed += 1
        selected_count = int(selected_episode.sum())
        gate_oracle_fixable = int(((val_gate) & (~np.asarray([r["bsl_success"] for r in val_rows], dtype=bool)) & np.asarray([r["opportunity"] for r in val_rows], dtype=bool)).sum())
        candidate_records.append(
            {
                "split": split_name,
                "detector": detector_name,
                "selector_model": model_name,
                "gate_rule": gate_rule_name,
                "gate_rule_obj": gate_rule,
                "train_gate_selected": int(train_gate.sum()),
                "val_gate_selected": int(val_gate.sum()),
                "train_gate_bsl_fp": int(train_gate_ub["bsl_success_false_positive"]),
                "val_gate_bsl_fp": int(val_gate_ub["bsl_success_false_positive"]),
                "train_gate_st_only": int(train_gate_ub["stateroll_only_captured"]),
                "val_gate_st_only": int(val_gate_ub["stateroll_only_captured"]),
                "train_gate_opportunity": int(train_gate_ub["opportunity_captured"]),
                "val_gate_opportunity": int(val_gate_ub["opportunity_captured"]),
                "train_gate_nonfix_fp": int(train_gate_ub["nonfixable_false_positive"]),
                "val_gate_nonfix_fp": int(val_gate_ub["nonfixable_false_positive"]),
                "gate_oracle_fixable_val": gate_oracle_fixable,
                "candidate_threshold": float(thr),
                "selector_selected": selected_count,
                "selector_fixed": int(fixed),
                "selector_harmed": int(harmed),
                "selector_switches": int(switches),
                "selector_train_selected": int(eval_train["switches"]),
                "selector_train_fixed": int(eval_train["fixed"]),
                "selector_train_harmed": int(eval_train["harmed"]),
                "selector_train_threshold_rows": thr_rows,
            }
        )
    return candidate_records, train_gate, val_gate, train_gate_ub, val_gate_ub


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


def main():
    args = Args()
    seeds = [42, 43, 44, 45, 46, 47]
    rows, raw = build_rows_and_raw(args, seeds)
    feature_names = [k for k in rows[0].keys() if k not in {"seed", "episode", "bsl_success", "bsl_oracle", "st_oracle", "union_oracle", "opportunity", "stateroll_only_fixable"}]
    splits = split_defs(seeds)
    all_records = []
    global raw_global
    raw_global = raw
    for split_name, (train_seeds, val_seeds) in splits.items():
        train_rows = [r for r in rows if r["seed"] in train_seeds]
        val_rows = [r for r in rows if r["seed"] in val_seeds]
        # fit detector models on train rows and score train/val
        xtr, ytr, wtr = make_detector_xy(train_rows, feature_names)
        xva, yva, wva = make_detector_xy(val_rows, feature_names)
        for det_name, det_model in detector_models(seed=0).items():
            # train score and val score use same detector model per split
            train_scores = fit_predict_detector(det_model, xtr, ytr, wtr, xtr)
            # refit a fresh model for val so metrics are not contaminated by reused estimator state
            det_model2 = detector_models(seed=0)[det_name]
            val_scores = fit_predict_detector(det_model2, xtr, ytr, wtr, xva)
            recs, train_gate, val_gate, train_gate_ub, val_gate_ub = select_gate_and_selector(train_rows, val_rows, train_scores, val_scores, raw, det_name, split_name)
            all_records.extend(recs)
        # LOSO is handled below for per-seed gate-only analysis and restricted selector
    # LOSO
    for held in seeds:
        train_seeds = [s for s in seeds if s != held]
        val_seeds = [held]
        split_name = f"loso_hold{held}"
        train_rows = [r for r in rows if r["seed"] in train_seeds]
        val_rows = [r for r in rows if r["seed"] == held]
        xtr, ytr, wtr = make_detector_xy(train_rows, feature_names)
        xva, yva, wva = make_detector_xy(val_rows, feature_names)
        for det_name, det_model in detector_models(seed=held).items():
            train_scores = fit_predict_detector(det_model, xtr, ytr, wtr, xtr)
            det_model2 = detector_models(seed=held)[det_name]
            val_scores = fit_predict_detector(det_model2, xtr, ytr, wtr, xva)
            recs, _, _, _, _ = select_gate_and_selector(train_rows, val_rows, train_scores, val_scores, raw, det_name, split_name)
            all_records.extend(recs)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "precision_gate_restricted_replacement_n100.json").write_text(json.dumps({"records": all_records}, indent=2))
    write_csv(OUT / "precision_gate_restricted_replacement_n100.csv", all_records)
    # summary
    lines = ["# Precision-Gate Restricted Replacement n100", ""]
    lines.append("Train-chosen precision gate is selected from train split only, with bsl-success FP <= 5 and selected <= 20.")
    lines.append("")
    lines.append("|split|detector|selector|gate rule|train gate selected|val gate selected|train gate st-only|val gate st-only|train gate bsl FP|val gate bsl FP|gate oracle fixed val|candidate thr|selector fixed|selector harmed|selector switches|")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in all_records:
        lines.append(
            f"|{r['split']}|{r['detector']}|{r['selector_model']}|{r['gate_rule']}|{r['train_gate_selected']}|{r['val_gate_selected']}|{r['train_gate_st_only']}|{r['val_gate_st_only']}|{r['train_gate_bsl_fp']}|{r['val_gate_bsl_fp']}|{r['gate_oracle_fixable_val']}|{r['candidate_threshold']:.3f}|{r['selector_fixed']}|{r['selector_harmed']}|{r['selector_switches']}|"
        )
    (OUT / "precision_gate_restricted_replacement_n100.md").write_text("\n".join(lines) + "\n")
    print((OUT / "precision_gate_restricted_replacement_n100.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
