from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

WM = Path("/data1/jingyixi/wm_runs")
OUT = WM / "env_traj_replacement_n100_20260527_104100"
sys.path.insert(0, str(WM))
import train_env_trajectory_replacement_n100 as tr

CONFIG = {
    "epoch": 30,
    "detector_topk": 50,
    "score_threshold": -0.5,
    "vote_k": 3,
    "max_switch_frac": 0.05,
}
SPLITS = {
    "splitA_train42_44_val45_47": ([42, 43, 44], [45, 46, 47]),
    "splitB_train45_47_val42_44": ([45, 46, 47], [42, 43, 44]),
}


def aggregate(per_seed):
    total = sum(r["episodes"] for r in per_seed)
    bsl = sum(r["bsl_top1"] * r["episodes"] / 100 for r in per_seed)
    sel = sum(r["selector_top1"] * r["episodes"] / 100 for r in per_seed)
    oracle = sum(r["oracle"] * r["episodes"] / 100 for r in per_seed)
    return {
        "episodes": total,
        "bsl_top1": 100 * bsl / total,
        "selector_top1": 100 * sel / total,
        "oracle": 100 * oracle / total,
        "fixed_vs_bsl": sum(r["fixed_vs_bsl"] for r in per_seed),
        "harmed_vs_bsl": sum(r["harmed_vs_bsl"] for r in per_seed),
        "switches": sum(r["switches"] for r in per_seed),
    }


def eval_split(args, dataset, split_name, train_seeds, val_seeds):
    train_idx, val_idx = tr.split_indices(dataset, train_seeds, val_seeds)
    clf = tr.train_detector(dataset, train_idx, args.detector, seed=0)
    det = tr.detector_scores(clf, dataset, val_idx)
    model_seeds = [0, 1, 2, 3]
    paths = [
        OUT / split_name / f"critic_seed{ms}" / f"checkpoint_epoch{CONFIG['epoch']}.pt"
        for ms in model_seeds
    ]
    score_sets = [tr.score_checkpoint(p, dataset, val_idx, args) for p in paths]
    labels = np.stack([dataset["episodes"][i]["labels"] for i in val_idx])
    bsl = np.asarray([dataset["episodes"][i]["bsl_success"] for i in val_idx], dtype=bool)
    seeds = np.asarray([dataset["episodes"][i]["seed"] for i in val_idx], dtype=int)
    epnos = np.asarray([dataset["episodes"][i]["episode"] for i in val_idx], dtype=int)
    oracle = labels.any(axis=1)
    score_mean = np.mean(score_sets, axis=0)
    score_stack = np.stack(score_sets, axis=0)
    order = np.argsort(-det)
    allowed = np.zeros(len(val_idx), dtype=bool)
    allowed[order[: min(CONFIG["detector_topk"], len(val_idx))]] = True
    votes = (score_stack[:, :, 1:].max(axis=2) > CONFIG["score_threshold"]).sum(axis=0)
    cand = []
    for ei in range(len(val_idx)):
        if not allowed[ei] or votes[ei] < CONFIG["vote_k"]:
            continue
        j = int(score_mean[ei, 1:].argmax() + 1)
        sc = float(score_mean[ei, j])
        if sc > CONFIG["score_threshold"]:
            margin = sc - float(score_mean[ei, 0])
            cand.append((sc, ei, j, margin))
    cand.sort(reverse=True)
    max_sw = max(1, int(round(len(val_idx) * CONFIG["max_switch_frac"])))
    pick = np.zeros(len(val_idx), dtype=int)
    score_margin = np.zeros(len(val_idx), dtype=np.float32)
    chosen_score = np.zeros(len(val_idx), dtype=np.float32)
    for sc, ei, j, margin in cand[:max_sw]:
        pick[ei] = j
        score_margin[ei] = margin
        chosen_score[ei] = sc
    succ = labels[np.arange(len(val_idx)), pick]
    switch_rows = []
    for ei, j in enumerate(pick):
        if j == 0:
            continue
        source = "bsl" if j < 30 else "stateroll"
        rank = int(j if j < 30 else j - 30)
        switch_rows.append(
            {
                "split": split_name,
                "seed": int(seeds[ei]),
                "episode_in_seed": int(epnos[ei]),
                "val_position": int(ei),
                "candidate_index_union": int(j),
                "source": source,
                "source_rank": rank,
                "bsl_success": bool(bsl[ei]),
                "selected_success": bool(succ[ei]),
                "fixed": bool((not bsl[ei]) and succ[ei]),
                "harmed": bool(bsl[ei] and (not succ[ei])),
                "detector_score": float(det[ei]),
                "selected_score": float(chosen_score[ei]),
                "bsl_score": float(score_mean[ei, 0]),
                "score_margin_selected_minus_bsl": float(score_margin[ei]),
                "vote_count": int(votes[ei]),
                "union_oracle": bool(oracle[ei]),
            }
        )
    per_seed = []
    for sd in sorted(set(seeds.tolist())):
        m = seeds == sd
        per_seed.append(
            {
                "split": split_name,
                "seed": int(sd),
                "episodes": int(m.sum()),
                "bsl_top1": float(bsl[m].mean() * 100),
                "selector_top1": float(succ[m].mean() * 100),
                "oracle": float(oracle[m].mean() * 100),
                "fixed_vs_bsl": int((~bsl[m] & succ[m]).sum()),
                "harmed_vs_bsl": int((bsl[m] & ~succ[m]).sum()),
                "switches": int((pick[m] != 0).sum()),
            }
        )
    summary = aggregate(per_seed)
    summary["split"] = split_name
    summary["detector_fixable_captured"] = int(
        sum(dataset["episodes"][val_idx[i]]["fixable"] for i in np.where(allowed)[0])
    )
    summary["avg_score_margin_switched"] = (
        float(np.mean([r["score_margin_selected_minus_bsl"] for r in switch_rows]))
        if switch_rows
        else 0.0
    )
    summary["source_counts"] = {
        "bsl": sum(r["source"] == "bsl" for r in switch_rows),
        "stateroll": sum(r["source"] == "stateroll" for r in switch_rows),
    }
    ranks = [r["source_rank"] for r in switch_rows]
    summary["rank_distribution"] = {str(k): int(ranks.count(k)) for k in sorted(set(ranks))}
    return summary, per_seed, switch_rows


def main():
    class Args:
        env_dir = "/data1/jingyixi/wm_runs/env_traj_features_n100"
        dropout = 0.2
        cpu = False
        detector = "extratrees"

    args = Args()
    dataset = tr.build_dataset(args, [42, 43, 44, 45, 46, 47])
    split_summaries, all_per_seed, all_switches = [], [], []
    for split_name, (train_seeds, val_seeds) in SPLITS.items():
        summary, per_seed, switches = eval_split(args, dataset, split_name, train_seeds, val_seeds)
        split_summaries.append(summary)
        all_per_seed.extend(per_seed)
        all_switches.extend(switches)
    oof = aggregate(all_per_seed)
    oof["avg_score_margin_switched"] = (
        float(np.mean([r["score_margin_selected_minus_bsl"] for r in all_switches]))
        if all_switches
        else 0.0
    )
    oof["source_counts"] = {
        "bsl": sum(r["source"] == "bsl" for r in all_switches),
        "stateroll": sum(r["source"] == "stateroll" for r in all_switches),
    }
    ranks = [r["source_rank"] for r in all_switches]
    oof["rank_distribution"] = {str(k): int(ranks.count(k)) for k in sorted(set(ranks))}
    result = {
        "config": CONFIG,
        "split_summaries": split_summaries,
        "oof": oof,
        "per_seed": all_per_seed,
        "switch_cases": all_switches,
        "comparisons": {
            "feature_mlp_mil_best_n100": "81.0 -> 81.3",
            "conservative_train_selected_env_trajectory": "81.0 -> 80.8",
            "robust_global_train_selected": "81.0 -> 80.5",
        },
    }
    (OUT / "fixed_policy_epoch30_vote3_analysis.json").write_text(json.dumps(result, indent=2))
    fields = [
        "split",
        "seed",
        "episode_in_seed",
        "val_position",
        "candidate_index_union",
        "source",
        "source_rank",
        "bsl_success",
        "selected_success",
        "fixed",
        "harmed",
        "detector_score",
        "selected_score",
        "bsl_score",
        "score_margin_selected_minus_bsl",
        "vote_count",
        "union_oracle",
    ]
    with (OUT / "fixed_policy_epoch30_vote3_switch_cases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_switches:
            writer.writerow(row)
    lines = [
        "# Fixed Policy epoch30 vote3",
        "",
        f"Source: `{OUT}`",
        "",
        "Fixed config:",
        "",
        f"```json\n{json.dumps(CONFIG, indent=2)}\n```",
        "",
        "## Main Result",
        "",
        f"OOF: bsl {oof['bsl_top1']:.1f} -> selector {oof['selector_top1']:.1f}, oracle {oof['oracle']:.1f}, fixed={oof['fixed_vs_bsl']}, harmed={oof['harmed_vs_bsl']}, switches={oof['switches']}",
        "",
        f"Average switched score margin: {oof['avg_score_margin_switched']:.4f}",
        f"Switch source counts: {oof['source_counts']}",
        f"Switch rank distribution: {oof['rank_distribution']}",
        "",
        "## Split Results",
        "",
        "|split|bsl|selector|oracle|fixed|harmed|switches|avg margin|bsl switches|stateroll switches|rank dist|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in split_summaries:
        lines.append(
            f"|{s['split']}|{s['bsl_top1']:.1f}|{s['selector_top1']:.1f}|{s['oracle']:.1f}|{s['fixed_vs_bsl']}|{s['harmed_vs_bsl']}|{s['switches']}|{s['avg_score_margin_switched']:.4f}|{s['source_counts']['bsl']}|{s['source_counts']['stateroll']}|{s['rank_distribution']}|"
        )
    lines += [
        "",
        "## Per Seed",
        "",
        "|split|seed|bsl|selector|oracle|fixed|harmed|switches|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in all_per_seed:
        lines.append(
            f"|{r['split']}|{r['seed']}|{r['bsl_top1']:.1f}|{r['selector_top1']:.1f}|{r['oracle']:.1f}|{r['fixed_vs_bsl']}|{r['harmed_vs_bsl']}|{r['switches']}|"
        )
    lines += [
        "",
        "## Switch Cases",
        "",
        "Detailed switch cases are saved in `fixed_policy_epoch30_vote3_switch_cases.csv`.",
        "",
        "## Comparisons",
        "",
        "|method|OOF result|",
        "|---|---:|",
        "|feature-MLP MIL best n100|81.0 -> 81.3|",
        "|conservative train-selected env trajectory|81.0 -> 80.8|",
        "|robust global train-selected|81.0 -> 80.5|",
        f"|fixed epoch30 vote3 policy|81.0 -> {oof['selector_top1']:.1f}|",
    ]
    (OUT / "fixed_policy_epoch30_vote3_summary.md").write_text("\n".join(lines) + "\n")
    print((OUT / "fixed_policy_epoch30_vote3_summary.md").read_text())


if __name__ == "__main__":
    main()
