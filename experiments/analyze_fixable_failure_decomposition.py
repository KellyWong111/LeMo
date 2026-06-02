from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
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


def load_proposal(seed: int, variant: str):
    root = (
        WM / "bsl_normalbudget_candidate_pool_s300_steps30_n100" / "proposal_data"
        if variant == "baseline"
        else WM / "stateroll_normalbudget_candidate_pool_s300_steps30_n100" / "proposal_data"
    )
    return np.load(root / f"{variant}_seed{seed}.npz", allow_pickle=True)


def classify_fixable(
    ep,
    local_pos: int,
    det_scores,
    allowed,
    votes,
    score_mean,
    selected_pick,
):
    labels = ep["labels"].astype(bool)
    success_idx = np.where(labels)[0]
    best_j = int(score_mean[local_pos, 1:].argmax() + 1)
    best_score = float(score_mean[local_pos, best_j])
    best_success_score = float(score_mean[local_pos, success_idx].max()) if len(success_idx) else float("nan")
    best_success_j = int(success_idx[np.argmax(score_mean[local_pos, success_idx])]) if len(success_idx) else -1
    gate_pass_best = bool(best_score > CONFIG["score_threshold"] and votes[local_pos] >= CONFIG["vote_k"])
    gate_pass_best_success = bool(best_success_score > CONFIG["score_threshold"] and votes[local_pos] >= CONFIG["vote_k"])
    if not allowed[local_pos]:
        category = "detector_missed"
    elif not labels[best_j]:
        category = "detector_hit_but_selector_wrong"
    elif not gate_pass_best:
        category = "selector_correct_but_gate_rejected"
    elif selected_pick[local_pos] == best_j and labels[selected_pick[local_pos]]:
        category = "successfully_fixed"
    else:
        # The best candidate is successful and passes local gates, but the global switch cap
        # or score ordering did not select this episode.
        category = "selector_correct_but_gate_rejected"
    return {
        "category": category,
        "best_j": best_j,
        "best_score": best_score,
        "best_is_success": bool(labels[best_j]),
        "best_success_j": best_success_j,
        "best_success_score": best_success_score,
        "gate_pass_best": gate_pass_best,
        "gate_pass_best_success": gate_pass_best_success,
        "detector_score": float(det_scores[local_pos]),
        "vote_count": int(votes[local_pos]),
    }


def source_rank(j: int):
    if j < 30:
        return "bsl", j
    return "stateroll", j - 30


def aggregate(rows):
    groups = {}
    for cat in ["detector_missed", "detector_hit_but_selector_wrong", "selector_correct_but_gate_rejected", "successfully_fixed"]:
        sub = [r for r in rows if r["category"] == cat]
        if not sub:
            groups[cat] = {"count": 0, "percentage": 0.0}
            continue
        groups[cat] = {
            "count": len(sub),
            "percentage": 100.0 * len(sub) / len(rows),
            "avg_bsl_margin_top2": float(np.mean([r["bsl_margin_top2"] for r in sub])),
            "avg_bsl_margin_top5": float(np.mean([r["bsl_margin_top5"] for r in sub])),
            "avg_detector_score": float(np.mean([r["detector_score"] for r in sub])),
            "avg_best_replacement_score": float(np.mean([r["best_score"] for r in sub])),
            "avg_best_success_score": float(np.mean([r["best_success_score"] for r in sub])),
            "avg_best_rank": float(np.mean([r["best_rank"] for r in sub])),
            "avg_best_success_rank": float(np.mean([r["best_success_rank"] for r in sub])),
            "source_counts_best": dict(Counter(r["best_source"] for r in sub)),
            "source_counts_best_success": dict(Counter(r["best_success_source"] for r in sub)),
        }
    return groups


def main():
    class Args:
        env_dir = "/data1/jingyixi/wm_runs/env_traj_features_n100"
        dropout = 0.2
        cpu = False
        detector = "extratrees"

    args = Args()
    dataset = tr.build_dataset(args, [42, 43, 44, 45, 46, 47])
    proposal_cache = {}
    all_rows = []
    split_summaries = {}

    for split_name, (train_seeds, val_seeds) in SPLITS.items():
        train_idx, val_idx = tr.split_indices(dataset, train_seeds, val_seeds)
        clf = tr.train_detector(dataset, train_idx, args.detector, seed=0)
        det = tr.detector_scores(clf, dataset, val_idx)
        paths = [
            OUT / split_name / f"critic_seed{ms}" / f"checkpoint_epoch{CONFIG['epoch']}.pt"
            for ms in [0, 1, 2, 3]
        ]
        score_sets = [tr.score_checkpoint(p, dataset, val_idx, args) for p in paths]
        score_mean = np.mean(score_sets, axis=0)
        score_stack = np.stack(score_sets, axis=0)
        votes = (score_stack[:, :, 1:].max(axis=2) > CONFIG["score_threshold"]).sum(axis=0)
        order = np.argsort(-det)
        allowed = np.zeros(len(val_idx), dtype=bool)
        allowed[order[: min(CONFIG["detector_topk"], len(val_idx))]] = True

        candidates = []
        for local_pos, idx in enumerate(val_idx):
            if not allowed[local_pos] or votes[local_pos] < CONFIG["vote_k"]:
                continue
            j = int(score_mean[local_pos, 1:].argmax() + 1)
            sc = float(score_mean[local_pos, j])
            if sc > CONFIG["score_threshold"]:
                candidates.append((sc, local_pos, j))
        candidates.sort(reverse=True)
        max_sw = max(1, int(round(len(val_idx) * CONFIG["max_switch_frac"])))
        selected_pick = np.zeros(len(val_idx), dtype=int)
        for _, local_pos, j in candidates[:max_sw]:
            selected_pick[local_pos] = j

        split_rows = []
        for local_pos, idx in enumerate(val_idx):
            ep = dataset["episodes"][idx]
            if not ep["fixable"]:
                continue
            seed = int(ep["seed"])
            episode = int(ep["episode"])
            if seed not in proposal_cache:
                proposal_cache[seed] = {
                    "bsl": load_proposal(seed, "baseline"),
                    "st": load_proposal(seed, "vf05_mix20"),
                }
            bsl_costs = proposal_cache[seed]["bsl"]["costs"][episode]
            st_labels = proposal_cache[seed]["st"]["labels"][episode]
            cls = classify_fixable(ep, local_pos, det, allowed, votes, score_mean, selected_pick)
            best_source, best_rank = source_rank(cls["best_j"])
            best_success_source, best_success_rank = source_rank(cls["best_success_j"])
            rec = {
                "split": split_name,
                "seed": seed,
                "episode": episode,
                "category": cls["category"],
                "bsl_margin_top2": float(bsl_costs[1] - bsl_costs[0]),
                "bsl_margin_top5": float(bsl_costs[4] - bsl_costs[0]),
                "bsl_cost_rank0": float(bsl_costs[0]),
                "detector_score": cls["detector_score"],
                "detector_allowed": bool(allowed[local_pos]),
                "vote_count": cls["vote_count"],
                "best_union_index": cls["best_j"],
                "best_source": best_source,
                "best_rank": best_rank,
                "best_score": cls["best_score"],
                "best_is_success": cls["best_is_success"],
                "best_success_union_index": cls["best_success_j"],
                "best_success_source": best_success_source,
                "best_success_rank": best_success_rank,
                "best_success_score": cls["best_success_score"],
                "gate_pass_best": cls["gate_pass_best"],
                "gate_pass_best_success": cls["gate_pass_best_success"],
                "selected_union_index": int(selected_pick[local_pos]),
                "successfully_selected": bool(selected_pick[local_pos] != 0 and ep["labels"][selected_pick[local_pos]]),
                "stateroll_rank0_success": bool(st_labels[0]),
                "num_success_candidates_union": int(ep["labels"].sum()),
                "num_success_candidates_stateroll": int(st_labels.sum()),
            }
            split_rows.append(rec)
            all_rows.append(rec)
        split_summaries[split_name] = aggregate(split_rows)

    fields = list(all_rows[0].keys()) if all_rows else []
    with (OUT / "fixable_failure_decomposition.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    per_seed_counts = defaultdict(Counter)
    for r in all_rows:
        per_seed_counts[r["seed"]][r["category"]] += 1
    summary = {
        "config": CONFIG,
        "total_fixable": len(all_rows),
        "overall": aggregate(all_rows),
        "by_split": split_summaries,
        "per_seed_counts": {str(k): dict(v) for k, v in sorted(per_seed_counts.items())},
    }
    (OUT / "fixable_failure_decomposition.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Fixable Failure Decomposition",
        "",
        f"Source: `{OUT}`",
        "",
        f"Fixed policy config: `{CONFIG}`",
        "",
        f"Total fixable episodes analyzed: {len(all_rows)}",
        "",
        "## Overall Categories",
        "",
        "|category|count|pct|avg bsl top2 margin|avg detector|avg best score|avg best-success score|avg best rank|avg best-success rank|best source counts|best-success source counts|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    overall = summary["overall"]
    for cat, s in overall.items():
        if s["count"] == 0:
            lines.append(f"|{cat}|0|0.0|NA|NA|NA|NA|NA|NA|{{}}|{{}}|")
        else:
            lines.append(
                f"|{cat}|{s['count']}|{s['percentage']:.1f}|{s['avg_bsl_margin_top2']:.4f}|"
                f"{s['avg_detector_score']:.4f}|{s['avg_best_replacement_score']:.4f}|{s['avg_best_success_score']:.4f}|"
                f"{s['avg_best_rank']:.2f}|{s['avg_best_success_rank']:.2f}|{s['source_counts_best']}|{s['source_counts_best_success']}|"
            )

    lines += [
        "",
        "## Per Seed Category Counts",
        "",
        "|seed|detector missed|selector wrong|gate rejected|successfully fixed|",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed, cnt in sorted(per_seed_counts.items()):
        lines.append(
            f"|{seed}|{cnt.get('detector_missed', 0)}|{cnt.get('detector_hit_but_selector_wrong', 0)}|"
            f"{cnt.get('selector_correct_but_gate_rejected', 0)}|{cnt.get('successfully_fixed', 0)}|"
        )

    lines += [
        "",
        "## Split Summaries",
        "",
        "|split|category|count|pct|avg detector|avg best score|avg best-success score|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split_name, cats in split_summaries.items():
        for cat, s in cats.items():
            if s["count"] == 0:
                lines.append(f"|{split_name}|{cat}|0|0.0|NA|NA|NA|")
            else:
                lines.append(
                    f"|{split_name}|{cat}|{s['count']}|{s['percentage']:.1f}|"
                    f"{s['avg_detector_score']:.4f}|{s['avg_best_replacement_score']:.4f}|{s['avg_best_success_score']:.4f}|"
                )

    lines += [
        "",
        "## Interpretation",
        "",
        "This decomposition separates the fixed-policy failure modes on the 82 fixable opportunities. If most cases are detector_missed, the next bottleneck is bsl-departure detection. If most cases are selector_wrong, the replacement scorer is ranking the wrong candidate. If many cases are gate_rejected, the scorer can identify a successful candidate but the conservative vote/threshold/cap prevents switching.",
    ]
    (OUT / "fixable_failure_decomposition.md").write_text("\n".join(lines) + "\n")
    print((OUT / "fixable_failure_decomposition.md").read_text())


if __name__ == "__main__":
    main()
