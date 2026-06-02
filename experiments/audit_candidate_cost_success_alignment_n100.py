from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("/data1/jingyixi/wm_runs")
OUT = ROOT / "overnight_moda_mechanism_summary_n100_20260529"

BSL_ROOT = ROOT / "bsl_normalbudget_candidate_pool_s300_steps30_n100" / "proposal_data"
ST_ROOT = ROOT / "stateroll_normalbudget_candidate_pool_s300_steps30_n100" / "proposal_data"
SEEDS = [42, 43, 44, 45, 46, 47]


def load_pool(root: Path, variant: str, seed: int):
    p = root / f"{variant}_seed{seed}.npz"
    data = np.load(p, allow_pickle=True)
    return {
        "costs": data["costs"].astype(np.float64),
        "labels": data["labels"].astype(bool),
    }


def build_union(bsl: dict, st: dict) -> dict:
    return {
        "costs": np.concatenate([bsl["costs"], st["costs"]], axis=1),
        "labels": np.concatenate([bsl["labels"], st["labels"]], axis=1),
    }


def candidate_auc(labels: np.ndarray, costs: np.ndarray):
    y = labels.reshape(-1).astype(np.int64)
    s = (-costs).reshape(-1)
    if len(np.unique(y)) < 2:
        return None
    return float(binary_auc(y, s))


def binary_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    y = y.astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    i = 0
    while i < len(score):
        j = i + 1
        while j < len(score) and sorted_score[j] == sorted_score[i]:
            j += 1
        # ranks are 1-based; average rank for ties
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos = ranks[y].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def first_success_rank(costs: np.ndarray, labels: np.ndarray):
    order = np.argsort(costs, axis=1, kind="stable")
    ranks = []
    no_success = 0
    first_success_costs = []
    rank0_gap = []
    near_miss = 0
    topk_hits = {1: 0, 3: 0, 5: 0, 10: 0, 30: 0}

    for ep in range(labels.shape[0]):
        lab = labels[ep]
        c = costs[ep]
        sorted_idx = order[ep]
        success_pos = np.where(lab[sorted_idx])[0]
        success_mask = lab.astype(bool)
        has_success = bool(success_mask.any())
        rank0_is_success = bool(lab[0])
        if not has_success:
            no_success += 1
            continue

        first_rank = int(success_pos[0] + 1)
        ranks.append(first_rank)
        first_success_cost = float(c[success_mask].min())
        first_success_costs.append(first_success_cost)
        rank0_gap.append(float(c[0] - first_success_cost))
        # near miss: rank0 fails but at least one success exists in the pool
        if not rank0_is_success:
            near_miss += 1

        for k in topk_hits:
            if np.any(lab[sorted_idx[: min(k, len(sorted_idx))]]):
                topk_hits[k] += 1

    return {
        "first_success_ranks": ranks,
        "first_success_costs": first_success_costs,
        "rank0_minus_best_success": rank0_gap,
        "no_success_count": no_success,
        "near_miss_failure_count": near_miss,
        "topk_hits": topk_hits,
    }


def summarize_pool(name: str, pool: dict, seed: int | str):
    costs = pool["costs"]
    labels = pool["labels"]
    auc = candidate_auc(labels, costs)
    per_ep = first_success_rank(costs, labels)
    ranks = per_ep["first_success_ranks"]
    ranks_arr = np.asarray(ranks, dtype=np.float64) if ranks else np.asarray([], dtype=np.float64)
    success_episodes = int(labels.any(axis=1).sum())
    total_episodes = int(labels.shape[0])
    first_success_mean = float(ranks_arr.mean()) if ranks else None
    first_success_median = float(np.median(ranks_arr)) if ranks else None
    out = {
        "pool": name,
        "seed": seed,
        "episodes": total_episodes,
        "candidates_per_episode": int(costs.shape[1]),
        "candidate_auc": auc,
        "episodes_with_success": success_episodes,
        "episodes_without_success": per_ep["no_success_count"],
        "first_success_rank_mean": first_success_mean,
        "first_success_rank_median": first_success_median,
        "near_miss_failure_count": per_ep["near_miss_failure_count"],
        "rank0_minus_best_success_mean": float(np.mean(per_ep["rank0_minus_best_success"])) if per_ep["rank0_minus_best_success"] else None,
        "rank0_minus_best_success_median": float(np.median(per_ep["rank0_minus_best_success"])) if per_ep["rank0_minus_best_success"] else None,
    }
    for k in [1, 3, 5, 10, 30]:
        out[f"top{k}_success_recall"] = float(per_ep["topk_hits"][k] / total_episodes * 100.0)
    return out, per_ep


def histogram_rows(pool: str, seed: int | str, per_ep: dict):
    rows = []
    for r in per_ep["first_success_ranks"]:
        rows.append({"pool": pool, "seed": seed, "rank": int(r), "count": 1})
    rows.append({"pool": pool, "seed": seed, "rank": "NO_SUCCESS", "count": int(per_ep["no_success_count"])})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    hist_rows = []
    case_rows = []
    for seed in SEEDS:
        bsl = load_pool(BSL_ROOT, "baseline", seed)
        st = load_pool(ST_ROOT, "vf05_mix20", seed)
        union = build_union(bsl, st)

        for pool_name, pool in [("bsl", bsl), ("stateroll", st), ("union", union)]:
            row, per_ep = summarize_pool(pool_name, pool, seed)
            summary_rows.append(row)
            hist_rows.extend(histogram_rows(pool_name, seed, per_ep))
            if pool_name == "union":
                # Episode-level case rows for union only.
                costs = pool["costs"]
                labels = pool["labels"]
                order = np.argsort(costs, axis=1, kind="stable")
                for ep in range(labels.shape[0]):
                    lab = labels[ep]
                    c = costs[ep]
                    success_idx = np.where(lab)[0]
                    if success_idx.size == 0:
                        continue
                    best_success = int(success_idx[np.argmin(c[success_idx])])
                    first_rank = int(np.where(lab[order[ep]])[0][0] + 1)
                    case_rows.append(
                        {
                            "seed": seed,
                            "episode": ep,
                            "best_success_index": best_success,
                            "best_success_cost": float(c[best_success]),
                            "rank0_cost": float(c[0]),
                            "rank0_minus_best_success": float(c[0] - c[best_success]),
                            "first_success_rank": first_rank,
                            "rank0_is_success": bool(lab[0]),
                            "rank0_failure": bool(not lab[0]),
                            "has_success": True,
                            "near_miss_failure": bool(not lab[0]),
                        }
                    )

    # Aggregate overall rows
    overall_rows = []
    for pool_name in ["bsl", "stateroll", "union"]:
        sub = [r for r in summary_rows if r["pool"] == pool_name]
        total_eps = sum(r["episodes"] for r in sub)
        total_success = sum(r["episodes_with_success"] for r in sub)
        total_no_success = sum(r["episodes_without_success"] for r in sub)
        overall = {
            "pool": pool_name,
            "seed": "ALL",
            "episodes": total_eps,
            "candidates_per_episode": sub[0]["candidates_per_episode"] if sub else None,
            "candidate_auc": float(binary_auc(
                np.concatenate([load_pool(BSL_ROOT if pool_name == "bsl" else ST_ROOT, "baseline" if pool_name == "bsl" else "vf05_mix20", s)["labels"].reshape(-1) for s in SEEDS]).astype(np.int64),
                -np.concatenate([load_pool(BSL_ROOT if pool_name == "bsl" else ST_ROOT, "baseline" if pool_name == "bsl" else "vf05_mix20", s)["costs"].reshape(-1) for s in SEEDS]).astype(np.float64)
            )) if pool_name in {"bsl", "stateroll"} else float(binary_auc(
                np.concatenate([build_union(load_pool(BSL_ROOT, "baseline", s), load_pool(ST_ROOT, "vf05_mix20", s))["labels"].reshape(-1) for s in SEEDS]).astype(np.int64),
                -np.concatenate([build_union(load_pool(BSL_ROOT, "baseline", s), load_pool(ST_ROOT, "vf05_mix20", s))["costs"].reshape(-1) for s in SEEDS]).astype(np.float64)
            )),
            "episodes_with_success": total_success,
            "episodes_without_success": total_no_success,
            "first_success_rank_mean": float(np.mean([x for r in sub for x in ([r["first_success_rank_mean"]] if r["first_success_rank_mean"] is not None else [])])) if sub else None,
            "first_success_rank_median": float(np.median([x for r in sub if r["first_success_rank_median"] is not None for x in [r["first_success_rank_median"]]])) if sub else None,
            "near_miss_failure_count": sum(r["near_miss_failure_count"] for r in sub),
            "rank0_minus_best_success_mean": float(np.mean([x for r in sub if r["rank0_minus_best_success_mean"] is not None for x in [r["rank0_minus_best_success_mean"]]])) if sub else None,
            "rank0_minus_best_success_median": float(np.median([x for r in sub if r["rank0_minus_best_success_median"] is not None for x in [r["rank0_minus_best_success_median"]]])) if sub else None,
        }
        for k in [1, 3, 5, 10, 30]:
            overall[f"top{k}_success_recall"] = float(sum(r[f"top{k}_success_recall"] * r["episodes"] for r in sub) / total_eps) if total_eps else None
        overall_rows.append(overall)

    all_rows = summary_rows + overall_rows

    # Write summary CSV/JSON
    with (OUT / "candidate_cost_success_alignment_n100.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    (OUT / "candidate_cost_success_alignment_n100.json").write_text(json.dumps({"per_seed": summary_rows, "overall": overall_rows, "cases": case_rows}, indent=2))

    # Histogram
    with (OUT / "success_rank_histogram_n100.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pool", "seed", "rank", "count"])
        writer.writeheader()
        # convert per-episode ranks to histogram counts
        hist_counts = []
        for pool in ["bsl", "stateroll", "union"]:
            for seed in SEEDS:
                if pool == "bsl":
                    pool_data = load_pool(BSL_ROOT, "baseline", seed)
                elif pool == "stateroll":
                    pool_data = load_pool(ST_ROOT, "vf05_mix20", seed)
                else:
                    pool_data = build_union(load_pool(BSL_ROOT, "baseline", seed), load_pool(ST_ROOT, "vf05_mix20", seed))
                per_ep = first_success_rank(pool_data["costs"], pool_data["labels"])
                ranks = per_ep["first_success_ranks"]
                counter = {}
                for r in ranks:
                    counter[r] = counter.get(r, 0) + 1
                for rank, count in sorted(counter.items(), key=lambda x: x[0]):
                    hist_counts.append({"pool": pool, "seed": seed, "rank": int(rank), "count": int(count)})
                hist_counts.append({"pool": pool, "seed": seed, "rank": "NO_SUCCESS", "count": int(per_ep["no_success_count"])})
        writer.writerows(hist_counts)

    with (OUT / "near_miss_failure_cases_n100.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0].keys()) if case_rows else ["seed", "episode"])
        writer.writeheader()
        writer.writerows(case_rows)

    # Markdown summary
    lines = [
        "# Candidate Cost-Success Alignment n100",
        "",
        "This is the actual candidate-level alignment audit computed directly from the n100 candidate pools.",
        "",
        "## Overall",
        "",
        "|pool|cand AUC|episodes w/ success|episodes no success|first-success rank mean|first-success rank median|top1 recall|top3 recall|top5 recall|top10 recall|top30 recall|near-miss failures|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in overall_rows:
        lines.append(
            f"|{r['pool']}|{r['candidate_auc']:.3f}|{r['episodes_with_success']}|{r['episodes_without_success']}|"
            f"{r['first_success_rank_mean']:.2f}|{r['first_success_rank_median']:.2f}|"
            f"{r['top1_success_recall']:.1f}|{r['top3_success_recall']:.1f}|{r['top5_success_recall']:.1f}|{r['top10_success_recall']:.1f}|{r['top30_success_recall']:.1f}|{r['near_miss_failure_count']}|"
        )
    lines += [
        "",
        "## Per-seed",
        "",
        "|pool|seed|cand AUC|episodes w/ success|episodes no success|first-success rank mean|first-success rank median|top1 recall|top3 recall|top5 recall|top10 recall|top30 recall|near-miss failures|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"|{r['pool']}|{r['seed']}|{r['candidate_auc']:.3f}|{r['episodes_with_success']}|{r['episodes_without_success']}|"
            f"{r['first_success_rank_mean']:.2f}|{r['first_success_rank_median']:.2f}|"
            f"{r['top1_success_recall']:.1f}|{r['top3_success_recall']:.1f}|{r['top5_success_recall']:.1f}|{r['top10_success_recall']:.1f}|{r['top30_success_recall']:.1f}|{r['near_miss_failure_count']}|"
        )
    (OUT / "candidate_cost_success_alignment_n100.md").write_text("\n".join(lines) + "\n")

    print(f"Wrote candidate-level alignment outputs to {OUT}")


if __name__ == "__main__":
    main()
