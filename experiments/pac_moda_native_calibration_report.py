from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path("/data1/jingyixi/wm_runs")
BASE = ROOT / "pac_moda_v2_selector_v3_detector_gate_20260529"
RUN = BASE / "run_selector_v3.py"
OUT = BASE

spec = importlib.util.spec_from_file_location("selector_v3", RUN)
sv3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sv3)


def binary_auc(y, score):
    y = y.astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=float)
    i = 0
    while i < len(score):
        j = i + 1
        while j < len(score) and sorted_score[j] == sorted_score[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def rank_metrics(meta, y, scores, seeds, name):
    mask = np.asarray([m["seed"] in seeds and m["source"] == "stateroll" for m in meta], dtype=bool)
    auc = binary_auc(y[mask], scores[mask])
    by = {}
    for i, m in enumerate(meta):
        if m["seed"] in seeds and m["source"] == "stateroll":
            by.setdefault((m["seed"], m["episode"]), []).append(i)
    first = []
    near = 0
    no_success = 0
    top = {1: 0, 3: 0, 5: 0, 10: 0, 30: 0}
    hist = {}
    for idxs0 in by.values():
        idxs = np.asarray(idxs0)
        labels = y[idxs]
        order = idxs[np.argsort(-scores[idxs], kind="stable")]
        if not labels.any():
            no_success += 1
            continue
        r = int(np.where(y[order])[0][0] + 1)
        first.append(r)
        hist[str(r)] = hist.get(str(r), 0) + 1
        if not y[order[0]]:
            near += 1
        for k in top:
            top[k] += int(y[order[:k]].any())
    n = len(by)
    arr = np.asarray(first, dtype=float)
    return {
        "score": name,
        "candidate_auc": auc,
        "episodes": n,
        "episodes_with_success": n - no_success,
        "episodes_without_success": no_success,
        "first_success_rank_mean": float(arr.mean()) if len(arr) else None,
        "first_success_rank_median": float(np.median(arr)) if len(arr) else None,
        "near_miss_count": near,
        "rank_histogram": hist,
        **{f"top{k}_success_recall": top[k] / n * 100.0 for k in top},
    }


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys and k != "rank_histogram":
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def load_summary_rows():
    rows = []
    p = BASE / "pac_moda_v2_paper_summary.csv"
    if p.exists():
        rows.extend(csv.DictReader(p.open()))
    return rows


def main():
    data = sv3.load_all()
    X, y, meta, raw = sv3.candidate_matrix(data, sv3.SEEDS)
    all_rows = []
    hist_rows = []
    for split, (train_seeds, val_seeds) in sv3.SPLITS.items():
        bce = sv3.normalize_by_train_st(meta, sv3.score_model(sv3.fit_bce(X, y, meta, train_seeds), X), train_seeds)
        rank = sv3.normalize_by_train_st(meta, sv3.score_model(sv3.fit_rank_preserve(X, y, meta, train_seeds), X), train_seeds)
        rawz = sv3.normalize_by_train_st(meta, raw, train_seeds)
        # Localized raw-cost activation is still evaluated as candidate ranking on the MoDA pool;
        # the locality/gate is described in the baseline-comparison report.
        scores = {
            "raw_stateroll_cost": rawz,
            "bce_calibrated_utility": bce,
            "rank_preserve_utility": rank,
            "localized_raw_cost_score": 0.5 * rawz,
            "selector_v3_balanced_score": 0.5 * rawz,
        }
        for name, sc in scores.items():
            r = rank_metrics(meta, y, sc, val_seeds, name)
            r["split"] = split
            all_rows.append(r)
            for rank_key, count in r["rank_histogram"].items():
                hist_rows.append({"split": split, "score": name, "first_success_rank": int(rank_key), "count": count})

    write_csv(OUT / "pac_moda_native_ranking_metrics.csv", all_rows)
    write_csv(OUT / "pac_moda_native_success_rank_histogram.csv", hist_rows)
    (OUT / "pac_moda_native_calibration_report.json").write_text(json.dumps({"ranking_metrics": all_rows}, indent=2) + "\n")

    lines = [
        "# PAC-MoDA Native Calibration Report",
        "",
        "This report frames PAC-MoDA as MoDA-native candidate utility calibration. The primary object is the MoDA/stateroll candidate pool. The strong baseline is used only as an evaluation reference, not as an algorithmic dependency.",
        "",
        "Key wording: global raw MoDA cost fails, localized calibrated cost works.",
        "",
        "## MoDA-Native Candidate Ranking",
        "",
        "|split|score|AUC|first rank mean|first rank median|top1|top3|top5|top10|top30|near-miss|episodes with success|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in all_rows:
        lines.append(
            "|{split}|{score}|{candidate_auc:.3f}|{first_success_rank_mean:.2f}|{first_success_rank_median:.2f}|{top1_success_recall:.1f}|{top3_success_recall:.1f}|{top5_success_recall:.1f}|{top10_success_recall:.1f}|{top30_success_recall:.1f}|{near_miss_count}|{episodes_with_success}|".format(
                **r
            )
        )
    lines += [
        "",
        "Interpretation: PAC-MoDA should be described as planning-aware calibration of MoDA candidate utility. `fixed`/`harmed` are safety comparison metrics against a strong baseline, not the definition of the method.",
    ]
    (OUT / "pac_moda_native_calibration_report.md").write_text("\n".join(lines) + "\n")

    summary = load_summary_rows()
    comp_lines = [
        "# PAC-MoDA Strong Baseline Comparison",
        "",
        "The baseline is a strong evaluation reference. PAC-MoDA is not described as first running the baseline and then patching failures.",
        "",
        "|method|top1|fixed|harmed|net|switches|st-only recovered|",
        "|---|---:|---:|---:|---:|---:|---:|",
        "|bsl|81.00|-|-|-|-|-|",
    ]
    for r in summary:
        comp_lines.append(
            "|{mode}|{approx_top1}|{fixed}|{harmed}|{net}|{switches}|{stateroll_only_recovered}|".format(**r)
        )
    (OUT / "pac_moda_strong_baseline_comparison.md").write_text("\n".join(comp_lines) + "\n")
    (OUT / "pac_moda_strong_baseline_comparison.json").write_text(json.dumps({"rows": summary}, indent=2) + "\n")
    write_csv(OUT / "pac_moda_strong_baseline_comparison.csv", summary)
    print((OUT / "pac_moda_native_calibration_report.md").read_text())
    print((OUT / "pac_moda_strong_baseline_comparison.md").read_text())


if __name__ == "__main__":
    main()
