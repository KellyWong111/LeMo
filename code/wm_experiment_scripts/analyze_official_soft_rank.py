from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def zscore_row(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    exp = np.exp(x)
    return exp / exp.sum()


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))


def labels_from_rank_major(candidate_successes_by_rank):
    # Stored as [rank][env]. Convert to [env][rank].
    return np.asarray(candidate_successes_by_rank, dtype=bool).T


def topm_metrics(labels: np.ndarray, student_costs: np.ndarray, teacher_costs: np.ndarray, tau: float, topm: int):
    n_env, topk = labels.shape
    m = min(topm, topk)
    rows = []
    for env_idx in range(n_env):
        y = labels[env_idx]
        s_cost = student_costs[env_idx]
        t_cost = teacher_costs[env_idx]
        t_order = np.argsort(t_cost)
        s_order = np.argsort(s_cost)
        t_elite = t_order[:m]
        s_z = zscore_row(s_cost)
        t_z = zscore_row(t_cost)
        p_teacher = softmax(-t_z / tau)
        p_student = softmax(-s_z / tau)
        best_in_teacher_elite_by_student = t_elite[np.argmin(s_cost[t_elite])]
        rows.append(
            {
                "teacher_topm_has_success": bool(y[t_elite].any()),
                "student_select_within_teacher_topm_success": bool(y[best_in_teacher_elite_by_student]),
                "student_mass_on_teacher_topm": float(p_student[t_elite].sum()),
                "teacher_mass_on_teacher_topm": float(p_teacher[t_elite].sum()),
                "success_mass_teacher": float(p_teacher[y].sum()) if y.any() else None,
                "success_mass_student": float(p_student[y].sum()) if y.any() else None,
                "kl_teacher_student": kl_div(p_teacher, p_student),
                "teacher_argmin_success": bool(y[t_order[0]]),
                "student_argmin_success": bool(y[s_order[0]]),
                "oracle_success": bool(y.any()),
            }
        )
    return rows


def mean_optional(vals):
    vals = [v for v in vals if v is not None]
    return None if not vals else float(np.mean(vals))


def summarize_group(items, taus, topms):
    out = {
        "n_rows": len(items),
        "candidate_top1": float(np.mean([r["candidate_cost_top1_success"] for r in items])),
        "teacher_argmin": float(np.mean([r["official_teacher_argmin_success"] for r in items])),
        "oracle": float(np.mean([r["oracle_topk_success"] for r in items])),
        "teacher_auc": mean_optional([r.get("teacher_success_auc") for r in items]),
        "candidate_auc": mean_optional([r.get("candidate_success_auc") for r in items]),
    }
    for tau in taus:
        for topm in topms:
            rows = []
            for item in items:
                labels = labels_from_rank_major(item["candidate_successes_by_rank"])
                student_costs = np.asarray(item["candidate_costs"], dtype=float)
                teacher_costs = np.asarray(item["official_teacher_costs"], dtype=float)
                rows.extend(topm_metrics(labels, student_costs, teacher_costs, tau=tau, topm=topm))
            key = f"tau{tau:g}_top{topm}"
            out[key] = {
                "teacher_topm_oracle": float(np.mean([r["teacher_topm_has_success"] for r in rows]) * 100.0),
                "student_within_teacher_topm": float(np.mean([r["student_select_within_teacher_topm_success"] for r in rows]) * 100.0),
                "student_mass_on_teacher_topm": float(np.mean([r["student_mass_on_teacher_topm"] for r in rows])),
                "teacher_mass_on_teacher_topm": float(np.mean([r["teacher_mass_on_teacher_topm"] for r in rows])),
                "success_mass_teacher": mean_optional([r["success_mass_teacher"] for r in rows]),
                "success_mass_student": mean_optional([r["success_mass_student"] for r in rows]),
                "kl_teacher_student": float(np.mean([r["kl_teacher_student"] for r in rows])),
                "teacher_argmin": float(np.mean([r["teacher_argmin_success"] for r in rows]) * 100.0),
                "student_argmin": float(np.mean([r["student_argmin_success"] for r in rows]) * 100.0),
                "oracle": float(np.mean([r["oracle_success"] for r in rows]) * 100.0),
            }
    return out


def fmt(x, digits=1):
    return "NA" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def build_markdown(grouped, taus, topms):
    lines = ["# Official-guided soft/listwise ranking diagnostic", ""]
    lines += [
        "## Base rerank",
        "|candidate_pool|n|candidate_top1|teacher_argmin|oracle|teacher_auc|candidate_auc|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in sorted(grouped.items()):
        lines.append(
            f"|{name}|{stats['n_rows']}|{fmt(stats['candidate_top1'])}|{fmt(stats['teacher_argmin'])}|"
            f"{fmt(stats['oracle'])}|{fmt(stats['teacher_auc'],3)}|{fmt(stats['candidate_auc'],3)}|"
        )
    for tau in taus:
        lines += ["", f"## Soft/Listwise tau={tau:g}"]
        lines += [
            "|candidate_pool|top_m|teacher_topm_oracle|student_within_teacher_topm|student_mass_on_teacher_topm|success_mass_teacher|success_mass_student|KL teacher->student|",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, stats in sorted(grouped.items()):
            for topm in topms:
                row = stats[f"tau{tau:g}_top{topm}"]
                lines.append(
                    f"|{name}|{topm}|{fmt(row['teacher_topm_oracle'])}|{fmt(row['student_within_teacher_topm'])}|"
                    f"{fmt(row['student_mass_on_teacher_topm'],3)}|{fmt(row['success_mass_teacher'],3)}|"
                    f"{fmt(row['success_mass_student'],3)}|{fmt(row['kl_teacher_student'],3)}|"
                )
    lines += [""]
    lines += [
        "Interpretation guide:",
        "- teacher_topm_oracle: whether official top-m contains at least one successful candidate.",
        "- student_within_teacher_topm: if restricted to official top-m, whether MoDA cost selects a successful candidate.",
        "- student_mass_on_teacher_topm: how much softmax(-zscore(J_moda)/tau) probability is assigned to official top-m.",
        "- KL teacher->student: direct listwise distillation loss diagnostic.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--taus", default="0.5,1.0,2.0")
    parser.add_argument("--topms", default="5,10")
    args = parser.parse_args()
    rows = json.loads(Path(args.input).read_text())
    taus = [float(x) for x in args.taus.split(",") if x]
    topms = [int(x) for x in args.topms.split(",") if x]
    by_name = {}
    for row in rows:
        by_name.setdefault(row["name"], []).append(row)
    grouped = {name: summarize_group(items, taus, topms) for name, items in by_name.items()}
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "soft_rank_summary.json").write_text(json.dumps(grouped, indent=2))
    md = build_markdown(grouped, taus, topms)
    (outdir / "soft_rank_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
