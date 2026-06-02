from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data1/jingyixi/wm_runs")

OUT = ROOT / "overnight_moda_mechanism_summary_n100_20260529"
OUT.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def flatten_record(prefix: str, obj: dict) -> dict:
    flat = {}
    for k, v in obj.items():
        if isinstance(v, (dict, list)):
            flat[f"{prefix}{k}"] = json.dumps(v, ensure_ascii=False)
        else:
            flat[f"{prefix}{k}"] = v
    return flat


def load_candidate_complementarity():
    p = ROOT / "candidate_complementarity_n100.json"
    if not p.exists():
        p = ROOT / "candidate_complementarity_n100_20260528" / "candidate_complementarity_n100.json"
    return read_json(p)


def load_gate_detector():
    return read_json(ROOT / "gate_only_opportunity_detector_n100_20260528" / "gate_only_opportunity_detector_n100.json")


def load_precision_gate_grid():
    return read_json(ROOT / "gate_only_opportunity_detector_n100_20260528" / "precision_gate_grid_n100.json")


def load_fixable_decomp():
    return read_json(ROOT / "env_traj_replacement_n100_20260527_104100" / "fixable_failure_decomposition.json")


def summarize_cost_success():
    # Existing artifacts already contain the needed candidate/rank information for the narrative.
    # We summarize what is directly available without retraining or re-scoring models.
    comp = load_candidate_complementarity()
    grid = load_precision_gate_grid()
    decomp = load_fixable_decomp()
    det = load_gate_detector()

    rows = []
    # cost-success alignment is approximated from the existing detector/grid summaries:
    # AUC and capture/FP tradeoffs are the available published evidence in these artifacts.
    for r in det["records"]:
        rows.append(
            {
                "split": r["split"],
                "model": r["model"],
                "kind": "gate_detector",
                "auc": r["auc"],
                "top10_st_only_capture": r["top10_st_only_capture"],
                "top10_false_positive_bsl_success": r["top10_false_positive_bsl_success"],
                "top20_st_only_capture": r["top20_st_only_capture"],
                "top20_false_positive_bsl_success": r["top20_false_positive_bsl_success"],
                "top50_st_only_capture": r["top50_st_only_capture"],
                "top50_false_positive_bsl_success": r["top50_false_positive_bsl_success"],
            }
        )
    write_csv(OUT / "cost_success_alignment_n100.csv", rows)

    md = [
        "# Cost-Success Alignment n100",
        "",
        "This summary is read-only and uses existing detector / complementarity / decomposition artifacts.",
        "",
        "## Gate detector summary",
        "",
        "|split|model|AUC|top10 st-only|top20 st-only|top50 st-only|top50 FP bsl-success|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in det["records"]:
        md.append(
            f"|{r['split']}|{r['model']}|{r['auc']:.3f}|"
            f"{r['top10_st_only_capture']}/{r['top10_st_only_total']}|"
            f"{r['top20_st_only_capture']}/{r['top20_st_only_total']}|"
            f"{r['top50_st_only_capture']}/{r['top50_st_only_total']}|"
            f"{r['top50_false_positive_bsl_success']}|"
        )
    md.append("")
    md.append("## Complementarity")
    overall = comp["overall"]
    md += [
        f"- bsl top1: {overall['bsl_top1_count']}/{overall['episodes']} ({overall['bsl_top1']:.1f}%)",
        f"- bsl oracle: {overall['bsl_oracle_count']}/{overall['episodes']} ({overall['bsl_oracle']:.1f}%)",
        f"- stateroll top1: {overall['stateroll_top1_count']}/{overall['episodes']} ({overall['stateroll_top1']:.1f}%)",
        f"- stateroll oracle: {overall['stateroll_oracle_count']}/{overall['episodes']} ({overall['stateroll_oracle']:.1f}%)",
        f"- union oracle: {overall['union_oracle_count']}/{overall['episodes']} ({overall['union_oracle']:.1f}%)",
        f"- stateroll-only success: {overall['stateroll_only_success_count']}/{overall['episodes']} ({overall['stateroll_only_success']:.1f}%)",
        "",
        "## Fixable decomposition",
        f"- detector_missed: {decomp['overall']['detector_missed']['count']}",
        f"- selector_wrong: {decomp['overall']['detector_hit_but_selector_wrong']['count']}",
        f"- gate_rejected: {decomp['overall']['selector_correct_but_gate_rejected']['count']}",
        f"- successfully_fixed: {decomp['overall']['successfully_fixed']['count']}",
    ]
    (OUT / "cost_success_alignment_n100.md").write_text("\n".join(md) + "\n")
    (OUT / "cost_success_alignment_n100.json").write_text(
        json.dumps({"detector": det, "complementarity": comp, "fixable_decomposition": decomp}, indent=2)
    )


def summarize_representation_quality():
    # Use existing training summaries as the strongest available readout.
    artifacts = {
        "pool_aware": ROOT / "pool_aware_planning_alignment_n100_20260528" / "summary.md",
        "c_strict": ROOT / "moda_c_strict_alignment_n100_20260528" / "summary.md",
        "os_moda_ra": ROOT / "os_moda_ra_n100_20260528" / "summary.md",
        "opportunity_residual": ROOT / "opportunity_conditioned_moda_residual_n100_20260528" / "summary.md",
    }
    rows = []
    for name, path in artifacts.items():
        if path.exists():
            rows.append({"method": name, "summary_path": str(path), "summary": path.read_text()[:2000]})
    write_csv(OUT / "representation_quality_bsl_vs_moda.csv", rows)
    md = ["# Representation Quality BSL vs MoDA", ""]
    for row in rows:
        md.append(f"## {row['method']}")
        md.append(f"- source: `{row['summary_path']}`")
        md.append("")
        md.append("```text")
        md.append(row["summary"])
        md.append("```")
        md.append("")
    (OUT / "representation_quality_bsl_vs_moda.md").write_text("\n".join(md) + "\n")
    (OUT / "representation_quality_bsl_vs_moda.json").write_text(json.dumps({"records": rows}, indent=2))


def summarize_complementarity():
    comp = load_candidate_complementarity()
    rows = comp["per_seed"]
    write_csv(OUT / "complementarity_report_n100.csv", rows)
    md = ["# Complementarity Report n100", "", "## Overall", ""]
    overall = comp["overall"]
    md += [
        "|metric|count|percent|",
        "|---|---:|---:|",
        f"|bsl top1|{overall['bsl_top1_count']}|{overall['bsl_top1']:.1f}|",
        f"|bsl oracle|{overall['bsl_oracle_count']}|{overall['bsl_oracle']:.1f}|",
        f"|stateroll top1|{overall['stateroll_top1_count']}|{overall['stateroll_top1']:.1f}|",
        f"|stateroll oracle|{overall['stateroll_oracle_count']}|{overall['stateroll_oracle']:.1f}|",
        f"|union oracle|{overall['union_oracle_count']}|{overall['union_oracle']:.1f}|",
        f"|bsl-only success|{overall['bsl_only_success_count']}|{overall['bsl_only_success']:.1f}|",
        f"|stateroll-only success|{overall['stateroll_only_success_count']}|{overall['stateroll_only_success']:.1f}|",
        f"|both success|{overall['both_success_count']}|{overall['both_success']:.1f}|",
        f"|neither success|{overall['neither_success_count']}|{overall['neither_success']:.1f}|",
        "",
        "## BSL failures",
        "",
        "|metric|count|percent of bsl failures|",
        "|---|---:|---:|",
        f"|bsl rank0 failures|{overall['bsl_rank0_failure_count']}|100.0|",
        f"|fixable by any pool|{overall['bsl_rank0_failure_fixable_any_count']}|{overall['bsl_rank0_failure_fixable_any']:.1f}|",
        f"|fixable by bsl pool only|{overall['bsl_rank0_failure_fixable_bsl_only_count']}|{overall['bsl_rank0_failure_fixable_bsl_only']:.1f}|",
        f"|fixable by stateroll pool only|{overall['bsl_rank0_failure_fixable_stateroll_only_count']}|{overall['bsl_rank0_failure_fixable_stateroll_only']:.1f}|",
        f"|fixable by both pools|{overall['bsl_rank0_failure_fixable_both_count']}|{overall['bsl_rank0_failure_fixable_both']:.1f}|",
        "",
        "## Per seed",
        "",
        "|seed|bsl top1|bsl oracle|st top1|st oracle|union oracle|bsl only|st only|both|neither|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"|{r['seed']}|{r['bsl_top1']:.1f}|{r['bsl_oracle']:.1f}|{r['stateroll_top1']:.1f}|{r['stateroll_oracle']:.1f}|{r['union_oracle']:.1f}|"
            f"{r['bsl_only_success_count']}|{r['stateroll_only_success_count']}|{r['both_success_count']}|{r['neither_success_count']}|"
        )
    (OUT / "complementarity_report_n100.md").write_text("\n".join(md) + "\n")
    (OUT / "complementarity_report_n100.json").write_text(json.dumps(comp, indent=2))


def summarize_detector_missed():
    decomp = load_fixable_decomp()
    rows = [r for r in read_json(ROOT / "env_traj_replacement_n100_20260527_104100" / "fixable_failure_decomposition.json")["overall"].keys()]
    # Use the detailed CSV as the source of per-episode rows.
    csv_path = ROOT / "env_traj_replacement_n100_20260527_104100" / "fixable_failure_decomposition.csv"
    raw = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            raw.append(row)
    missed = [r for r in raw if r["category"] == "detector_missed"]
    captured = [r for r in raw if r["category"] != "detector_missed"]
    bsl_fp = [r for r in raw if r["category"] == "detector_hit_but_selector_wrong"]
    summary_rows = [
        {"group": "detector_missed", "n": len(missed)},
        {"group": "captured_fixable", "n": len(captured)},
        {"group": "bsl_success_fp", "n": len(bsl_fp)},
    ]
    write_csv(OUT / "detector_missed_analysis_n100.csv", summary_rows)
    md = [
        "# Detector Missed Analysis n100",
        "",
        f"- detector_missed: {len(missed)}",
        f"- captured_fixable: {len(captured)}",
        f"- bsl_success_fp: {len(bsl_fp)}",
        "",
        "This is a read-only high-level partition derived from the fixed-policy decomposition.",
    ]
    (OUT / "detector_missed_analysis_n100.md").write_text("\n".join(md) + "\n")
    (OUT / "detector_missed_analysis_n100.json").write_text(json.dumps({"missed": missed, "captured": captured, "bsl_success_fp": bsl_fp}, indent=2))


def summarize_fixed_gate_reranker():
    grid = load_precision_gate_grid()
    rows = []
    for r in grid["records"]:
        if r["split"] == "splitA_train42_44_val45_47" and r["rule"] == "top10+st_gap_bottom20+AND" and r["model"] == "extratrees":
            rows.append({"split": r["split"], "rule": r["rule"], "model": r["model"], "selected": r["selected_count"], "st_only": r["stateroll_only_captured"], "fp": r["bsl_success_false_positive"]})
        if r["split"] == "splitB_train45_47_val42_44" and r["rule"] == "top10+abs_gap_top10+AND" and r["model"] in {"randomforest", "extratrees"}:
            rows.append({"split": r["split"], "rule": r["rule"], "model": r["model"], "selected": r["selected_count"], "st_only": r["stateroll_only_captured"], "fp": r["bsl_success_false_positive"]})
    write_csv(OUT / "fixed_gate_conservative_reranker_n100.csv", rows)
    (OUT / "fixed_gate_conservative_reranker_n100.json").write_text(json.dumps({"records": rows}, indent=2))
    md = ["# Fixed Gate Conservative Reranker n100", ""]
    for r in rows:
        md.append(f"- {r['split']} {r['model']} {r['rule']}: selected={r['selected']} st-only={r['st_only']} fp={r['fp']}")
    (OUT / "fixed_gate_conservative_reranker_n100.md").write_text("\n".join(md) + "\n")


def main():
    summarize_cost_success()
    summarize_representation_quality()
    summarize_complementarity()
    summarize_detector_missed()
    summarize_fixed_gate_reranker()
    print(f"Wrote summaries to {OUT}")


if __name__ == "__main__":
    main()
