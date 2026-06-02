import csv
import json
from pathlib import Path

root = Path("/data1/jingyixi/wm_runs")
orig = root / "pac_moda_v2_full_n100_20260529"
corr = root / "pac_moda_v2_full_n100_corrected_20260529"
v2 = root / "cost_calibration_v2_n100_20260529"
budget = root / "pac_moda_v2_budget_generalization_20260529"


def sum_deploy(path: Path, method: str):
    rows = list(csv.DictReader((path / "pac_moda_v2_ablation_deployment_n100.csv").open()))
    rows = [r for r in rows if r.get("method") == method]
    return {
        "fixed": sum(int(r["fixed"]) for r in rows),
        "harmed": sum(int(r["harmed"]) for r in rows),
        "switches": sum(int(r["switches"]) for r in rows),
        "stateroll_only_recovered": sum(int(r["stateroll_only_recovered"]) for r in rows),
    }


def loso_sum(path: Path):
    rows = list(csv.DictReader((path / "pac_moda_v2_loso_n100.csv").open()))
    return {
        "fixed": sum(int(r["fixed"]) for r in rows),
        "harmed": sum(int(r["harmed"]) for r in rows),
        "switches": sum(int(r["switches"]) for r in rows),
        "stateroll_only_recovered": sum(int(r["stateroll_only_recovered"]) for r in rows),
    }


corrected_legacy = sum_deploy(corr, "legacy_rank_combined")
corrected_full = sum_deploy(corr, "full_bce_pairwise_listwise_preserve")
orig_legacy = sum_deploy(orig, "legacy_rank_combined")
loso_corrected = loso_sum(corr)

report = {
    "status": "issues_found_corrected_report_written",
    "audited_dirs": [str(orig), str(v2), str(budget)],
    "corrected_dir": str(corr),
    "checks": {
        "episode_level_fixed_harmed_switches": {
            "status": "pass",
            "detail": "eval_final_selection groups by (seed, episode); selected_rows length equals switches and fixed/harmed are counted once per switched episode.",
        },
        "loso_train_eval_separation": {
            "status": "partial_pass_with_caveat",
            "detail": "Held seed is excluded from training data and threshold selection. Fixed gate is reused from splitA/splitB precomputed gate rows, not learned from held labels in the LOSO run. Original legacy normalization used all feature rows and is deprecated; corrected run uses train-only normalization.",
        },
        "calibrated_topk_nearmiss_order": {
            "status": "pass",
            "detail": "rank_metrics orders by np.argsort(-scores), and near_miss checks labels[order[0]], so calibrated top-k/near-miss use calibrated order.",
        },
        "legacy_vs_full_definition": {
            "status": "pass_with_required_caveat",
            "detail": "legacy_rank_combined is ranking/listwise/preserve without BCE; full_bce_pairwise_listwise_preserve is BCE plus pairwise/listwise/preserve. They must not be merged in text.",
        },
        "budget_feature_normalization": {
            "status": "issue_found",
            "detail": "Budget generalization uses the same feature definitions and train-only normalization in its legacy fitter. n50 has no matching fixed gate grid, so deployment comparison is unavailable. Earlier old full/cost_calibration_v2 ranking variants used all-X normalization and are deprecated for strict deployment claims.",
        },
    },
    "deprecated_numbers": {
        "pac_moda_v2_full_n100_20260529 legacy_rank_combined": "deprecated due to all-X normalization leakage in fit_combined_ranker",
        "cost_calibration_v2_n100_20260529 pairwise/listwise/combined": "deprecated for strict split claims due to all-X normalization in ranking fitters; BCE remains train-normalized",
    },
    "old_legacy_oof": orig_legacy,
    "corrected_legacy_oof": corrected_legacy,
    "corrected_full_bce_pairwise_listwise_preserve_oof": corrected_full,
    "corrected_loso_legacy": loso_corrected,
    "main_result_valid": "corrected legacy/rank-preserve PAC-MoDA v2 OOF fixed=6, harmed=0, switches=8, stateroll-only recovered=6",
}

md = [
    "# PAC-MoDA v2 Result Audit",
    "",
    "## Verdict",
    "",
    "Issue found. The old `legacy_rank_combined` result in `pac_moda_v2_full_n100_20260529` used all-candidate feature normalization inside `fit_combined_ranker`, so it touched validation/held feature distribution. I reran a corrected version with train-only normalization in `pac_moda_v2_full_n100_corrected_20260529`.",
    "",
    "The corrected main result remains valid: `legacy_rank_combined` / rank-preserve PAC-MoDA v2 OOF `fixed=6`, `harmed=0`, `switches=8`, `stateroll-only recovered=6`.",
    "",
    "Old `fixed=6, harmed=0, switches=9` should be deprecated and replaced by corrected `fixed=6, harmed=0, switches=8`.",
    "",
    "## Audit Checks",
    "",
]
for key, value in report["checks"].items():
    md.append(f"- `{key}`: **{value['status']}**. {value['detail']}")
md.extend(["", "## Deprecated Outputs", ""])
for key, value in report["deprecated_numbers"].items():
    md.append(f"- `{key}`: {value}")
md.extend(
    [
        "",
        "## Corrected OOF Deployment",
        "",
        "|method|fixed|harmed|switches|stateroll-only recovered|",
        "|---|---:|---:|---:|---:|",
        f"|legacy_rank_combined corrected|{corrected_legacy['fixed']}|{corrected_legacy['harmed']}|{corrected_legacy['switches']}|{corrected_legacy['stateroll_only_recovered']}|",
        f"|full_bce_pairwise_listwise_preserve corrected|{corrected_full['fixed']}|{corrected_full['harmed']}|{corrected_full['switches']}|{corrected_full['stateroll_only_recovered']}|",
        "",
        "## Corrected LOSO",
        "",
        f"Corrected LOSO legacy total: fixed={loso_corrected['fixed']}, harmed={loso_corrected['harmed']}, switches={loso_corrected['switches']}, stateroll-only recovered={loso_corrected['stateroll_only_recovered']}.",
        "",
        "This means A/B OOF remains strong, but LOSO is weaker after strict train-only normalization. Report LOSO as robustness evidence, not as another fixed=6 claim.",
    ]
)

for directory in [orig, corr, v2, budget]:
    (directory / "pac_moda_v2_result_audit.md").write_text("\n".join(md) + "\n")
    (directory / "pac_moda_v2_result_audit.json").write_text(json.dumps(report, indent=2) + "\n")

print("wrote clean audit reports")
print(json.dumps(report["corrected_legacy_oof"], indent=2))
