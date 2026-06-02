import csv
import json
from pathlib import Path

root = Path("/data1/jingyixi/wm_runs")
out = root / "pac_moda_v2_selector_v3_detector_gate_20260529"


def sum_deploy(path, method):
    rows = list(csv.DictReader(path.open()))
    rows = [r for r in rows if r.get("method") == method]
    return {
        "fixed": sum(int(r["fixed"]) for r in rows),
        "harmed": sum(int(r["harmed"]) for r in rows),
        "net": sum(int(r["fixed"]) - int(r["harmed"]) for r in rows),
        "switches": sum(int(r["switches"]) for r in rows),
        "st": sum(int(r["stateroll_only_recovered"]) for r in rows),
    }


corrected = root / "pac_moda_v2_full_n100_corrected_20260529" / "pac_moda_v2_ablation_deployment_n100.csv"
detector_v2 = root / "pac_moda_v2_opportunity_detector_v2_20260529" / "pac_moda_v2_detector_v2_oof_deployment.csv"
selector_v3 = out / "pac_moda_v2_score_ensemble_pareto.csv"

rows = []
safe = sum_deploy(corrected, "legacy_rank_combined")
rows.append({"mode": "conservative_rank_preserve", "fixed": safe["fixed"], "harmed": safe["harmed"], "net": safe["net"], "switches": safe["switches"], "stateroll_only_recovered": safe["st"], "approx_top1": 81.0 + safe["net"] / 600 * 100})

det = list(csv.DictReader(detector_v2.open()))
best_det = sorted(det, key=lambda r: (int(r["net"]), int(r["fixed"]), -int(r["harmed"])), reverse=True)[0]
rows.append({"mode": "detector_v2_gate_rank_preserve", "fixed": int(best_det["fixed"]), "harmed": int(best_det["harmed"]), "net": int(best_det["net"]), "switches": int(best_det["switches"]), "stateroll_only_recovered": int(best_det["stateroll_only_recovered"]), "approx_top1": 81.0 + int(best_det["net"]) / 600 * 100})

sel = [r for r in csv.DictReader(selector_v3.open()) if r.get("method") == "ensemble_oof"]
best_sel = sorted(sel, key=lambda r: (int(r["net"]), int(r["fixed"]), -int(r["harmed"])), reverse=True)[0]
rows.append({"mode": "selector_v3_balanced", "fixed": int(best_sel["fixed"]), "harmed": int(best_sel["harmed"]), "net": int(best_sel["net"]), "switches": int(best_sel["switches"]), "stateroll_only_recovered": int(best_sel["stateroll_only_recovered"]), "approx_top1": 81.0 + int(best_sel["net"]) / 600 * 100})

with (out / "pac_moda_v2_paper_summary.csv").open("w", newline="") as f:
    keys = list(rows[0].keys())
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
(out / "pac_moda_v2_paper_summary.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")

md = ["# PAC-MoDA v2 Paper Summary", "", "|mode|fixed|harmed|net|switches|st-only recovered|approx top1|", "|---|---:|---:|---:|---:|---:|---:|"]
for r in rows:
    md.append("|{mode}|{fixed}|{harmed}|{net}|{switches}|{stateroll_only_recovered}|{approx_top1:.2f}|".format(**r))
md += [
    "",
    "Use `conservative_rank_preserve` as the harmed=0 safe result.",
    "Use `selector_v3_balanced` as the aggressive/balanced operating point; it has harmed=2 and should not be described as harmed-free.",
]
(out / "pac_moda_v2_paper_summary.md").write_text("\n".join(md) + "\n")
print((out / "pac_moda_v2_paper_summary.md").read_text())
