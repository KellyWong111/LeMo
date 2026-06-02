import csv
import json
from pathlib import Path

base = Path("/data1/jingyixi/wm_runs/pac_moda_v2_selector_v3_detector_gate_20260529")
rows = list(csv.DictReader((base / "pac_moda_v2_score_ensemble_pareto.csv").open()))
best_oof = [r for r in rows if r.get("method") == "ensemble_oof"]

md = [
    "# PAC-MoDA v2 Selector v3 Audit",
    "",
    "Selector v3 uses detector-v2 gate `logistic fp_le_3_max30`, not the original fixed precision gate. The inherited report sentence saying fixed precision gates is stale and should be read as detector-v2 gate restricted deployment.",
    "",
    "Thresholds are selected on train seeds only in `train_thresholds`. OOF harmed budgets are train-side budgets, so the best OOF row can have `harmed=2` even for `harmed_budget=0`.",
    "",
    "Best current operating point:",
    "",
    "|w_bce|w_rank|w_raw|candidate_topk|fixed|harmed|net|switches|st-only recovered|",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in best_oof[:1]:
    md.append(
        "|{w_bce}|{w_rank}|{w_raw}|{candidate_topk}|{fixed}|{harmed}|{net}|{switches}|{stateroll_only_recovered}|".format(
            **row
        )
    )
md += [
    "",
    "Interpretation: this is a balanced/aggressive operating point, not the harmed=0 conservative result.",
]

(base / "pac_moda_v2_selector_v3_audit.md").write_text("\n".join(md) + "\n")
(base / "pac_moda_v2_selector_v3_audit.json").write_text(
    json.dumps({"best_oof": best_oof[:4], "caveat": "uses detector-v2 gate; harmed_budget is train-side"}, indent=2) + "\n"
)
print((base / "pac_moda_v2_selector_v3_audit.md").read_text())
