import csv
import json
from pathlib import Path

out = Path("/data1/jingyixi/wm_runs/pac_moda_v2_opportunity_detector_v2_20260529")
rows = list(csv.DictReader((out / "pac_moda_v2_detector_v2_deployment.csv").open()))
combos = sorted(set((r["model"], r["mode"]) for r in rows))
oof = []
for combo in combos:
    rs = [r for r in rows if (r["model"], r["mode"]) == combo]
    if len(rs) != 2:
        continue
    oof.append(
        {
            "model": combo[0],
            "mode": combo[1],
            "selected": sum(int(r["selected"]) for r in rs),
            "fixed": sum(int(r["fixed"]) for r in rs),
            "harmed": sum(int(r["harmed"]) for r in rs),
            "net": sum(int(r["net"]) for r in rs),
            "switches": sum(int(r["switches"]) for r in rs),
            "stateroll_only_recovered": sum(int(r["stateroll_only_recovered"]) for r in rs),
            "bsl_success_fp": sum(int(r["bsl_success_fp"]) for r in rs),
        }
    )

with (out / "pac_moda_v2_detector_v2_oof_deployment.csv").open("w", newline="") as f:
    keys = list(oof[0].keys())
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(oof)
(out / "pac_moda_v2_detector_v2_oof_deployment.json").write_text(json.dumps({"records": oof}, indent=2) + "\n")

md = (out / "pac_moda_v2_opportunity_detector_v2.md").read_text()
lines = md.splitlines()
lines += [
    "",
    "## OOF Deployment Aggregate",
    "",
    "|model|mode|selected|fixed|harmed|net|switches|st-only recovered|bsl FP|",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in sorted(oof, key=lambda r: (r["net"], r["fixed"], -r["harmed"], -r["bsl_success_fp"]), reverse=True)[:12]:
    lines.append(
        "|{model}|{mode}|{selected}|{fixed}|{harmed}|{net}|{switches}|{stateroll_only_recovered}|{bsl_success_fp}|".format(**row)
    )
lines += [
    "",
    "Conclusion: detector v2 improves gate recall, but corrected PAC-MoDA v2 deployment remains capped at fixed=6, harmed=0 because the rank-preserve selector only switches 8 episodes. The current bottleneck is selector activation/ranking inside newly captured gate episodes, not gate recall alone.",
]
(out / "pac_moda_v2_opportunity_detector_v2.md").write_text("\n".join(lines) + "\n")
(out / "pac_moda_v2_detector_v2_deployment.md").write_text("\n".join(lines) + "\n")
print("wrote oof")
