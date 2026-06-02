import json
import re
from pathlib import Path

stable = Path("/data1/jingyixi/.stable_worldmodel")
outdir = Path("/data1/jingyixi/wm_runs/state_rollout_final_table")
outdir.mkdir(parents=True, exist_ok=True)

settings = [
    ("l005 ep1 standard", "l005", "h4_s300_k30_n30"),
    ("l005 ep1 strong", "l005", "h4_s1000_k100_n20"),
    ("l010 ep1 standard", "l010", "h4_s300_k30_n30"),
    ("l010 ep1 strong", "l010", "h4_s1000_k100_n20"),
]
seeds = [42, 43, 44]

rows = []
for name, tag, suffix in settings:
    directory = stable / f"pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_{tag}"
    vals = []
    row = {"setting": name}
    for seed in seeds:
        path = directory / f"staterollseq_{tag}_ep1_seed{seed}_{suffix}.txt"
        val = None
        if path.exists():
            text = path.read_text(errors="ignore")
            match = re.search(r"success_rate['\"]?\s*[:=]\s*([0-9.]+)", text) or re.search(
                r"'success_rate':\s*([0-9.]+)", text
            )
            if match:
                val = float(match.group(1))
        row[f"seed{seed}"] = val
        if val is not None:
            vals.append(val)
    row["mean"] = sum(vals) / len(vals) if vals else None
    row["min"] = min(vals) if vals else None
    row["max"] = max(vals) if vals else None
    row["n"] = len(vals)
    rows.append(row)

(outdir / "summary.json").write_text(json.dumps(rows, indent=2))
lines = [
    "|setting|seed42|seed43|seed44|mean|min|max|n|",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]


def fmt(x):
    return "NA" if x is None else f"{x:.1f}"


for row in rows:
    lines.append(
        f"|{row['setting']}|{fmt(row['seed42'])}|{fmt(row['seed43'])}|{fmt(row['seed44'])}|"
        f"{fmt(row['mean'])}|{fmt(row['min'])}|{fmt(row['max'])}|{row['n']}|"
    )

(outdir / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
