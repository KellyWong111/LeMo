from __future__ import annotations

import csv
import json
from pathlib import Path

OUT = Path("/data1/jingyixi/wm_runs/env_traj_replacement_n100_20260527_104100")
SPLITS = ["splitA_train42_44_val45_47", "splitB_train45_47_val42_44"]
ALLOWED_TOPK = {5, 10, 15, 20}
ALLOWED_VOTE = {3, 4}
MAX_SWITCH_FRAC = 0.05 + 1e-12


def epoch_from_path(p: Path) -> int:
    return int(p.stem.split("epoch")[-1])


def load_rows(split: str, kind: str):
    rows = []
    for p in sorted((OUT / split).glob(f"{kind}_grid_epoch*.json")):
        ep = epoch_from_path(p)
        for r in json.loads(p.read_text()):
            rr = dict(r)
            rr["epoch"] = ep
            rows.append(rr)
    return rows


def cfg_key(r):
    return (
        r["epoch"],
        r["detector_topk"],
        float(r["score_threshold"]),
        r["vote_k"],
        float(r["max_switch_frac"]),
    )


def detector_precision(r) -> float:
    topk = max(1, int(r["detector_topk"]))
    return float(r.get("detector_fixable_captured", 0)) / topk


def compact(r):
    keys = [
        "epoch",
        "detector_topk",
        "score_threshold",
        "vote_k",
        "max_switch_frac",
        "bsl_top1",
        "selector_top1",
        "oracle",
        "fixed_vs_bsl",
        "harmed_vs_bsl",
        "switches",
        "detector_fixable_captured",
    ]
    out = {k: r[k] for k in keys if k in r}
    out["detector_precision"] = detector_precision(r)
    return out


def choose_train(train_rows):
    candidates = []
    for r in train_rows:
        if r["harmed_vs_bsl"] != 0:
            continue
        if r["switches"] < 1:
            continue
        if float(r["max_switch_frac"]) > MAX_SWITCH_FRAC:
            continue
        if int(r["detector_topk"]) not in ALLOWED_TOPK:
            continue
        if int(r["vote_k"]) not in ALLOWED_VOTE:
            continue
        candidates.append(r)
    if not candidates:
        return None, []
    best = max(
        candidates,
        key=lambda r: (
            r["fixed_vs_bsl"] - 10 * r["harmed_vs_bsl"],
            -r["switches"],
            detector_precision(r),
            r["selector_top1"],
        ),
    )
    return best, candidates


def oracle_val_zero_harm(val_rows):
    candidates = [r for r in val_rows if r["harmed_vs_bsl"] == 0 and r["switches"] >= 1]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda r: (
            r["selector_top1"],
            r["fixed_vs_bsl"],
            -r["switches"],
            detector_precision(r),
        ),
    )


def main():
    analysis = {
        "output_dir": str(OUT),
        "rule": {
            "harmed_train": 0,
            "switches_train_min": 1,
            "max_switch_frac_max": 0.05,
            "detector_topk_allowed": sorted(ALLOWED_TOPK),
            "vote_k_allowed": sorted(ALLOWED_VOTE),
            "objective": "fixed_train - 10*harmed_train; tie fewer switches then higher detector precision",
        },
    }
    per_seed_rows = []
    selected_per_seed = []
    md = [
        "# Conservative Gate Analysis",
        "",
        f"Source: `{OUT}`",
        "",
        "## Train-Selected Conservative Gate",
        "",
        "|split|epoch|topk|thr|vote|maxsw|train bsl|train sel|train fix|train harm|train sw|train det precision|val bsl|val sel|val oracle|val fix|val harm|val sw|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for split in SPLITS:
        train_rows = load_rows(split, "train")
        val_rows = load_rows(split, "val")
        lookup = {cfg_key(r): r for r in val_rows}
        chosen, candidates = choose_train(train_rows)
        oracle = oracle_val_zero_harm(val_rows)
        rec = {
            "num_train_rows": len(train_rows),
            "num_val_rows": len(val_rows),
            "num_rule_candidates": len(candidates),
        }
        if chosen is None:
            rec["selected_train"] = None
            rec["selected_val"] = None
            md.append(f"|{split}|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|")
        else:
            val = lookup[cfg_key(chosen)]
            rec["selected_train"] = compact(chosen)
            rec["selected_val"] = compact(val)
            for ps in val.get("per_seed", []):
                row = dict(ps)
                row["split"] = split
                row["selection"] = "train_selected"
                row.update(
                    {
                        "epoch": chosen["epoch"],
                        "detector_topk": chosen["detector_topk"],
                        "score_threshold": chosen["score_threshold"],
                        "vote_k": chosen["vote_k"],
                        "max_switch_frac": chosen["max_switch_frac"],
                    }
                )
                selected_per_seed.append(row)
                per_seed_rows.append(row)
            md.append(
                f"|{split}|{chosen['epoch']}|{chosen['detector_topk']}|{chosen['score_threshold']:.2f}|{chosen['vote_k']}|{chosen['max_switch_frac']:.2f}|"
                f"{chosen['bsl_top1']:.1f}|{chosen['selector_top1']:.1f}|{chosen['fixed_vs_bsl']}|{chosen['harmed_vs_bsl']}|{chosen['switches']}|{detector_precision(chosen):.2f}|"
                f"{val['bsl_top1']:.1f}|{val['selector_top1']:.1f}|{val['oracle']:.1f}|{val['fixed_vs_bsl']}|{val['harmed_vs_bsl']}|{val['switches']}|"
            )

        rec["oracle_val_harmed0_best"] = compact(oracle) if oracle else None
        if oracle:
            for ps in oracle.get("per_seed", []):
                row = dict(ps)
                row["split"] = split
                row["selection"] = "oracle_val_harmed0"
                row.update(
                    {
                        "epoch": oracle["epoch"],
                        "detector_topk": oracle["detector_topk"],
                        "score_threshold": oracle["score_threshold"],
                        "vote_k": oracle["vote_k"],
                        "max_switch_frac": oracle["max_switch_frac"],
                    }
                )
                per_seed_rows.append(row)
        rec["top_train_rule_candidates"] = [
            compact(r)
            for r in sorted(
                candidates,
                key=lambda r: (
                    r["fixed_vs_bsl"] - 10 * r["harmed_vs_bsl"],
                    -r["switches"],
                    detector_precision(r),
                    r["selector_top1"],
                ),
                reverse=True,
            )[:10]
        ]
        analysis[split] = rec

    if selected_per_seed:
        total = sum(r["episodes"] for r in selected_per_seed)
        bsl = sum(r["bsl_top1"] * r["episodes"] / 100 for r in selected_per_seed)
        sel = sum(r["selector_top1"] * r["episodes"] / 100 for r in selected_per_seed)
        oracle = sum(r["oracle"] * r["episodes"] / 100 for r in selected_per_seed)
        fixed = sum(r["fixed_vs_bsl"] for r in selected_per_seed)
        harmed = sum(r["harmed_vs_bsl"] for r in selected_per_seed)
        switches = sum(r["switches"] for r in selected_per_seed)
        analysis["train_selected_oof"] = {
            "episodes": total,
            "bsl_top1": 100 * bsl / total,
            "selector_top1": 100 * sel / total,
            "oracle": 100 * oracle / total,
            "fixed_vs_bsl": fixed,
            "harmed_vs_bsl": harmed,
            "switches": switches,
        }
        md += [
            "",
            "OOF train-selected aggregate:",
            "",
            f"- bsl {100*bsl/total:.1f} -> selector {100*sel/total:.1f}, oracle {100*oracle/total:.1f}, fixed={fixed}, harmed={harmed}, switches={switches}",
        ]

    md += [
        "",
        "## Oracle-Style Val Harmed=0 Upper Bound",
        "",
        "This is diagnostic only, selected directly on held-out val grids.",
        "",
        "|split|epoch|topk|thr|vote|maxsw|val bsl|val sel|val oracle|val fix|val harm|val sw|det precision|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        oracle = analysis[split]["oracle_val_harmed0_best"]
        if oracle is None:
            md.append(f"|{split}|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|")
        else:
            md.append(
                f"|{split}|{oracle['epoch']}|{oracle['detector_topk']}|{oracle['score_threshold']:.2f}|{oracle['vote_k']}|{oracle['max_switch_frac']:.2f}|"
                f"{oracle['bsl_top1']:.1f}|{oracle['selector_top1']:.1f}|{oracle['oracle']:.1f}|{oracle['fixed_vs_bsl']}|{oracle['harmed_vs_bsl']}|{oracle['switches']}|{oracle['detector_precision']:.2f}|"
            )

    md += ["", "## Per-Seed Table", "", "Saved as `per_seed_fixed_harmed_conservative.csv`."]

    (OUT / "conservative_grid_analysis.json").write_text(json.dumps(analysis, indent=2))
    (OUT / "conservative_selected_summary.md").write_text("\n".join(md) + "\n")
    fieldnames = [
        "selection",
        "split",
        "seed",
        "episodes",
        "bsl_top1",
        "selector_top1",
        "oracle",
        "fixed_vs_bsl",
        "harmed_vs_bsl",
        "switches",
        "epoch",
        "detector_topk",
        "score_threshold",
        "vote_k",
        "max_switch_frac",
    ]
    with (OUT / "per_seed_fixed_harmed_conservative.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_seed_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print((OUT / "conservative_selected_summary.md").read_text())


if __name__ == "__main__":
    main()
