from __future__ import annotations

import json
from pathlib import Path

OUT = Path("/data1/jingyixi/wm_runs/env_traj_replacement_n100_20260527_104100")
SPLITS = ["splitA_train42_44_val45_47", "splitB_train45_47_val42_44"]


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


def key(r):
    return (
        int(r["epoch"]),
        int(r["detector_topk"]),
        float(r["score_threshold"]),
        int(r["vote_k"]),
        float(r["max_switch_frac"]),
    )


def precision(r) -> float:
    return float(r.get("detector_fixable_captured", 0)) / max(1, int(r["detector_topk"]))


def slim(r):
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
    d = {k: r[k] for k in keys if k in r}
    d["detector_precision"] = precision(r)
    return d


def aggregate(rows):
    total = sum(ps["episodes"] for r in rows for ps in r["per_seed"])
    bsl = sum(ps["bsl_top1"] * ps["episodes"] / 100 for r in rows for ps in r["per_seed"])
    sel = sum(ps["selector_top1"] * ps["episodes"] / 100 for r in rows for ps in r["per_seed"])
    oracle = sum(ps["oracle"] * ps["episodes"] / 100 for r in rows for ps in r["per_seed"])
    fixed = sum(ps["fixed_vs_bsl"] for r in rows for ps in r["per_seed"])
    harmed = sum(ps["harmed_vs_bsl"] for r in rows for ps in r["per_seed"])
    switches = sum(ps["switches"] for r in rows for ps in r["per_seed"])
    return {
        "episodes": total,
        "bsl_top1": 100 * bsl / total,
        "selector_top1": 100 * sel / total,
        "oracle": 100 * oracle / total,
        "fixed_vs_bsl": fixed,
        "harmed_vs_bsl": harmed,
        "switches": switches,
    }


def main():
    train = {s: {key(r): r for r in load_rows(s, "train")} for s in SPLITS}
    val = {s: {key(r): r for r in load_rows(s, "val")} for s in SPLITS}
    common = sorted(set(train[SPLITS[0]]) & set(train[SPLITS[1]]) & set(val[SPLITS[0]]) & set(val[SPLITS[1]]))
    analysis = {"source": str(OUT), "num_common_configs": len(common)}

    candidates = []
    for k in common:
        a = train[SPLITS[0]][k]
        b = train[SPLITS[1]][k]
        if a["harmed_vs_bsl"] != 0 or b["harmed_vs_bsl"] != 0:
            continue
        if a["switches"] < 1 and b["switches"] < 1:
            continue
        if float(a["max_switch_frac"]) > 0.10:
            continue
        if int(a["detector_topk"]) > 20:
            continue
        candidates.append(k)

    def train_obj(k):
        a = train[SPLITS[0]][k]
        b = train[SPLITS[1]][k]
        fixed_a, fixed_b = a["fixed_vs_bsl"], b["fixed_vs_bsl"]
        total_switch = a["switches"] + b["switches"]
        maxsw = float(a["max_switch_frac"])
        return (min(fixed_a, fixed_b), fixed_a + fixed_b, -total_switch, -maxsw)

    selected = max(candidates, key=train_obj) if candidates else None
    if selected:
        train_rows = [train[s][selected] for s in SPLITS]
        val_rows = [val[s][selected] for s in SPLITS]
        analysis["selected_global_config"] = {
            "key": selected,
            "split_train": {s: slim(train[s][selected]) for s in SPLITS},
            "split_val": {s: slim(val[s][selected]) for s in SPLITS},
            "train_aggregate": aggregate(train_rows),
            "val_oof_aggregate": aggregate(val_rows),
        }
    else:
        analysis["selected_global_config"] = None

    val_safe = []
    for k in common:
        va = val[SPLITS[0]][k]
        vb = val[SPLITS[1]][k]
        if va["harmed_vs_bsl"] != 0 or vb["harmed_vs_bsl"] != 0:
            continue
        if va["switches"] < 1 and vb["switches"] < 1:
            continue
        val_safe.append(k)

    def val_obj(k):
        ag = aggregate([val[s][k] for s in SPLITS])
        return (ag["selector_top1"], ag["fixed_vs_bsl"], -ag["switches"], -float(val[SPLITS[0]][k]["max_switch_frac"]))

    best_val = max(val_safe, key=val_obj) if val_safe else None
    if best_val:
        rows = [val[s][best_val] for s in SPLITS]
        analysis["oracle_style_val_harmed0_shared_config"] = {
            "key": best_val,
            "split_val": {s: slim(val[s][best_val]) for s in SPLITS},
            "val_oof_aggregate": aggregate(rows),
        }
    else:
        analysis["oracle_style_val_harmed0_shared_config"] = None

    analysis["top_train_global_candidates"] = []
    for k in sorted(candidates, key=train_obj, reverse=True)[:20]:
        analysis["top_train_global_candidates"].append(
            {
                "key": k,
                "trainA": slim(train[SPLITS[0]][k]),
                "trainB": slim(train[SPLITS[1]][k]),
                "valA": slim(val[SPLITS[0]][k]),
                "valB": slim(val[SPLITS[1]][k]),
                "val_oof": aggregate([val[s][k] for s in SPLITS]),
            }
        )

    (OUT / "robust_global_gate_analysis.json").write_text(json.dumps(analysis, indent=2))
    lines = [
        "# Robust Global Gate Analysis",
        "",
        f"Source: `{OUT}`",
        "",
        f"Common configs: {len(common)}",
        f"Train-side robust candidates: {len(candidates)}",
        "",
        "## Selected Shared Gate From Train",
        "",
    ]
    sel = analysis["selected_global_config"]
    if sel is None:
        lines.append("No shared train-side config satisfied the robust constraints.")
    else:
        k = sel["key"]
        ag = sel["val_oof_aggregate"]
        lines.append(f"Selected config: epoch={k[0]}, detector_topk={k[1]}, threshold={k[2]}, vote_k={k[3]}, max_switch_frac={k[4]}")
        lines.append("")
        lines.append(
            f"Held-out OOF: bsl {ag['bsl_top1']:.1f} -> selector {ag['selector_top1']:.1f}, "
            f"oracle {ag['oracle']:.1f}, fixed={ag['fixed_vs_bsl']}, harmed={ag['harmed_vs_bsl']}, switches={ag['switches']}"
        )
        lines.append("")
        lines.append("|split|train bsl|train sel|train fix|train harm|train sw|val bsl|val sel|val oracle|val fix|val harm|val sw|")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in SPLITS:
            tr = sel["split_train"][s]
            va = sel["split_val"][s]
            lines.append(
                f"|{s}|{tr['bsl_top1']:.1f}|{tr['selector_top1']:.1f}|{tr['fixed_vs_bsl']}|{tr['harmed_vs_bsl']}|{tr['switches']}|"
                f"{va['bsl_top1']:.1f}|{va['selector_top1']:.1f}|{va['oracle']:.1f}|{va['fixed_vs_bsl']}|{va['harmed_vs_bsl']}|{va['switches']}|"
            )

    lines += [
        "",
        "## Diagnostic Shared Val Harmed=0 Upper Bound",
        "",
        "This is selected on held-out val and is not an official result.",
        "",
    ]
    ov = analysis["oracle_style_val_harmed0_shared_config"]
    if ov is None:
        lines.append("No shared config has harmed_val=0 on both val splits with at least one switch.")
    else:
        k = ov["key"]
        ag = ov["val_oof_aggregate"]
        lines.append(f"Best shared val-safe config: epoch={k[0]}, detector_topk={k[1]}, threshold={k[2]}, vote_k={k[3]}, max_switch_frac={k[4]}")
        lines.append("")
        lines.append(
            f"Diagnostic OOF upper bound: bsl {ag['bsl_top1']:.1f} -> selector {ag['selector_top1']:.1f}, "
            f"oracle {ag['oracle']:.1f}, fixed={ag['fixed_vs_bsl']}, harmed={ag['harmed_vs_bsl']}, switches={ag['switches']}"
        )
        lines.append("")
        lines.append("|split|val bsl|val sel|val oracle|val fix|val harm|val sw|")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for s in SPLITS:
            va = ov["split_val"][s]
            lines.append(f"|{s}|{va['bsl_top1']:.1f}|{va['selector_top1']:.1f}|{va['oracle']:.1f}|{va['fixed_vs_bsl']}|{va['harmed_vs_bsl']}|{va['switches']}|")

    (OUT / "robust_global_gate_summary.md").write_text("\n".join(lines) + "\n")
    print((OUT / "robust_global_gate_summary.md").read_text())


if __name__ == "__main__":
    main()
