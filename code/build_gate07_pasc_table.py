from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path("/data1/jingyixi/wm_runs")
OUT = ROOT / "prediction_vs_planning"


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def parse_success_rate(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    patterns = [
        r"success_rate': np\.float64\(([^)]+)\)",
        r"success_rate': ([0-9.]+)",
        r'"success_rate": ([0-9.]+)',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return float(m.group(1))
    return None


def load_pred_loss() -> dict[int, float]:
    pred_loss: dict[int, float] = {}
    old_table = read_json(OUT / "gate07_prediction_vs_planning_table.json")
    if isinstance(old_table, list):
        for row in old_table:
            value = row.get("val_pred_loss")
            if value not in (None, ""):
                pred_loss[int(row["epoch"])] = float(value)
    return pred_loss


def load_default_success() -> dict[int, float]:
    data = read_json(ROOT / "latest_planning_new" / "summary_seed42.json") or {}
    default = {
        int(row["epoch"]): float(row["success_rate"])
        for row in data.get("gate07", [])
    }
    base = Path(
        "/data1/jingyixi/.stable_worldmodel/"
        "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07"
    )
    ep4_default = parse_success_rate(base / "gate07_ep4_s42_e20_s300_n30_k30.txt")
    if ep4_default is None:
        ep4_default = parse_success_rate(base / "pred6_gate07_ep4_s300_n30_k30.txt")
    if ep4_default is not None:
        default[4] = ep4_default
    return default


def load_strong_success() -> tuple[dict[int, float], dict[int, dict[int, float]]]:
    strong: dict[int, float] = {}
    seeds: dict[int, dict[int, float]] = {}
    data = read_json(ROOT / "best_models_strong_cem" / "summary.json") or []
    for row in data:
        m = re.search(r"gate07_ep(\d+)", row["tag"])
        if m:
            strong[int(m.group(1))] = float(row["success_rate"])

    hparam = read_json(ROOT / "cem_hparam_sweep_now" / "summary.json") or []
    for row in hparam:
        if row.get("tag") == "gate07_ep4_e20_s1000_k100_n30":
            strong[4] = float(row["success_rate"])
            seeds.setdefault(4, {})[42] = float(row["success_rate"])

    base = Path(
        "/data1/jingyixi/.stable_worldmodel/"
        "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07"
    )
    for seed in [43, 44]:
        value = parse_success_rate(base / f"gate07_ep4_seed{seed}_s1000_k100_n30.txt")
        if value is not None:
            seeds.setdefault(4, {})[seed] = value
    return strong, seeds


def load_geometry() -> dict[int, dict]:
    geom: dict[int, dict] = {}
    trace_paths = [
        OUT / "gate07_ep2_ep3_trace.json",
        ROOT / "oracle_sweep" / "cem_trace_gate07_ep4_e20_s300_k30_steps30.json",
        OUT / "gate07_ep9_ep16_ep17_trace.json",
    ]
    for path in trace_paths:
        data = read_json(path)
        if not data:
            continue
        if "models" in data:
            for name, rec in data["models"].items():
                m = re.search(r"ep(\d+)", name)
                if m:
                    geom[int(m.group(1))] = rec.get("aggregate", {})
        elif "aggregate" in data:
            geom[4] = data["aggregate"]
    return geom


def load_oracles() -> dict[int, dict]:
    data = read_json(ROOT / "oracle_sweep" / "summary.json") or {}
    oracles: dict[int, dict] = {}
    ep4 = data.get("oracle", {}).get("gate07_ep4_e20_top100_s1000", {})
    if ep4:
        oracles[4] = ep4
    ep9_path = OUT / "gate07_ep9_top30_s300_oracle.json"
    ep9 = read_json(ep9_path)
    if ep9:
        oracles[9] = {
            "oracle_topk_success_rate": ep9.get("oracle_topk_success_rate"),
            "oracle_gap": ep9.get("oracle_topk_success_rate", 0)
            - ep9.get("top1_success_rate", 0),
        }
    ep16_path = OUT / "gate07_ep16_top30_s300_oracle.json"
    ep16 = read_json(ep16_path)
    if ep16:
        oracles[16] = {
            "oracle_topk_success_rate": ep16.get("oracle_topk_success_rate"),
            "oracle_gap": ep16.get("oracle_topk_success_rate", 0)
            - ep16.get("top1_success_rate", 0),
        }
    return oracles


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    pred_loss = load_pred_loss()
    default = load_default_success()
    strong, strong_seeds = load_strong_success()
    geom = load_geometry()
    oracles = load_oracles()

    rows = []
    for ep in [2, 3, 4, 9, 16, 17]:
        g = geom.get(ep, {})
        seed_note = "; ".join(
            f"seed{seed}:{value:.1f}"
            for seed, value in sorted(strong_seeds.get(ep, {}).items())
        )
        rows.append(
            {
                "epoch": ep,
                "val_pred_loss": pred_loss.get(ep),
                "default_cem_success_seed42": default.get(ep),
                "strong_cem_success": strong.get(ep),
                "strong_cem_multiseed_note": seed_note,
                "oracle_topk_success": oracles.get(ep, {}).get("oracle_topk_success_rate"),
                "oracle_gap": oracles.get(ep, {}).get("oracle_gap"),
                "top2_margin_mean": g.get("top2_margin_mean"),
                "cost_std_mean": g.get("cost_std_mean"),
                "candidate_traj_spread_mean": g.get("candidate_traj_spread_mean"),
                "topk_traj_spread_mean": g.get("topk_traj_spread_mean"),
                "top1_cost_mean": g.get("top1_cost_mean"),
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate07_pasc_table.json").write_text(json.dumps(rows, indent=2))

    headers = list(rows[0].keys())
    lines = [
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(fmt(row.get(h)) for h in headers) + "|")
    table = "\n".join(lines) + "\n"
    (OUT / "gate07_pasc_table.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
