from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize_case_file(path: Path) -> dict:
    data = json.load(open(path))
    top1 = np.asarray(data["top1_episode_successes"], dtype=bool)
    oracle = np.asarray(data["oracle_episode_successes"], dtype=bool)
    first = data["first_success_rank"]
    costs = np.asarray(data["topk_costs"], dtype=float)

    misrank = np.nonzero((~top1) & oracle)[0].tolist()
    nohit = np.nonzero((~top1) & (~oracle))[0].tolist()
    selected_success = np.nonzero(top1)[0].tolist()

    cost_spreads = costs[:, -1] - costs[:, 0]
    top1_top2 = costs[:, 1] - costs[:, 0] if costs.shape[1] > 1 else np.zeros(costs.shape[0])
    success_ranks = [r for r in first if r is not None]

    def mean_for(indices, values):
        if not indices:
            return None
        return float(np.asarray(values)[indices].mean())

    return {
        "file": str(path),
        "settings": data.get("settings", {}),
        "top1_success_rate": data["top1_success_rate"],
        "oracle_topk_success_rate": data["oracle_topk_success_rate"],
        "oracle_gap": data["oracle_topk_success_rate"] - data["top1_success_rate"],
        "counts": {
            "num_eval": int(len(top1)),
            "top1_success": int(top1.sum()),
            "oracle_success": int(oracle.sum()),
            "misrank_rescuable": int(len(misrank)),
            "no_success_in_topk": int(len(nohit)),
        },
        "episodes": {
            "selected_success": selected_success,
            "misrank_rescuable": misrank,
            "no_success_in_topk": nohit,
            "first_success_rank": first,
        },
        "cost_diagnostics": {
            "mean_top1_top2_margin_all": float(top1_top2.mean()),
            "mean_top1_top2_margin_misrank": mean_for(misrank, top1_top2),
            "mean_top1_top2_margin_nohit": mean_for(nohit, top1_top2),
            "mean_topk_cost_spread_all": float(cost_spreads.mean()),
            "mean_topk_cost_spread_misrank": mean_for(misrank, cost_spreads),
            "mean_topk_cost_spread_nohit": mean_for(nohit, cost_spreads),
            "mean_first_success_rank": None if not success_ranks else float(np.mean(success_ranks)),
            "max_first_success_rank": None if not success_ranks else int(max(success_ranks)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summaries = {Path(p).stem: summarize_case_file(Path(p)) for p in args.inputs}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
