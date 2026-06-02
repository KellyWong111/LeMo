from __future__ import annotations

import json
import re
from pathlib import Path


STABLE = Path(
    "/data1/jingyixi/.stable_worldmodel/"
    "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07"
)
OUT = Path("/data1/jingyixi/wm_runs/horizon_sensitivity")


def parse_success(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    m = re.search(r"success_rate':\s*([0-9.]+)", text)
    if m:
        return float(m.group(1))
    return None


def main() -> None:
    rows = []
    for ep in [4, 9, 16]:
        rows.append(
            {
                "epoch": ep,
                "h3_rh3_15step": parse_success(
                    STABLE / f"gate07_ep{ep}_h3_rh3_b5_s300_n30_k30.txt"
                ),
                "h5_rh5_25step_default": parse_success(
                    STABLE / f"gate07_ep{ep}_s42_e20_s300_n30_k30.txt"
                )
                if ep == 4
                else parse_success(STABLE / f"gate07_ep{ep}_seed42_s300_n30_k30.txt"),
                "h10_rh5_50step": parse_success(
                    STABLE / f"gate07_ep{ep}_h10_b5_seed42_s300_n30_k30.txt"
                ),
                "strong_h5_s1000_k100": parse_success(
                    STABLE / f"gate07_ep{ep}_s1000_k100_n30.txt"
                )
                if ep != 4
                else parse_success(STABLE / "gate07_ep4_e20_s1000_k100_n30.txt"),
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate07_horizon_sensitivity_summary.json").write_text(
        json.dumps(rows, indent=2)
    )
    headers = list(rows[0].keys())
    lines = [
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        vals = []
        for h in headers:
            v = row[h]
            vals.append("" if v is None else f"{v:.1f}" if isinstance(v, float) else str(v))
        lines.append("|" + "|".join(vals) + "|")
    table = "\n".join(lines) + "\n"
    (OUT / "gate07_horizon_sensitivity_summary.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
