import re
from pathlib import Path


STABLE = Path("/data1/jingyixi/.stable_worldmodel")
DIRS = {
    "pred6": STABLE / "pusht_encoder_moda_v14_full_visible_bs32_pred6",
    "gate07": STABLE / "pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07",
}
OLD = {
    ("pred6_ep4", 42): "pred6_ep4_s300_n30_k30.txt",
    ("pred6_ep7", 42): "pred6_ep7_s300_n30_k30.txt",
    ("pred6_ep10", 42): "pred6_ep10_s300_n30_k30.txt",
    ("gate07_ep1", 42): "pred6_gate07_ep1_s300_n30_k30.txt",
    ("gate07_ep4", 42): "pred6_gate07_ep4_s300_n30_k30.txt",
    ("gate07_ep7", 42): "pred6_gate07_ep7_s300_n30_k30.txt",
}
PAIRS = [
    ("+1", "pred6_ep4", "gate07_ep1"),
    ("+4", "pred6_ep7", "gate07_ep4"),
    ("+7", "pred6_ep10", "gate07_ep7"),
]


def result_path(name, seed):
    group = "gate07" if name.startswith("gate07") else "pred6"
    if (name, seed) in OLD:
        return DIRS[group] / OLD[(name, seed)]
    return DIRS[group] / f"{name}_seed{seed}_s300_n30_k30.txt"


def successes(name, seed):
    txt = result_path(name, seed).read_text(errors="ignore")
    match = re.search(r"episode_successes': array\((\[.*?\])\)", txt, re.S)
    if not match:
        raise RuntimeError(f"could not parse episode_successes: {result_path(name, seed)}")
    vals = re.findall(r"True|False", match.group(1))
    return [v == "True" for v in vals]


def main():
    print("| rel | seed | pred succ | gate succ | gate-only | pred-only | both | neither |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for rel, pred, gate in PAIRS:
        agg = [0, 0, 0, 0]
        for seed in [42, 43, 44]:
            ps = successes(pred, seed)
            gs = successes(gate, seed)
            gate_only = sum((not a) and b for a, b in zip(ps, gs))
            pred_only = sum(a and (not b) for a, b in zip(ps, gs))
            both = sum(a and b for a, b in zip(ps, gs))
            neither = sum((not a) and (not b) for a, b in zip(ps, gs))
            for i, v in enumerate([gate_only, pred_only, both, neither]):
                agg[i] += v
            print(
                f"| {rel} | {seed} | {sum(ps)}/20 | {sum(gs)}/20 | "
                f"{gate_only} | {pred_only} | {both} | {neither} |"
            )
        print(f"| {rel} | all | - | - | {agg[0]} | {agg[1]} | {agg[2]} | {agg[3]} |")


if __name__ == "__main__":
    main()
