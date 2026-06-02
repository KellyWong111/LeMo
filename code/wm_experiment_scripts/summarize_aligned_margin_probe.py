import json
import re
from pathlib import Path


STABLE = Path("/data1/jingyixi/.stable_worldmodel")
MARGIN = Path("/data1/jingyixi/wm_runs/cost_gap_aligned")
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
    return [v == "True" for v in re.findall(r"True|False", match.group(1))]


def margin(name, seed):
    p = MARGIN / f"{name}_seed{seed}_n20_c64.json"
    d = json.loads(p.read_text())
    return d["per_env_top2_margin"], d["top2_margin_mean"], d["top5_margin_mean"], d["top1_mean"]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    print("| rel | model | seed | succ | top2 mean | top5 mean | top1 mean | succ margin | fail margin |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for rel, pred, gate in PAIRS:
        for name in [pred, gate]:
            all_succ = []
            all_margin = []
            for seed in [42, 43, 44]:
                ss = successes(name, seed)
                per, top2, top5, top1 = margin(name, seed)
                succ_m = [m for m, s in zip(per, ss) if s]
                fail_m = [m for m, s in zip(per, ss) if not s]
                all_succ.extend(ss)
                all_margin.extend(per)
                print(
                    f"| {rel} | {name} | {seed} | {sum(ss)}/20 | "
                    f"{top2:.3f} | {top5:.3f} | {top1:.3f} | {mean(succ_m):.3f} | {mean(fail_m):.3f} |"
                )
            succ_m = [m for m, s in zip(all_margin, all_succ) if s]
            fail_m = [m for m, s in zip(all_margin, all_succ) if not s]
            print(
                f"| {rel} | {name} | all | {sum(all_succ)}/60 | "
                f"{mean(all_margin):.3f} | - | - | {mean(succ_m):.3f} | {mean(fail_m):.3f} |"
            )


if __name__ == "__main__":
    main()
