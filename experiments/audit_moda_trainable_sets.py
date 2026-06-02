from __future__ import annotations

import sys
from pathlib import Path

import stable_worldmodel as swm

REPO = Path("/data1/jingyixi/.cache_runtime/LeWM_src/le-wm-official-clean")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "wm_experiment_scripts"))
from wm_experiment_scripts.pool_coverage_compare_variants import POLICIES


def main():
    model = swm.policy.AutoCostModel(POLICIES["stateroll_l003_ep1"], cache_dir="/data1/jingyixi/.stable_worldmodel")
    params = list(model.named_parameters())
    modules = dict(model.named_modules())
    keywords = ["moda", "retriev", "gate", "visible", "depth", "mix", "proj"]
    print("MODEL", type(model).__name__)
    print("encoder", type(model.encoder).__name__)
    print("predictor", type(model.predictor).__name__)

    print("\nENCODER MODULES matching keywords:")
    for name, module in modules.items():
        if not name.startswith("encoder"):
            continue
        text = (name + " " + type(module).__name__).lower()
        if any(k in text for k in keywords) or "attention" in text or "block" in text:
            print(name, type(module).__name__)

    print("\nENCODER PARAMS matching keywords or MoDA modules:")
    matched = []
    for name, p in params:
        if not name.startswith("encoder"):
            continue
        parent = name.rsplit(".", 1)[0]
        ptype = type(modules.get(parent, None)).__name__ if parent in modules else ""
        text = (name + " " + ptype).lower()
        if any(k in text for k in keywords) or "moda" in ptype.lower() or "attention" in ptype.lower():
            matched.append((name, tuple(p.shape), p.numel(), ptype))
            print(name, tuple(p.shape), p.numel(), "parent", ptype)

    print("\nCOUNTS")
    print("matched_encoder_params", len(matched), sum(x[2] for x in matched))

    def count(predicates):
        selected = []
        for name, p in params:
            if any(fn(name) for fn in predicates):
                selected.append((name, p.numel()))
        return selected

    set_a = count(
        [
            lambda n: n.startswith("pred_proj."),
            lambda n: n.startswith("predictor.transformer.layers.5."),
            lambda n: n.startswith("predictor.transformer.norm."),
        ]
    )
    late_layers = (10, 11)
    set_b_broad = count(
        [
            lambda n: any(n.startswith(f"encoder.transformer.layers.{i}.") for i in late_layers)
            and any(k in n.lower() for k in ["attn", "gate", "proj", "mlp", "norm"]),
            lambda n: n.startswith("encoder.transformer.norm."),
        ]
    )
    set_b_strict = count(
        [
            lambda n: any(n.startswith(f"encoder.transformer.layers.{i}.attn") for i in late_layers),
            lambda n: any(n.startswith(f"encoder.transformer.layers.{i}.mlp.gate_proj") for i in late_layers),
            lambda n: any(
                n.startswith(f"encoder.transformer.layers.{i}.attn_norm")
                or n.startswith(f"encoder.transformer.layers.{i}.mlp_norm")
                for i in late_layers
            ),
            lambda n: n.startswith("encoder.transformer.norm."),
        ]
    )
    set_c_broad = {name: c for name, c in set_a + set_b_broad}
    set_c_strict = {name: c for name, c in set_a + set_b_strict}

    print("\nTRAINABLE SET A predictor_only")
    print("tensors", len(set_a), "params", sum(c for _, c in set_a))
    for name, c in set_a:
        print(name, c)

    print("\nTRAINABLE SET B moda_late_only last2 broad")
    print("tensors", len(set_b_broad), "params", sum(c for _, c in set_b_broad))
    for name, c in set_b_broad:
        print(name, c)

    print("\nTRAINABLE SET B_strict moda_late_only last2 strict")
    print("tensors", len(set_b_strict), "params", sum(c for _, c in set_b_strict))
    for name, c in set_b_strict:
        print(name, c)

    print("\nTRAINABLE SET C moda_late_plus_predictor broad")
    print("tensors", len(set_c_broad), "params", sum(set_c_broad.values()))

    print("\nTRAINABLE SET C_strict moda_late_plus_predictor")
    print("tensors", len(set_c_strict), "params", sum(set_c_strict.values()))


if __name__ == "__main__":
    main()
