#!/usr/bin/env python3
import ast
import json
import re
from pathlib import Path

STABLE = Path('/data1/jingyixi/.stable_worldmodel')
COST = Path('/data1/jingyixi/wm_runs/cost_gap')
PAIRS = [
    ('+1', 'pred6_ep4', 'gate07_ep1', 'pred6_epoch4', 'pred6_gate07_epoch1'),
    ('+4', 'pred6_ep7', 'gate07_ep4', 'pred6_epoch7', 'pred6_gate07_epoch4'),
    ('+7', 'pred6_ep10', 'gate07_ep7', 'pred6_epoch10', 'pred6_gate07_epoch7'),
]
DIRS = {
    'pred6_ep4': STABLE/'pusht_encoder_moda_v14_full_visible_bs32_pred6',
    'pred6_ep7': STABLE/'pusht_encoder_moda_v14_full_visible_bs32_pred6',
    'pred6_ep10': STABLE/'pusht_encoder_moda_v14_full_visible_bs32_pred6',
    'gate07_ep1': STABLE/'pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07',
    'gate07_ep4': STABLE/'pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07',
    'gate07_ep7': STABLE/'pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07',
}

def read_success(name, seed):
    p = DIRS[name] / f'{name}_seed{seed}_s300_n30_k30.txt'
    if not p.exists() and seed == 42:
        # historical filenames before multi-seed naming
        old = {
            'pred6_ep4': 'pred6_ep4_s300_n30_k30.txt',
            'pred6_ep7': 'pred6_ep7_s300_n30_k30.txt',
            'pred6_ep10': 'pred6_ep10_s300_n30_k30.txt',
            'gate07_ep1': 'pred6_gate07_ep1_s300_n30_k30.txt',
            'gate07_ep4': 'pred6_gate07_ep4_s300_n30_k30.txt',
            'gate07_ep7': 'pred6_gate07_ep7_s300_n30_k30.txt',
        }
        p = DIRS[name] / old[name]
    if not p.exists():
        return None
    txt = p.read_text(errors='ignore')
    m = re.search(r"success_rate':\s*([0-9.]+)", txt)
    return float(m.group(1)) if m else None

def read_margin(key):
    p = COST / f'{key}.json'
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get('top2_margin_mean')

seeds = [42, 43, 44]
print('| relative | pred6 margin | gate07 margin | pred6 planning mean | gate07 planning mean | pred6 seeds | gate07 seeds |')
print('|---|---:|---:|---:|---:|---|---|')
for rel, pred, gate, pred_cost, gate_cost in PAIRS:
    pred_vals = [read_success(pred, s) for s in seeds]
    gate_vals = [read_success(gate, s) for s in seeds]
    pred_clean = [v for v in pred_vals if v is not None]
    gate_clean = [v for v in gate_vals if v is not None]
    pred_mean = sum(pred_clean)/len(pred_clean) if pred_clean else None
    gate_mean = sum(gate_clean)/len(gate_clean) if gate_clean else None
    fmt = lambda x: 'NA' if x is None else f'{x:.3f}'
    fmtp = lambda x: 'NA' if x is None else f'{x:.1f}%'
    print(f'| {rel} | {fmt(read_margin(pred_cost))} | {fmt(read_margin(gate_cost))} | {fmtp(pred_mean)} | {fmtp(gate_mean)} | {pred_vals} | {gate_vals} |')
