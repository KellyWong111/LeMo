from __future__ import annotations

import json
import math
import re
from pathlib import Path

STABLE = Path('/data1/jingyixi/.stable_worldmodel')
OUTDIR = Path('/data1/jingyixi/wm_runs/official_state_roll_l003_ep1')
POLICY_DIR = STABLE / 'pusht_official_clean_stateroll_l003_ep1'
SEEDS = [42, 43, 44, 45, 46, 47]
BASELINE = {
    'seed42': 95.0,
    'seed43': 95.0,
    'seed44': 95.0,
    'seed45': 100.0,
    'seed46': 85.0,
    'seed47': 85.0,
    'mean': 92.5,
    'std': 5.5901699437494745,
    'min': 85.0,
    'max': 100.0,
}


def parse_success(path: Path):
    if not path.exists():
        return None
    text = path.read_text(errors='ignore')
    match = re.search(r"success_rate['\"]?\s*[:=]\s*([0-9.]+)", text)
    if match is None:
        match = re.search(r"'success_rate':\s*np\.float64\(([0-9.]+)\)", text)
    if match is None:
        match = re.search(r"'success_rate':\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {'mean': None, 'std': None, 'min': None, 'max': None, 'n': 0}
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {'mean': mean, 'std': math.sqrt(var), 'min': min(vals), 'max': max(vals), 'n': len(vals)}


def fmt(x):
    return 'NA' if x is None else f'{x:.1f}'


rows = []
method_by_seed = {}
for seed in SEEDS:
    path = POLICY_DIR / f'official_stateroll_l003_ep1_seed{seed}_h4_s300_k30_n30.txt'
    value = parse_success(path)
    method_by_seed[f'seed{seed}'] = value
    rows.append({'seed': seed, 'success_rate': value, 'file': str(path)})

method_stats = stats([method_by_seed[f'seed{s}'] for s in SEEDS])
method = {**method_by_seed, **method_stats}

raw = {
    'baseline_official_original_ep13': BASELINE,
    'official_state_roll_l003_ep1': method,
    'per_seed_files': rows,
    'delta_vs_baseline': {
        'mean': None if method['mean'] is None else method['mean'] - BASELINE['mean'],
        'min': None if method['min'] is None else method['min'] - BASELINE['min'],
        'max': None if method['max'] is None else method['max'] - BASELINE['max'],
    },
}
OUTDIR.mkdir(parents=True, exist_ok=True)
(OUTDIR / 'raw_results.json').write_text(json.dumps(raw, indent=2), encoding='utf-8')

lines = []
lines.append('# official clean + state-roll lambda=0.03 ep1')
lines.append('')
lines.append('|method|seed42|seed43|seed44|seed45|seed46|seed47|mean|std|min|max|n|delta_mean|')
lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
base_cells = [BASELINE[f'seed{s}'] for s in SEEDS]
lines.append(
    '|official original ep13|'
    + '|'.join(fmt(v) for v in base_cells)
    + f"|{fmt(BASELINE['mean'])}|{fmt(BASELINE['std'])}|{fmt(BASELINE['min'])}|{fmt(BASELINE['max'])}|6|0.0|"
)
method_cells = [method_by_seed[f'seed{s}'] for s in SEEDS]
lines.append(
    '|official + state-roll l003 ep1|'
    + '|'.join(fmt(v) for v in method_cells)
    + f"|{fmt(method['mean'])}|{fmt(method['std'])}|{fmt(method['min'])}|{fmt(method['max'])}|{method['n']}|{fmt(raw['delta_vs_baseline']['mean'])}|"
)
lines.append('')
if method['n'] == len(SEEDS):
    if method['mean'] > BASELINE['mean']:
        verdict = 'official+state-roll improves over official ep13; this supports state-roll as a general planning-aware auxiliary.'
    elif method['mean'] >= BASELINE['mean'] - 2.5:
        verdict = 'official+state-roll is roughly tied with official ep13; this suggests official already has strong rollout geometry, while state-roll is not a clear universal booster.'
    else:
        verdict = 'official+state-roll drops below official ep13; this supports the interpretation that state-roll mainly repairs MoDA/gate07-induced degradation rather than improving an already strong official model.'
    lines.append('Verdict: ' + verdict)
else:
    lines.append(f"Progress: {method['n']}/6 eval seeds parsed. Verdict pending.")
lines.append('')
lines.append('Baseline reference: official original ep13 = 95/95/95/100/85/85, mean 92.5, min 85, max 100.')
(OUTDIR / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
