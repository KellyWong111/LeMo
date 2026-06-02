import json, re, math
from pathlib import Path
from statistics import mean, pstdev
STABLE=Path('/data1/jingyixi/.stable_worldmodel')
OUT=Path('/data1/jingyixi/wm_runs/state_rollout_confirm_main')
BASE='pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07'
L003=BASE+'_staterollseq_l003'

def parse_success(p):
    if not p.exists(): return None
    t=p.read_text(errors='ignore')
    m=re.search(r"success_rate['\"]?\s*[:=]\s*([0-9.]+)",t) or re.search(r"'success_rate':\s*([0-9.]+)",t)
    return float(m.group(1)) if m else None

def stats(vals):
    xs=[v for v in vals if v is not None]
    if not xs: return dict(mean=None,std=None,min=None,max=None,n=0)
    return dict(mean=mean(xs),std=pstdev(xs),min=min(xs),max=max(xs),n=len(xs))

def fmt(x): return 'NA' if x is None else f'{x:.1f}'
# Experiment A
seedsA=[42,43,44,45,46,47]
base_vals=[]; meth_vals=[]
for s in seedsA:
    base_vals.append(parse_success(STABLE/BASE/f'gate07_epoch4_standard_seed{s}_h4_s300_k30_n30.txt'))
    meth_vals.append(parse_success(STABLE/L003/f'staterollseq_l003_ep1_standard_seed{s}_h4_s300_k30_n30.txt'))
base_st=stats(base_vals); meth_st=stats(meth_vals)
delta_vals=[(m-b) if m is not None and b is not None else None for b,m in zip(base_vals,meth_vals)]
delta_st=stats(delta_vals)
# Experiment B
B=[]
for label, subdir, namepat in [
    ('epoch0 original', BASE, 'gate07_epoch4_standard_seed{seed}_h4_s300_k30_n30.txt'),
    ('epoch1 l003', L003, 'staterollseq_l003_ep1_standard_seed{seed}_h4_s300_k30_n30.txt'),
    ('epoch2 l003', L003, 'staterollseq_l003_ep2_standard_seed{seed}_h4_s300_k30_n30.txt'),
    ('epoch4 l003', L003, 'staterollseq_l003_ep4_standard_seed{seed}_h4_s300_k30_n30.txt'),
]:
    vals=[parse_success(STABLE/subdir/namepat.format(seed=s)) for s in [42,43,44]]
    row={'checkpoint':label,'seed42':vals[0],'seed43':vals[1],'seed44':vals[2],**stats(vals)}
    B.append(row)
# Experiment C
C=[]
for tag in ['baseline_gate07_ep4','stateroll_l003_ep1']:
    for seed in [42,43,44]:
        p=OUT/f'cost_success_{tag}_seed{seed}.json'
        if not p.exists(): continue
        d=json.loads(p.read_text())
        labels=[]
        for rank_list in d.get('candidate_successes_by_rank',[]):
            labels.append(rank_list)
        # rank-major -> episode x rank
        if labels:
            import numpy as np
            lab=np.array(labels,dtype=bool).T
            costs=np.array(d.get('topk_costs'),dtype=float)
            succ_cost=float(costs[lab].mean()) if lab.any() else None
            fail_cost=float(costs[~lab].mean()) if (~lab).any() else None
            gap=None if succ_cost is None or fail_cost is None else fail_cost-succ_cost
        else:
            succ_cost=fail_cost=gap=None
        C.append({'policy':tag,'seed':seed,'top1_success':d.get('top1_success_rate'),'oracle_top30':d.get('oracle_topk_success_rate'),'mean_cost_success':succ_cost,'mean_cost_failure':fail_cost,'cost_gap':gap})
raw={'experiment_A':{'seeds':seedsA,'baseline':base_vals,'method':meth_vals,'delta':delta_vals,'baseline_stats':base_st,'method_stats':meth_st,'delta_stats':delta_st},'experiment_B':B,'experiment_C':C}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'raw_results.json').write_text(json.dumps(raw,indent=2))
lines=[]
lines+=['## Experiment A: 6-seed main comparison','|policy|seed42|seed43|seed44|seed45|seed46|seed47|mean|std|min|max|n|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
lines.append('|original gate07|'+'|'.join(fmt(v) for v in base_vals)+f"|{fmt(base_st['mean'])}|{fmt(base_st['std'])}|{fmt(base_st['min'])}|{fmt(base_st['max'])}|{base_st['n']}|")
lines.append('|state-roll l003 ep1|'+'|'.join(fmt(v) for v in meth_vals)+f"|{fmt(meth_st['mean'])}|{fmt(meth_st['std'])}|{fmt(meth_st['min'])}|{fmt(meth_st['max'])}|{meth_st['n']}|")
lines.append('|delta method-baseline|'+'|'.join(fmt(v) for v in delta_vals)+f"|{fmt(delta_st['mean'])}|{fmt(delta_st['std'])}|{fmt(delta_st['min'])}|{fmt(delta_st['max'])}|{delta_st['n']}|")
lines+=['','## Experiment B: early stopping curve','|checkpoint|seed42|seed43|seed44|mean|std|min|max|n|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in B: lines.append(f"|{r['checkpoint']}|{fmt(r['seed42'])}|{fmt(r['seed43'])}|{fmt(r['seed44'])}|{fmt(r['mean'])}|{fmt(r['std'])}|{fmt(r['min'])}|{fmt(r['max'])}|{r['n']}|")
lines+=['','## Experiment C: cost-success alignment','|policy|seed|top1_success|oracle_top30|mean_cost_success|mean_cost_failure|cost_gap|','|---|---:|---:|---:|---:|---:|---:|']
for r in C: lines.append(f"|{r['policy']}|{r['seed']}|{fmt(r['top1_success'])}|{fmt(r['oracle_top30'])}|{fmt(r['mean_cost_success'])}|{fmt(r['mean_cost_failure'])}|{fmt(r['cost_gap'])}|")
(OUT/'summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
