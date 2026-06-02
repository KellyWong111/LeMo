import json, re
from pathlib import Path
from statistics import mean

stable = Path('/data1/jingyixi/.stable_worldmodel')
outdirs = [Path('/data1/jingyixi/wm_runs/state_rollout_overnight_confirm'), Path('/data1/jingyixi/wm_runs/state_rollout_l003_l007_resume')]
outdir = Path('/data1/jingyixi/wm_runs/state_rollout_l003_l007_resume')
base = 'pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_'
tags = ['l003', 'l005', 'l007', 'l010']
epochs = [1, 2]
cems = {'standard': (300, 30, 30), 'medium': (600, 60, 20), 'strong': (1000, 100, 20)}
seeds = [42,43,44]

def parse(path):
    if not path.exists(): return None
    text = path.read_text(errors='ignore')
    m = re.search(r"success_rate['\"]?\s*[:=]\s*([0-9.]+)", text) or re.search(r"'success_rate':\s*([0-9.]+)", text)
    return float(m.group(1)) if m else None
raw=[]
for tag in tags:
  subdir = stable / f'{base}{tag}'
  for ep in epochs:
    for cem,(samples,topk,steps) in cems.items():
      for seed in seeds:
        names=[f'staterollseq_{tag}_ep{ep}_{cem}_seed{seed}_h4_s{samples}_k{topk}_n{steps}.txt', f'staterollseq_{tag}_ep{ep}_seed{seed}_h4_s{samples}_k{topk}_n{steps}.txt']
        val=None; file=None
        for name in names:
          p=subdir/name
          val=parse(p)
          if val is not None:
            file=str(p); break
        raw.append({'tag':tag,'epoch':ep,'cem':cem,'seed':seed,'success_rate':val,'file':file})
outdir.mkdir(parents=True, exist_ok=True)
(outdir/'raw_results.json').write_text(json.dumps(raw, indent=2))
rows=[]
for tag in tags:
  for ep in epochs:
    for cem in cems:
      vals_by_seed={r['seed']:r['success_rate'] for r in raw if r['tag']==tag and r['epoch']==ep and r['cem']==cem}
      vals=[v for v in vals_by_seed.values() if v is not None]
      if not vals: continue
      rows.append({'setting':f'{tag} ep{ep} {cem}','tag':tag,'epoch':ep,'cem':cem,'seed42':vals_by_seed.get(42),'seed43':vals_by_seed.get(43),'seed44':vals_by_seed.get(44),'mean':mean(vals),'min':min(vals),'max':max(vals),'n':len(vals)})
rows.sort(key=lambda r:(r['tag'], r['epoch'], {'standard':0,'medium':1,'strong':2}[r['cem']]))
(outdir/'aggregate.json').write_text(json.dumps(rows, indent=2))
complete=[r for r in rows if r['n']==3]
def fmt(x): return 'NA' if x is None else f'{x:.1f}'
lines=['|setting|seed42|seed43|seed44|mean|min|max|n|','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"|{r['setting']}|{fmt(r['seed42'])}|{fmt(r['seed43'])}|{fmt(r['seed44'])}|{fmt(r['mean'])}|{fmt(r['min'])}|{fmt(r['max'])}|{r['n']}|")
lines += ['', '|criterion|setting|mean|min|max|n|','|---|---|---:|---:|---:|---:|']
if complete:
    bm=max(complete,key=lambda r:(r['mean'],r['min']))
    bn=max(complete,key=lambda r:(r['min'],r['mean']))
    lines.append(f"|highest mean|{bm['setting']}|{fmt(bm['mean'])}|{fmt(bm['min'])}|{fmt(bm['max'])}|{bm['n']}|")
    lines.append(f"|highest min|{bn['setting']}|{fmt(bn['mean'])}|{fmt(bn['min'])}|{fmt(bn['max'])}|{bn['n']}|")
    for cem in ['standard','medium','strong']:
        c=[r for r in complete if r['cem']==cem]
        if c:
            b=max(c,key=lambda r:(r['mean'],r['min']))
            lines.append(f"|best {cem}|{b['setting']}|{fmt(b['mean'])}|{fmt(b['min'])}|{fmt(b['max'])}|{b['n']}|")
(outdir/'summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
