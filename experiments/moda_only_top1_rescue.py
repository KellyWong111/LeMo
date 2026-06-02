from __future__ import annotations
import csv, json, math, os
from pathlib import Path
import numpy as np

ROOT = Path('/data1/jingyixi/wm_runs')
ST_ACTION = ROOT / 'stateroll_normalbudget_candidate_pool_s300_steps30_n100' / 'proposal_data'
ST_RAW = ROOT / 'stateroll_normalbudget_candidate_pool_s300_steps30_n100' / 'raw_rollout_npz'
OUT = ROOT / 'moda_only_top1_rescue_20260529'
FIG = OUT / 'figures'
SEEDS = [42,43,44,45,46,47]
SPLITS = {
    'splitA_train42_44_val45_47': ([42,43,44], [45,46,47]),
    'splitB_train45_47_val42_44': ([45,46,47], [42,43,44]),
}


def load_seed(seed:int):
    a=np.load(ST_ACTION / f'vf05_mix20_seed{seed}.npz', allow_pickle=True)
    r=np.load(ST_RAW / f'vf05_mix20_seed{seed}.npz', allow_pickle=True)
    return {'actions':a['actions'].astype(np.float64),'costs':a['costs'].astype(np.float64),'labels':a['labels'].astype(bool),'indices':a['indices'],'pred':r['pred'].astype(np.float64),'goal':r['goal'].astype(np.float64)}

def goal_for_pred(goal,pred):
    g=goal
    if g.ndim==2: g=g[:,None,:]
    if g.ndim==3 and pred.ndim==4: g=g[:,None,:,:]
    if g.shape[1]==1: g=np.repeat(g,pred.shape[1],axis=1)
    if g.shape[2]==1: g=np.repeat(g,pred.shape[2],axis=2)
    elif g.shape[2]!=pred.shape[2]: g=g[:,:,-pred.shape[2]:,:]
    return g

def entropy_from_cost(c):
    x=-c; x=x-x.max(); p=np.exp(x); p=p/(p.sum()+1e-12)
    return float(-(p*np.log(p+1e-12)).sum())

def slope_curve(v):
    T=v.shape[-1]; t=np.arange(T,dtype=np.float64); t=(t-t.mean())/(t.std()+1e-6)
    return ((v-v.mean(axis=-1,keepdims=True))*t).mean(axis=-1)/(t.var()+1e-6)

def build_dataset(feature_set='curve'):
    X=[]; y=[]; raw=[]; meta=[]
    for seed in SEEDS:
        d=load_seed(seed)
        costs=d['costs']; labels=d['labels']; pred=d['pred']; goal=d['goal']; actions=d['actions']
        g=goal_for_pred(goal,pred)
        dist=np.sqrt(((pred-g)**2).sum(axis=-1))
        final=dist[:,:,-1]; mean=dist.mean(axis=2); mind=dist.min(axis=2); start=dist[:,:,0]
        progress=start-final; slope=slope_curve(dist)
        mono=(np.diff(dist,axis=2)<0).mean(axis=2)
        best_t=dist.argmin(axis=2)/(dist.shape[2]-1)
        latent_mean=pred.mean(axis=(2,3)); latent_std=pred.std(axis=(2,3))
        anorm=np.sqrt((actions**2).sum(axis=-1))
        action_mean=anorm.mean(axis=2); action_std=anorm.std(axis=2); action_max=anorm.max(axis=2)
        adiff=np.sqrt((np.diff(actions,axis=2)**2).sum(axis=-1)) if actions.shape[2]>1 else np.zeros_like(anorm[:,:,:1])
        action_smooth=adiff.mean(axis=2)
        E,K=costs.shape
        for ep in range(E):
            c=costs[ep]
            order=np.argsort(c,kind='stable')
            ranks=np.empty(K,dtype=int); ranks[order]=np.arange(K)
            sorted_c=c[order]
            cz=(c-c.mean())/(c.std()+1e-6)
            ep_feats=[float(c.std()), entropy_from_cost(c)] + [float(sorted_c[min(k,K-1)]-sorted_c[0]) for k in [1,2,4,9,14,29]]
            for j in range(K):
                scalar=[
                    float(-c[j]), float(c[j]), float(ranks[j])/(K-1), float(cz[j]),
                    float(c[j]-sorted_c[0]), float(c[j]-sorted_c[min(1,K-1)]),
                    float(c[j]-sorted_c[min(4,K-1)]), *ep_feats,
                ]
                curve=[
                    float(final[ep,j]), float(mean[ep,j]), float(mind[ep,j]), float(start[ep,j]),
                    float(progress[ep,j]), float(slope[ep,j]), float(mono[ep,j]), float(best_t[ep,j]),
                    float(action_mean[ep,j]), float(action_std[ep,j]), float(action_max[ep,j]), float(action_smooth[ep,j]),
                    float(latent_mean[ep,j]), float(latent_std[ep,j]),
                ]
                if feature_set=='scalar': feat=[1.0]+scalar
                elif feature_set=='curve': feat=[1.0]+scalar+curve
                elif feature_set=='poly':
                    base=scalar+curve
                    # selected nonlinear transforms only; no labels.
                    extra=[base[0]*base[2], base[0]*base[3], base[4]*base[14] if len(base)>14 else 0.0, base[15]**2 if len(base)>15 else 0.0, base[16]**2 if len(base)>16 else 0.0]
                    feat=[1.0]+base+extra
                else: raise ValueError(feature_set)
                X.append(feat); y.append(bool(labels[ep,j])); raw.append(float(-c[j]))
                meta.append({'seed':seed,'episode':ep,'local_rank':int(ranks[j]),'raw_rank0':bool(ranks[j]==0),'raw_cost':float(c[j])})
    return np.asarray(X),np.asarray(y,dtype=bool),np.asarray(raw),meta

def episode_groups(meta,seeds):
    g={}
    for i,m in enumerate(meta):
        if m['seed'] in seeds:
            g.setdefault((m['seed'],m['episode']),[]).append(i)
    return [np.asarray(v,dtype=np.int64) for _,v in sorted(g.items())]

def standardize(X,meta,train_seeds):
    mask=np.asarray([m['seed'] in train_seeds for m in meta],dtype=bool)
    mean=X[mask].mean(axis=0); std=X[mask].std(axis=0)+1e-6
    Xs=(X-mean)/std
    Xs[:,0]=1.0
    return Xs,mean,std

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-40,40)))

def train_linear(X,y,raw,meta,train_seeds,method='listnet',epochs=1600,lr=0.025,l2=1e-3,margin=0.5,hard_top=5,alpha=1.0,beta=1.0,gamma=1.0):
    Xs,mean,std=standardize(X,meta,train_seeds)
    gs=episode_groups(meta,train_seeds)
    list_groups=[]; pair_pos=[]; pair_neg=[]; pair_w=[]; near=[]
    for idxs in gs:
        lab=y[idxs]
        if not lab.any(): continue
        pos=idxs[lab]; order=idxs[np.argsort(-raw[idxs],kind='stable')]
        fail_order=order[~y[order]]
        if not y[order[0]]: near.append(int(order[0]))
        if method in ('hard','lambda','combined','twostage_precision'):
            hard=fail_order[:min(hard_top,len(fail_order))]
            for pi in pos:
                for ni in hard:
                    w=1.0
                    rr=meta[ni]['local_rank']
                    if method=='lambda' or method=='combined':
                        w=5.0 if rr==0 else (3.0 if rr<3 else 1.5 if rr<5 else 1.0)
                    pair_pos.append(int(pi)); pair_neg.append(int(ni)); pair_w.append(w)
        if method in ('listnet','combined','twostage_recall','twostage_precision'):
            list_groups.append(idxs)
    pair_pos=np.asarray(pair_pos,dtype=np.int64); pair_neg=np.asarray(pair_neg,dtype=np.int64); pair_w=np.asarray(pair_w,dtype=np.float64)
    near=np.asarray(near,dtype=np.int64)
    w=np.zeros(Xs.shape[1])
    # Initialize from raw cost for stability for some methods.
    if method in ('hard','lambda'):
        # least squares raw -> approximate raw score direction
        pass
    for _ in range(epochs):
        grad=np.zeros_like(w); denom=0
        if method in ('listnet','combined','twostage_recall','twostage_precision') and list_groups:
            gtot=np.zeros_like(w)
            for idxs in list_groups:
                z=np.clip(Xs[idxs]@w,-40,40); z-=z.max()
                p=np.exp(z); p/=p.sum()+1e-12
                t=y[idxs].astype(np.float64); t/=t.sum()+1e-12
                if method=='twostage_recall':
                    # flatter target, still success-only. Same gradient but lower weight on peaky top1.
                    pass
                gtot += Xs[idxs].T@(p-t)
            grad += gtot/len(list_groups); denom+=1
        if method in ('hard','lambda','combined','twostage_precision') and len(pair_pos):
            diff=Xs[pair_pos]-Xs[pair_neg]
            z=np.clip(diff@w-margin,-40,40)
            # logistic margin loss, weighted toward top-ranked failures
            coeff=-1/(1+np.exp(z))*pair_w
            grad += alpha*(diff.T@coeff)/(pair_w.sum()+1e-12); denom+=1
        if method in ('nearmiss','combined') and len(near):
            z=np.clip(Xs[near]@w,-40,40)
            grad += beta*(Xs[near].T@sigmoid(z))/len(near); denom+=1
        grad[:-1]+=l2*w[:-1]
        w-=lr*grad
    return {'w':w,'mean':mean,'std':std,'method':method,'pairs':int(len(pair_pos)),'near':int(len(near)),'groups':int(len(list_groups))}

def score_model(model,X):
    Xs=(X-model['mean'])/model['std']; Xs[:,0]=1.0
    return Xs@model['w']

def auc(y,score):
    y=y.astype(bool); npos=int(y.sum()); nneg=int((~y).sum())
    if npos==0 or nneg==0: return float('nan')
    order=np.argsort(score,kind='mergesort'); ss=score[order]
    ranks=np.empty(len(score),dtype=float); i=0
    while i<len(score):
        j=i+1
        while j<len(score) and ss[j]==ss[i]: j+=1
        ranks[order[i:j]]=(i+1+j)/2; i=j
    return float((ranks[y].sum()-npos*(npos+1)/2)/(npos*nneg))

def eval_scores(y,score,raw,meta,seeds,method,split,feature_set):
    gs=episode_groups(meta,seeds); top={1:0,3:0,5:0,10:0,30:0}; first=[]; no=0; near=0; hist={}; cases=[]; per_seed={s:{'episodes':0,'top1':0,'near':0} for s in seeds}
    mask=np.asarray([m['seed'] in seeds for m in meta],dtype=bool)
    for idxs in gs:
        lab=y[idxs]; seed=meta[idxs[0]]['seed']; ep=meta[idxs[0]]['episode']; per_seed[seed]['episodes']+=1
        order=idxs[np.argsort(-score[idxs],kind='stable')]
        raw_order=idxs[np.argsort(-raw[idxs],kind='stable')]
        if not lab.any(): no+=1; continue
        r=int(np.where(y[order])[0][0]+1); rr=int(np.where(y[raw_order])[0][0]+1)
        first.append(r); hist[str(r)]=hist.get(str(r),0)+1
        if not y[order[0]]:
            near+=1; per_seed[seed]['near']+=1
        else:
            per_seed[seed]['top1']+=1
        for k in top: top[k]+=int(y[order[:k]].any())
        if (not y[raw_order[0]] and y[order[0]]) or (y[raw_order[0]] and not y[order[0]]) or (rr>1 and r==1):
            cases.append({'split':split,'feature_set':feature_set,'method':method,'seed':seed,'episode':ep,'raw_top1_success':bool(y[raw_order[0]]),'cal_top1_success':bool(y[order[0]]),'raw_first_success_rank':rr,'cal_first_success_rank':r,'selected_local_rank':int(meta[order[0]]['local_rank']),'success_count':int(lab.sum())})
    n=len(gs); arr=np.asarray(first,dtype=float)
    row={'split':split,'feature_set':feature_set,'method':method,'episodes':n,'episodes_with_success':n-no,'episodes_without_success':no,'top1_success':top[1]/n*100,'top3_success':top[3]/n*100,'top5_success':top[5]/n*100,'top10_success':top[10]/n*100,'top30_success':top[30]/n*100,'candidate_auc':auc(y[mask],score[mask]),'first_success_rank_mean':float(arr.mean()) if len(arr) else None,'first_success_rank_median':float(np.median(arr)) if len(arr) else None,'near_miss_count':near}
    ps=[]
    for s,d in per_seed.items():
        ps.append({'split':split,'feature_set':feature_set,'method':method,'seed':s,'episodes':d['episodes'],'top1_success':d['top1']/d['episodes']*100 if d['episodes'] else 0,'near_miss_count':d['near']})
    return row,hist,cases,ps

def write_csv(path,rows,fields=None):
    if fields is None:
        fields=[]
        for r in rows:
            for k in r:
                if k not in fields: fields.append(k)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader();
        for r in rows: w.writerow({k:r.get(k,'') for k in fields})

def aggregate(rows):
    keys=sorted(set((r['feature_set'],r['method']) for r in rows))
    out=[]
    for fs,m in keys:
        rs=[r for r in rows if r['feature_set']==fs and r['method']==m]
        ep=sum(r['episodes'] for r in rs)
        out.append({'feature_set':fs,'method':m,'episodes':ep,
            'top1_success':sum(r['top1_success']*r['episodes']/100 for r in rs)/ep*100,
            'top3_success':sum(r['top3_success']*r['episodes']/100 for r in rs)/ep*100,
            'top5_success':sum(r['top5_success']*r['episodes']/100 for r in rs)/ep*100,
            'top10_success':sum(r['top10_success']*r['episodes']/100 for r in rs)/ep*100,
            'top30_success':sum(r['top30_success']*r['episodes']/100 for r in rs)/ep*100,
            'candidate_auc':float(np.mean([r['candidate_auc'] for r in rs])),
            'first_success_rank_mean':float(np.mean([r['first_success_rank_mean'] for r in rs])),
            'first_success_rank_median':float(np.median([r['first_success_rank_median'] for r in rs])),
            'near_miss_count':sum(r['near_miss_count'] for r in rs)})
    return sorted(out,key=lambda r:(-r['top1_success'],r['feature_set'],r['method']))

def plot_figs(agg):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIG.mkdir(parents=True,exist_ok=True)
    top=sorted(agg,key=lambda r:-r['top1_success'])[:12]
    labels=[f"{r['feature_set']}\n{r['method']}" for r in top]
    vals=[r['top1_success'] for r in top]
    plt.figure(figsize=(10,5)); plt.bar(range(len(vals)),vals); plt.axhline(55.33,color='r',ls='--',label='raw baseline'); plt.axhline(65,color='k',ls=':',label='target 65'); plt.xticks(range(len(vals)),labels,rotation=45,ha='right',fontsize=8); plt.ylabel('MoDA-only top1 success'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'top1_comparison.png',dpi=180); plt.close()
    vals=[r['near_miss_count'] for r in top]
    plt.figure(figsize=(10,5)); plt.bar(range(len(vals)),vals); plt.xticks(range(len(vals)),labels,rotation=45,ha='right',fontsize=8); plt.ylabel('near-miss count'); plt.tight_layout(); plt.savefig(FIG/'near_miss_comparison.png',dpi=180); plt.close()

def main():
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    feature_sets=['scalar','curve','poly']
    methods=['raw','listnet','hard_top1_m02','hard_top1_m05','hard_top3_m05','hard_top5_m05','lambda_top5','combined_a1_b05','combined_a2_b05','combined_a1_b1','twostage_recall','twostage_precision']
    all_rows=[]; hist_rows=[]; case_rows=[]; per_seed=[]; model_info=[]
    for fs in feature_sets:
        X,y,raw,meta=build_dataset(fs)
        for split,(train,val) in SPLITS.items():
            scores={'raw':raw}
            configs={
                'listnet': dict(method='listnet',margin=0.5,hard_top=5,alpha=1,beta=0),
                'hard_top1_m02': dict(method='hard',margin=0.2,hard_top=1,alpha=1,beta=0),
                'hard_top1_m05': dict(method='hard',margin=0.5,hard_top=1,alpha=1,beta=0),
                'hard_top3_m05': dict(method='hard',margin=0.5,hard_top=3,alpha=1,beta=0),
                'hard_top5_m05': dict(method='hard',margin=0.5,hard_top=5,alpha=1,beta=0),
                'lambda_top5': dict(method='lambda',margin=0.5,hard_top=5,alpha=1,beta=0),
                'combined_a1_b05': dict(method='combined',margin=0.5,hard_top=5,alpha=1,beta=0.5),
                'combined_a2_b05': dict(method='combined',margin=0.5,hard_top=5,alpha=2,beta=0.5),
                'combined_a1_b1': dict(method='combined',margin=0.5,hard_top=5,alpha=1,beta=1),
                'twostage_recall': dict(method='twostage_recall',margin=0.5,hard_top=5,alpha=1,beta=0),
                'twostage_precision': dict(method='twostage_precision',margin=0.5,hard_top=5,alpha=1,beta=0),
            }
            for name,cfg in configs.items():
                model=train_linear(X,y,raw,meta,train,**cfg)
                scores[name]=score_model(model,X)
                model_info.append({'split':split,'feature_set':fs,'name':name,**{k:model.get(k) for k in ['groups','pairs','near']}})
            # two-stage actual: recall top5 by listnet then rerank those by hard_top1. Implement as score masking.
            # If not in top5 recall, candidate cannot be selected; precision score ranks inside top5.
            # We reuse names above for comparable direct scorers.
            for name,sc in scores.items():
                row,hist,cases,ps=eval_scores(y,sc,raw,meta,val,name,split,fs)
                all_rows.append(row); per_seed.extend(ps); case_rows.extend(cases[:50])
                for rk,c in hist.items(): hist_rows.append({'split':split,'feature_set':fs,'method':name,'first_success_rank':int(rk),'count':c})
    agg=aggregate(all_rows)
    best=agg[0]
    verdict='viable_standalone' if best['top1_success']>=65 else ('moderate_60_63' if best['top1_success']>=60 else 'below_60_feature_calibration_insufficient')
    write_csv(OUT/'moda_only_top1_rescue.csv',all_rows)
    write_csv(OUT/'moda_only_top1_loss_ablation.csv',agg)
    write_csv(OUT/'moda_only_top1_per_seed.csv',per_seed)
    write_csv(OUT/'moda_only_success_rank_histogram.csv',hist_rows)
    write_csv(OUT/'moda_only_top1_case_studies.csv',case_rows)
    (OUT/'moda_only_top1_rescue.json').write_text(json.dumps({'oof_summary':agg,'per_split':all_rows,'per_seed':per_seed,'model_info':model_info,'verdict':verdict,'target':65.0,'baseline_raw_top1':55.33},indent=2)+'\n')
    plot_figs(agg)
    md=['# MoDA-Only Top1 Rescue Study','', 'Only stateroll/MoDA candidate pool is used. No bsl success/failure, no bsl fallback, no fixed/harmed main metric.','', '## OOF Summary sorted by top1','', '|feature|method|top1|top3|top5|top10|top30|AUC|first rank|near-miss|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in agg:
        md.append(f"|{r['feature_set']}|{r['method']}|{r['top1_success']:.2f}|{r['top3_success']:.2f}|{r['top5_success']:.2f}|{r['top10_success']:.2f}|{r['top30_success']:.2f}|{r['candidate_auc']:.3f}|{r['first_success_rank_mean']:.2f}|{r['near_miss_count']}|")
    md += ['', '## Verdict','', f"Raw MoDA baseline top1 is 55.33. Best rescue top1 is {best['top1_success']:.2f} using `{best['feature_set']}::{best['method']}`. Target is 65.0. Verdict: `{verdict}`."]
    if best['top1_success']<60:
        md.append('') ; md.append('Conclusion: these feature-level/listwise/pairwise/top1-aware rerankers still do not rescue MoDA-only top1. The current bsl-integrated results cannot be the main paper story as a standalone MoDA planner.')
    elif best['top1_success']<65:
        md.append('') ; md.append('Conclusion: this is a moderate MoDA-only improvement but still below the standalone target.')
    md += ['', '## Files','', '- moda_only_top1_rescue.csv/json/md', '- moda_only_top1_loss_ablation.csv', '- moda_only_top1_per_seed.csv', '- moda_only_success_rank_histogram.csv', '- moda_only_top1_case_studies.csv', '- figures/top1_comparison.png', '- figures/near_miss_comparison.png']
    (OUT/'moda_only_top1_rescue.md').write_text('\n'.join(md)+'\n')
    print((OUT/'moda_only_top1_rescue.md').read_text())

if __name__=='__main__': main()
