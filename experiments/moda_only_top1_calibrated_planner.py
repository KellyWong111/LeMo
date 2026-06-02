from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np

ROOT = Path('/data1/jingyixi/wm_runs')
ST_ACTION = ROOT / 'stateroll_normalbudget_candidate_pool_s300_steps30_n100' / 'proposal_data'
ST_RAW = ROOT / 'stateroll_normalbudget_candidate_pool_s300_steps30_n100' / 'raw_rollout_npz'
OUT = ROOT / 'moda_only_top1_calibrated_planner_20260529'
SEEDS = [42,43,44,45,46,47]
SPLITS = {
    'splitA_train42_44_val45_47': ([42,43,44], [45,46,47]),
    'splitB_train45_47_val42_44': ([45,46,47], [42,43,44]),
}


def load_seed(seed:int):
    a=np.load(ST_ACTION / f'vf05_mix20_seed{seed}.npz', allow_pickle=True)
    r=np.load(ST_RAW / f'vf05_mix20_seed{seed}.npz', allow_pickle=True)
    return {
        'actions': a['actions'].astype(np.float64),
        'costs': a['costs'].astype(np.float64),
        'labels': a['labels'].astype(bool),
        'indices': a['indices'],
        'pred': r['pred'].astype(np.float64),
        'goal': r['goal'].astype(np.float64),
    }

def goal_for_pred(goal,pred):
    g=goal
    if g.ndim==2: g=g[:,None,:]
    if g.ndim==3 and pred.ndim==4: g=g[:,None,:,:]
    if g.shape[1]==1: g=np.repeat(g,pred.shape[1],axis=1)
    if g.shape[2]==1: g=np.repeat(g,pred.shape[2],axis=2)
    elif g.shape[2]!=pred.shape[2]: g=g[:,:,-pred.shape[2]:,:]
    return g

def entropy_from_cost(costs):
    x=-costs.astype(np.float64)
    x=x-x.max(axis=-1,keepdims=True)
    p=np.exp(x); p=p/(p.sum(axis=-1,keepdims=True)+1e-12)
    return -(p*np.log(p+1e-12)).sum(axis=-1)

def build_dataset():
    X=[]; y=[]; raw=[]; meta=[]
    for seed in SEEDS:
        d=load_seed(seed)
        costs=d['costs']; labels=d['labels']; pred=d['pred']; goal=d['goal']; actions=d['actions']
        g=goal_for_pred(goal,pred)
        dist=np.sqrt(((pred-g)**2).sum(axis=-1))
        final=dist[:,:,-1]
        mean=dist.mean(axis=2)
        mind=dist.min(axis=2)
        progress=dist[:,:,0]-dist[:,:,-1]
        latent_mean=pred.mean(axis=(2,3))
        latent_std=pred.std(axis=(2,3))
        anorm=np.sqrt((actions**2).sum(axis=-1))
        action_norm=anorm.mean(axis=2)
        action_std=anorm.std(axis=2)
        E,K=costs.shape
        for ep in range(E):
            c=costs[ep]
            order=np.argsort(c, kind='stable')
            ranks=np.empty(K,dtype=int); ranks[order]=np.arange(K)
            sorted_c=c[order]
            cmean=c.mean(); cstd=c.std()+1e-6
            cz=(c-cmean)/cstd
            ent=float(entropy_from_cost(c[None])[0])
            margins=[float(sorted_c[min(k,K-1)]-sorted_c[0]) for k in [1,2,4,9]]
            ep_feats=[float(c.std()), ent, *margins, float(labels[ep].sum()), float(labels[ep].any())]
            for j in range(K):
                feat=[
                    1.0,
                    float(c[j]),
                    float(-c[j]),
                    float(ranks[j])/(K-1),
                    float(cz[j]),
                    float(c[j]-sorted_c[0]),
                    float(c[j]-sorted_c[min(1,K-1)]),
                    float(final[ep,j]),
                    float(mean[ep,j]),
                    float(mind[ep,j]),
                    float(progress[ep,j]),
                    float(action_norm[ep,j]),
                    float(action_std[ep,j]),
                    float(latent_mean[ep,j]),
                    float(latent_std[ep,j]),
                    *ep_feats,
                ]
                X.append(feat); y.append(bool(labels[ep,j])); raw.append(float(-c[j]))
                meta.append({'seed':seed,'episode':ep,'local_rank':int(ranks[j]),'raw_rank0':bool(ranks[j]==0)})
    return np.asarray(X),np.asarray(y,dtype=bool),np.asarray(raw),meta

def groups(meta,seeds):
    g={}
    for i,m in enumerate(meta):
        if m['seed'] in seeds:
            g.setdefault((m['seed'],m['episode']),[]).append(i)
    return [np.asarray(v,dtype=np.int64) for k,v in sorted(g.items())]

def standardize(X, meta, train_seeds):
    mask=np.asarray([m['seed'] in train_seeds for m in meta],dtype=bool)
    mean=X[mask].mean(axis=0); std=X[mask].std(axis=0)+1e-6
    Xs=(X-mean)/std
    Xb=np.concatenate([Xs,np.ones((Xs.shape[0],1))],axis=1)
    return Xb,mean,std

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-40,40)))

def fit_bce(X,y,meta,train_seeds,epochs=1800,lr=0.03,l2=1e-3):
    Xb,mean,std=standardize(X,meta,train_seeds)
    idx=np.where([m['seed'] in train_seeds for m in meta])[0]
    yy=y[idx].astype(float)
    pos=yy.sum(); neg=len(yy)-pos
    wt=np.ones(len(idx)); wt[yy.astype(bool)]=max(1.0,neg/max(pos,1.0)); wt/=wt.mean()
    w=np.zeros(Xb.shape[1])
    for _ in range(epochs):
        p=sigmoid(Xb[idx]@w)
        grad=Xb[idx].T@((p-yy)*wt)/len(idx)
        grad[:-1]+=l2*w[:-1]
        w-=lr*grad
    return {'w':w,'mean':mean,'std':std}

def fit_top1(X,y,raw,meta,train_seeds,mode='combined',epochs=2200,lr=0.025,l2=1e-3,alpha=1.0,beta=0.5,margin=0.5):
    Xb,mean,std=standardize(X,meta,train_seeds)
    gs=groups(meta,train_seeds)
    list_groups=[]; pairs=[]; near=[]
    for idxs in gs:
        lab=y[idxs]
        if not lab.any(): continue
        pos=idxs[lab]
        neg=idxs[~lab]
        list_groups.append(idxs)
        order=idxs[np.argsort(-raw[idxs], kind='stable')]
        rank0=order[0]
        low_fail=order[~y[order]][:min(8, int((~lab).sum()))]
        if (not y[rank0]):
            near.append(rank0)
        if len(neg):
            hard=np.unique(np.concatenate([np.asarray([rank0]) if not y[rank0] else np.asarray([],dtype=int), low_fail])).astype(int)
            for pi in pos:
                for ni in hard:
                    if not y[ni]: pairs.append((int(pi),int(ni)))
    pairs=np.asarray(pairs,dtype=np.int64) if pairs else np.zeros((0,2),dtype=np.int64)
    near=np.asarray(near,dtype=np.int64) if near else np.zeros(0,dtype=np.int64)
    w=np.zeros(Xb.shape[1])
    use_list=mode in ('listwise','combined','listwise_pairwise','listwise_nearmiss')
    use_pair=mode in ('pairwise','combined','listwise_pairwise','pairwise_nearmiss')
    use_near=mode in ('nearmiss','combined','listwise_nearmiss','pairwise_nearmiss')
    for _ in range(epochs):
        grad=np.zeros_like(w)
        parts=0
        if use_list and list_groups:
            gtot=np.zeros_like(w)
            for idxs in list_groups:
                z=np.clip(Xb[idxs]@w,-40,40); z-=z.max()
                p=np.exp(z); p/=p.sum()+1e-12
                t=y[idxs].astype(float); t/=t.sum()+1e-12
                gtot += Xb[idxs].T@(p-t)
            grad += gtot/len(list_groups); parts+=1
        if use_pair and len(pairs):
            diff=Xb[pairs[:,0]]-Xb[pairs[:,1]]
            z=np.clip(diff@w-margin,-40,40)
            grad += alpha*(diff.T@(-1/(1+np.exp(z))))/len(pairs); parts+=1
        if use_near and len(near):
            # penalize high score for raw rank0 failures in fixable episodes: softplus(score)
            z=np.clip(Xb[near]@w,-40,40)
            p=sigmoid(z)
            grad += beta*(Xb[near].T@p)/len(near); parts+=1
        if parts==0: break
        grad[:-1]+=l2*w[:-1]
        w-=lr*grad
    return {'w':w,'mean':mean,'std':std,'mode':mode,'pairs':int(len(pairs)),'near':int(len(near)),'groups':int(len(list_groups))}

def score_model(model,X):
    Xs=(X-model['mean'])/model['std']
    Xb=np.concatenate([Xs,np.ones((Xs.shape[0],1))],axis=1)
    return Xb@model['w']

def auc(y,score):
    y=y.astype(bool); npos=int(y.sum()); nneg=int((~y).sum())
    if npos==0 or nneg==0: return float('nan')
    order=np.argsort(score,kind='mergesort'); ss=score[order]
    ranks=np.empty(len(score),dtype=float); i=0
    while i<len(score):
        j=i+1
        while j<len(score) and ss[j]==ss[i]: j+=1
        ranks[order[i:j]]=(i+1+j)/2.0; i=j
    return float((ranks[y].sum()-npos*(npos+1)/2)/(npos*nneg))

def eval_scores(y,score,raw,meta,seeds,method,split):
    gs=groups(meta,seeds)
    top={1:0,3:0,5:0,10:0,30:0}; first=[]; no=0; near=0; hist={}; cases=[]
    mask=np.asarray([m['seed'] in seeds for m in meta],dtype=bool)
    for idxs in gs:
        lab=y[idxs]
        order=idxs[np.argsort(-score[idxs],kind='stable')]
        raw_order=idxs[np.argsort(-raw[idxs],kind='stable')]
        key=(meta[idxs[0]]['seed'], meta[idxs[0]]['episode'])
        if not lab.any():
            no+=1; continue
        r=int(np.where(y[order])[0][0]+1); rr=int(np.where(y[raw_order])[0][0]+1)
        first.append(r); hist[str(r)]=hist.get(str(r),0)+1
        if not y[order[0]]: near+=1
        for k in top: top[k]+=int(y[order[:k]].any())
        if (not y[raw_order[0]]) and y[order[0]]:
            cases.append({'split':split,'method':method,'seed':key[0],'episode':key[1],'raw_first_success_rank':rr,'cal_first_success_rank':r,'selected_rank':int(meta[order[0]]['local_rank']),'selected_success':bool(y[order[0]]),'success_count':int(lab.sum()),'raw_rank0_success':bool(y[raw_order[0]])})
    n=len(gs); arr=np.asarray(first,dtype=float)
    return {
        'split':split,'method':method,'episodes':n,'episodes_with_success':n-no,'episodes_without_success':no,
        'top1_success':top[1]/n*100,'top3_success':top[3]/n*100,'top5_success':top[5]/n*100,'top10_success':top[10]/n*100,'top30_success':top[30]/n*100,
        'candidate_auc':auc(y[mask],score[mask]),'first_success_rank_mean':float(arr.mean()) if len(arr) else None,'first_success_rank_median':float(np.median(arr)) if len(arr) else None,'near_miss_count':near,'hist':hist,'cases':cases[:50]
    }

def write_csv(path,rows,fields):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader();
        for r in rows: w.writerow({k:r.get(k,'') for k in fields})

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    X,y,raw,meta=build_dataset()
    methods=['raw','bce','listwise','pairwise','nearmiss','listwise_pairwise','listwise_nearmiss','pairwise_nearmiss','combined']
    rows=[]; hist_rows=[]; case_rows=[]; model_info=[]
    for split,(train,val) in SPLITS.items():
        models={'raw':None, 'bce':fit_bce(X,y,meta,train)}
        for m in ['listwise','pairwise','nearmiss','listwise_pairwise','listwise_nearmiss','pairwise_nearmiss','combined']:
            models[m]=fit_top1(X,y,raw,meta,train,mode=m)
            model_info.append({'split':split,'method':m,**{k:models[m][k] for k in ['groups','pairs','near']}})
        for m in methods:
            sc=raw if m=='raw' else score_model(models[m],X)
            ev=eval_scores(y,sc,raw,meta,val,m,split)
            hist=ev.pop('hist'); cases=ev.pop('cases')
            rows.append(ev)
            for rk,c in hist.items(): hist_rows.append({'split':split,'method':m,'first_success_rank':int(rk),'count':c})
            case_rows.extend(cases)
    fields=['split','method','episodes','episodes_with_success','episodes_without_success','top1_success','top3_success','top5_success','top10_success','top30_success','candidate_auc','first_success_rank_mean','first_success_rank_median','near_miss_count']
    write_csv(OUT/'moda_only_top1_calibrated_planner.csv',rows,fields)
    write_csv(OUT/'moda_only_top1_loss_ablation.csv',rows,fields)
    write_csv(OUT/'moda_only_success_rank_histogram.csv',hist_rows,['split','method','first_success_rank','count'])
    write_csv(OUT/'moda_only_case_studies.csv',case_rows,['split','method','seed','episode','raw_first_success_rank','cal_first_success_rank','selected_rank','selected_success','success_count','raw_rank0_success'])
    # aggregate
    agg=[]
    for m in methods:
        rs=[r for r in rows if r['method']==m]
        ep=sum(r['episodes'] for r in rs)
        agg.append({'method':m,'episodes':ep,**{k:sum(r[k]*r['episodes']/100 for r in rs)/ep*100 for k in ['top1_success','top3_success','top5_success','top10_success','top30_success']},'candidate_auc':float(np.mean([r['candidate_auc'] for r in rs])),'first_success_rank_mean':float(np.mean([r['first_success_rank_mean'] for r in rs])),'first_success_rank_median':float(np.median([r['first_success_rank_median'] for r in rs])),'near_miss_count':sum(r['near_miss_count'] for r in rs)})
    best=max(agg,key=lambda r:r['top1_success'])
    verdict='viable_standalone' if best['top1_success']>=65 else ('below_60_feature_calibration_insufficient' if best['top1_success']<60 else 'partial_but_below_target')
    (OUT/'moda_only_top1_calibrated_planner.json').write_text(json.dumps({'oof_summary':agg,'per_split':rows,'model_info':model_info,'verdict':verdict,'target_top1':65.0,'baseline_raw_top1':next(r for r in agg if r['method']=='raw')['top1_success']},indent=2)+'\n')
    md=['# MoDA-Only Top1-Aware Calibrated Planner','', 'Only stateroll/MoDA candidate pool is used. No bsl success/failure, no bsl fallback, no fixed/harmed main metric.','', '## OOF Summary','', '|method|top1|top3|top5|top10|top30|AUC|first rank mean|median|near-miss|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in agg:
        md.append(f"|{r['method']}|{r['top1_success']:.2f}|{r['top3_success']:.2f}|{r['top5_success']:.2f}|{r['top10_success']:.2f}|{r['top30_success']:.2f}|{r['candidate_auc']:.3f}|{r['first_success_rank_mean']:.2f}|{r['first_success_rank_median']:.2f}|{r['near_miss_count']}|")
    md += ['', '## Per Split','', '|split|method|top1|top3|top5|top10|AUC|first rank mean|near-miss|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        md.append(f"|{r['split']}|{r['method']}|{r['top1_success']:.1f}|{r['top3_success']:.1f}|{r['top5_success']:.1f}|{r['top10_success']:.1f}|{r['candidate_auc']:.3f}|{r['first_success_rank_mean']:.2f}|{r['near_miss_count']}|")
    md += ['', '## Verdict','']
    md.append(f"Raw MoDA top1 is {next(r for r in agg if r['method']=='raw')['top1_success']:.2f}. Best calibrated MoDA-only top1 is {best['top1_success']:.2f} from `{best['method']}`. Target was 65.0. Verdict: `{verdict}`.")
    if best['top1_success']<60:
        md.append('')
        md.append('Conclusion: current feature-level top1-aware calibration is insufficient. The bsl-integrated PAC-MoDA results cannot be the main paper story as a standalone MoDA planner improvement.')
    elif best['top1_success']<65:
        md.append('')
        md.append('Conclusion: calibration improves MoDA-only top1 somewhat, but remains below the standalone-method target.')
    (OUT/'moda_only_top1_calibrated_planner.md').write_text('\n'.join(md)+'\n')
    print((OUT/'moda_only_top1_calibrated_planner.md').read_text())

if __name__=='__main__': main()
