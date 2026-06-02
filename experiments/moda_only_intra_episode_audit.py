from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

ROOT=Path('/data1/jingyixi/wm_runs')
ST_ACTION=ROOT/'stateroll_normalbudget_candidate_pool_s300_steps30_n100'/'proposal_data'
ST_RAW=ROOT/'stateroll_normalbudget_candidate_pool_s300_steps30_n100'/'raw_rollout_npz'
OUT=ROOT/'moda_only_intra_episode_audit_20260529'
SEEDS=[42,43,44,45,46,47]
SPLITS={'splitA_train42_44_val45_47':([42,43,44],[45,46,47]),'splitB_train45_47_val42_44':([45,46,47],[42,43,44])}


def load_seed(seed):
    a=np.load(ST_ACTION/f'vf05_mix20_seed{seed}.npz',allow_pickle=True)
    r=np.load(ST_RAW/f'vf05_mix20_seed{seed}.npz',allow_pickle=True)
    return {'actions':a['actions'].astype(np.float64),'costs':a['costs'].astype(np.float64),'labels':a['labels'].astype(bool),'pred':r['pred'].astype(np.float64),'goal':r['goal'].astype(np.float64)}

def goal_for_pred(goal,pred):
    g=goal
    if g.ndim==2: g=g[:,None,:]
    if g.ndim==3 and pred.ndim==4: g=g[:,None,:,:]
    if g.shape[1]==1: g=np.repeat(g,pred.shape[1],axis=1)
    if g.shape[2]==1: g=np.repeat(g,pred.shape[2],axis=2)
    elif g.shape[2]!=pred.shape[2]: g=g[:,:,-pred.shape[2]:,:]
    return g

def entropy(c):
    x=-c; x=x-x.max(); p=np.exp(x); p=p/(p.sum()+1e-12)
    return float(-(p*np.log(p+1e-12)).sum())

def slope_curve(v):
    T=v.shape[-1]; t=np.arange(T,dtype=np.float64); t=(t-t.mean())/(t.std()+1e-6)
    return ((v-v.mean(axis=-1,keepdims=True))*t).mean(axis=-1)/(t.var()+1e-6)

def build():
    rows=[]; y=[]; raw=[]; meta=[]
    for seed in SEEDS:
        d=load_seed(seed); costs=d['costs']; labels=d['labels']; pred=d['pred']; goal=d['goal']; actions=d['actions']
        g=goal_for_pred(goal,pred)
        dist=np.sqrt(((pred-g)**2).sum(axis=-1))
        final=dist[:,:,-1]; mean=dist.mean(axis=2); mind=dist.min(axis=2); start=dist[:,:,0]
        progress=start-final; slope=slope_curve(dist); mono=(np.diff(dist,axis=2)<0).mean(axis=2); best_t=dist.argmin(axis=2)/(dist.shape[2]-1)
        latent_mean=pred.mean(axis=(2,3)); latent_std=pred.std(axis=(2,3))
        anorm=np.sqrt((actions**2).sum(axis=-1)); action_mean=anorm.mean(axis=2); action_std=anorm.std(axis=2); action_max=anorm.max(axis=2)
        adiff=np.sqrt((np.diff(actions,axis=2)**2).sum(axis=-1)) if actions.shape[2]>1 else np.zeros_like(anorm[:,:,:1])
        action_smooth=adiff.mean(axis=2)
        E,K=costs.shape
        for ep in range(E):
            c=costs[ep]; order=np.argsort(c,kind='stable'); ranks=np.empty(K,dtype=int); ranks[order]=np.arange(K)
            sorted_c=c[order]; cz=(c-c.mean())/(c.std()+1e-6)
            ep_cost=[float(c.std()), entropy(c)] + [float(sorted_c[min(k,K-1)]-sorted_c[0]) for k in [1,2,4,9,14,29]]
            ep_traj=[float(final[ep].mean()),float(final[ep].std()),float(mean[ep].mean()),float(progress[ep].mean()),float(progress[ep].std()),float(mind[ep].mean())]
            ep_action=[float(action_mean[ep].mean()),float(action_mean[ep].std()),float(action_smooth[ep].mean()),float(action_smooth[ep].std())]
            ep_latent=[float(latent_mean[ep].mean()),float(latent_mean[ep].std()),float(latent_std[ep].mean()),float(latent_std[ep].std())]
            ep_all=ep_cost+ep_traj+ep_action+ep_latent
            for j in range(K):
                cost=[float(-c[j]),float(c[j]),float(ranks[j])/(K-1),float(cz[j]),float(c[j]-sorted_c[0]),float(c[j]-sorted_c[min(1,K-1)]),float(c[j]-sorted_c[min(4,K-1)])]
                traj=[float(final[ep,j]),float(mean[ep,j]),float(mind[ep,j]),float(start[ep,j]),float(progress[ep,j]),float(slope[ep,j]),float(mono[ep,j]),float(best_t[ep,j])]
                action=[float(action_mean[ep,j]),float(action_std[ep,j]),float(action_max[ep,j]),float(action_smooth[ep,j])]
                latent=[float(latent_mean[ep,j]),float(latent_std[ep,j])]
                rows.append({'episode_only':ep_all,'cost':cost,'trajectory':traj,'action':action,'latent':latent,'candidate_only':cost+traj+action+latent,'cost_trajectory':cost+traj,'all':cost+traj+action+latent+ep_all})
                y.append(bool(labels[ep,j])); raw.append(float(-c[j])); meta.append({'seed':seed,'episode':ep,'local_rank':int(ranks[j]),'raw_rank0':bool(ranks[j]==0)})
    return rows,np.asarray(y,dtype=bool),np.asarray(raw),meta

def make_X(rows,kind):
    return np.asarray([[1.0]+r[kind] for r in rows],dtype=np.float64)

def groups(meta,seeds):
    g={}
    for i,m in enumerate(meta):
        if m['seed'] in seeds: g.setdefault((m['seed'],m['episode']),[]).append(i)
    return [np.asarray(v,dtype=np.int64) for _,v in sorted(g.items())]

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-40,40)))

def fit_bce(X,y,meta,train,epochs=1800,lr=0.03,l2=1e-3):
    mask=np.asarray([m['seed'] in train for m in meta],dtype=bool); mean=X[mask].mean(axis=0); std=X[mask].std(axis=0)+1e-6; std[0]=1; mean[0]=0
    Xs=(X-mean)/std; Xs[:,0]=1
    idx=np.where(mask)[0]; yy=y[idx].astype(float); pos=yy.sum(); neg=len(yy)-pos
    wt=np.ones(len(idx)); wt[yy.astype(bool)]=max(1,neg/max(pos,1)); wt/=wt.mean()
    w=np.zeros(X.shape[1])
    for _ in range(epochs):
        p=sigmoid(Xs[idx]@w); grad=Xs[idx].T@((p-yy)*wt)/len(idx); grad[1:]+=l2*w[1:]; w-=lr*grad
    return {'w':w,'mean':mean,'std':std}

def score(model,X):
    Xs=(X-model['mean'])/model['std']; Xs[:,0]=1; return Xs@model['w']

def auc(y,s):
    y=y.astype(bool); npos=int(y.sum()); nneg=int((~y).sum())
    if npos==0 or nneg==0: return float('nan')
    order=np.argsort(s,kind='mergesort'); ss=s[order]; ranks=np.empty(len(s),dtype=float); i=0
    while i<len(s):
        j=i+1
        while j<len(s) and ss[j]==ss[i]: j+=1
        ranks[order[i:j]]=(i+1+j)/2; i=j
    return float((ranks[y].sum()-npos*(npos+1)/2)/(npos*nneg))

def eval_score(y,s,raw,meta,seeds,split,kind):
    gs=groups(meta,seeds); top={1:0,3:0,5:0,10:0,30:0}; first=[]; no=0; near=0; intra=[]; pair_total=0; pair_win=0; case=[]
    mask=np.asarray([m['seed'] in seeds for m in meta],dtype=bool)
    for idxs in gs:
        lab=y[idxs]; order=idxs[np.argsort(-s[idxs],kind='stable')]; raw_order=idxs[np.argsort(-raw[idxs],kind='stable')]
        seed=meta[idxs[0]]['seed']; ep=meta[idxs[0]]['episode']
        if lab.any() and (~lab).any(): intra.append(auc(lab,s[idxs]))
        if not lab.any(): no+=1; continue
        r=int(np.where(y[order])[0][0]+1); first.append(r)
        if not y[order[0]]: near+=1
        for k in top: top[k]+=int(y[order[:k]].any())
        if (not y[raw_order[0]]) and lab.any():
            pair_total+=1
            pos=idxs[lab]
            best_pos=pos[np.argmax(s[pos])]
            win=bool(s[best_pos] > s[raw_order[0]])
            pair_win += int(win)
            if len(case)<100:
                case.append({'split':split,'model':kind,'seed':seed,'episode':ep,'raw_rank0_score':float(s[raw_order[0]]),'best_success_score':float(s[best_pos]),'success_over_rank0':win,'raw_first_success_rank':int(np.where(y[raw_order])[0][0]+1),'score_first_success_rank':r,'success_count':int(lab.sum())})
    n=len(gs); arr=np.asarray(first,dtype=float); intra_arr=np.asarray(intra,dtype=float)
    return {'split':split,'model':kind,'episodes':n,'global_auc':auc(y[mask],s[mask]),'top1_success':top[1]/n*100,'top3_success':top[3]/n*100,'top5_success':top[5]/n*100,'top10_success':top[10]/n*100,'top30_success':top[30]/n*100,'first_success_rank_mean':float(arr.mean()),'first_success_rank_median':float(np.median(arr)),'near_miss_count':near,'intra_episode_auc_mean':float(intra_arr.mean()),'intra_episode_auc_median':float(np.median(intra_arr)),'frac_intra_auc_gt_0p7':float((intra_arr>0.7).mean()),'pair_rank0_total':pair_total,'pair_success_over_rank0':pair_win,'pair_success_over_rank0_rate':pair_win/max(pair_total,1)*100,'cases':case}

def write_csv(path,rows,fields=None):
    if fields is None:
        fields=[]
        for r in rows:
            for k in r:
                if k!='cases' and k not in fields: fields.append(k)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader();
        for r in rows: w.writerow({k:r.get(k,'') for k in fields})

def aggregate(rows):
    out=[]
    for model in sorted(set(r['model'] for r in rows)):
        rs=[r for r in rows if r['model']==model]; ep=sum(r['episodes'] for r in rs)
        out.append({'model':model,'episodes':ep,'global_auc':float(np.mean([r['global_auc'] for r in rs])),'top1_success':sum(r['top1_success']*r['episodes']/100 for r in rs)/ep*100,'top3_success':sum(r['top3_success']*r['episodes']/100 for r in rs)/ep*100,'top5_success':sum(r['top5_success']*r['episodes']/100 for r in rs)/ep*100,'top10_success':sum(r['top10_success']*r['episodes']/100 for r in rs)/ep*100,'top30_success':sum(r['top30_success']*r['episodes']/100 for r in rs)/ep*100,'first_success_rank_mean':float(np.mean([r['first_success_rank_mean'] for r in rs])),'first_success_rank_median':float(np.median([r['first_success_rank_median'] for r in rs])),'near_miss_count':sum(r['near_miss_count'] for r in rs),'intra_episode_auc_mean':float(np.mean([r['intra_episode_auc_mean'] for r in rs])),'intra_episode_auc_median':float(np.mean([r['intra_episode_auc_median'] for r in rs])),'frac_intra_auc_gt_0p7':float(np.mean([r['frac_intra_auc_gt_0p7'] for r in rs])),'pair_rank0_total':sum(r['pair_rank0_total'] for r in rs),'pair_success_over_rank0':sum(r['pair_success_over_rank0'] for r in rs),'pair_success_over_rank0_rate':sum(r['pair_success_over_rank0'] for r in rs)/max(sum(r['pair_rank0_total'] for r in rs),1)*100})
    return sorted(out,key=lambda r:(-r['top1_success'], -r['intra_episode_auc_mean']))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows,y,raw,meta=build()
    feature_models=['episode_only','candidate_only','cost','trajectory','action','latent','cost_trajectory','all']
    all_rows=[]; case_rows=[]
    for split,(train,val) in SPLITS.items():
        # raw baseline
        ev=eval_score(y,raw,raw,meta,val,split,'raw_cost')
        case_rows.extend(ev.pop('cases')); all_rows.append(ev)
        for kind in feature_models:
            X=make_X(rows,kind); model=fit_bce(X,y,meta,train); s=score(model,X)
            ev=eval_score(y,s,raw,meta,val,split,kind)
            case_rows.extend(ev.pop('cases')); all_rows.append(ev)
    agg=aggregate(all_rows)
    write_csv(OUT/'moda_only_intra_episode_audit.csv',all_rows)
    write_csv(OUT/'moda_only_feature_ablation.csv',agg)
    write_csv(OUT/'moda_only_pairwise_rank0_audit.csv',case_rows)
    verdict='feature_level_not_identifiable'
    best=max(agg,key=lambda r:r['top1_success'])
    best_intra=max(agg,key=lambda r:r['intra_episode_auc_mean'])
    if best['top1_success']>=65 and best_intra['intra_episode_auc_mean']>0.7:
        verdict='feature_level_identifiable'
    elif best['top1_success']>=60:
        verdict='weakly_identifiable_but_below_target'
    (OUT/'moda_only_intra_episode_audit.json').write_text(json.dumps({'oof_summary':agg,'per_split':all_rows,'verdict':verdict},indent=2)+'\n')
    md=['# MoDA-Only Intra-Episode Discriminability Audit','', 'Only stateroll/MoDA candidate pool is used. This is a diagnostic audit, not a new selector result. Main purpose: explain why global BCE AUC can be high while MoDA-only top1 does not improve.','', '## OOF Summary','', '|model|top1|global AUC|intra AUC mean|intra AUC median|frac intra AUC > 0.7|pair success>rank0|near-miss|first rank|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in agg:
        md.append(f"|{r['model']}|{r['top1_success']:.2f}|{r['global_auc']:.3f}|{r['intra_episode_auc_mean']:.3f}|{r['intra_episode_auc_median']:.3f}|{r['frac_intra_auc_gt_0p7']:.3f}|{r['pair_success_over_rank0_rate']:.1f}% ({r['pair_success_over_rank0']}/{r['pair_rank0_total']})|{r['near_miss_count']}|{r['first_success_rank_mean']:.2f}|")
    md += ['', '## Interpretation','']
    ep=[r for r in agg if r['model']=='episode_only'][0]
    cand=[r for r in agg if r['model']=='candidate_only'][0]
    rawr=[r for r in agg if r['model']=='raw_cost'][0]
    md.append(f"Episode-only diagnostic: global AUC={ep['global_auc']:.3f}, top1={ep['top1_success']:.2f}, intra-episode AUC={ep['intra_episode_auc_mean']:.3f}. Because episode-only features are constant across candidates, any high global AUC here indicates episode difficulty leakage rather than candidate discrimination.")
    md.append('')
    md.append(f"Candidate-only diagnostic: global AUC={cand['global_auc']:.3f}, top1={cand['top1_success']:.2f}, intra-episode AUC={cand['intra_episode_auc_mean']:.3f}, success-over-rank0={cand['pair_success_over_rank0_rate']:.1f}%. This is the relevant signal for MoDA-only top1 rescue.")
    md.append('')
    md.append(f"Raw cost baseline: top1={rawr['top1_success']:.2f}, intra-episode AUC={rawr['intra_episode_auc_mean']:.3f}, success-over-rank0={rawr['pair_success_over_rank0_rate']:.1f}%. Best top1 model is `{best['model']}` with top1={best['top1_success']:.2f}. Verdict: `{verdict}`.")
    if verdict=='feature_level_not_identifiable':
        md.append('')
        md.append('Conclusion: feature-level MoDA-only selection is not identifiable enough in the current candidate features. High global AUC is not sufficient; intra-episode discrimination and success-vs-rank0 rescue are too weak to improve top1. The bsl-integrated results therefore cannot be used as the main standalone MoDA planner story.')
    (OUT/'moda_only_intra_episode_audit.md').write_text('\n'.join(md)+'\n')
    print((OUT/'moda_only_intra_episode_audit.md').read_text())

if __name__=='__main__': main()
