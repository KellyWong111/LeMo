from __future__ import annotations
import argparse, json, os
from pathlib import Path
from collections import deque

os.environ.setdefault('MUJOCO_GL','egl')

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

import analyze_cem_margin as base
from topk_oracle_pilot import eval_fixed_plans

ROOT=Path('/data1/jingyixi/wm_runs')
ST_ACTION=ROOT/'stateroll_normalbudget_candidate_pool_s300_steps30_n100'/'proposal_data'
ST_RAW=ROOT/'stateroll_normalbudget_candidate_pool_s300_steps30_n100'/'raw_rollout_npz'
OUT=ROOT/'moda_only_calibrated_cem_20260529'
SEEDS=[42,43,44,45,46,47]
SPLITS={'splitA_train42_44_val45_47':([42,43,44],[45,46,47]),'splitB_train45_47_val42_44':([45,46,47],[42,43,44])}
POLICY='pusht_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003/lewm_encoder_moda_v14_full_visible_bs32_pred6_gate07_staterollseq_l003_epoch_1'


def goal_for_pred_np(goal,pred):
    g=goal
    if g.ndim==2: g=g[:,None,:]
    if g.ndim==3 and pred.ndim==4: g=g[:,None,:,:]
    if g.shape[1]==1: g=np.repeat(g,pred.shape[1],axis=1)
    if g.shape[2]==1: g=np.repeat(g,pred.shape[2],axis=2)
    elif g.shape[2]!=pred.shape[2]: g=g[:,:,-pred.shape[2]:,:]
    return g

def pool_features_for_seed(seed):
    a=np.load(ST_ACTION/f'vf05_mix20_seed{seed}.npz',allow_pickle=True)
    r=np.load(ST_RAW/f'vf05_mix20_seed{seed}.npz',allow_pickle=True)
    costs=a['costs'].astype(np.float64); labels=a['labels'].astype(bool)
    pred=r['pred'].astype(np.float64); goal=r['goal'].astype(np.float64); actions=a['actions'].astype(np.float64)
    g=goal_for_pred_np(goal,pred); dist=np.sqrt(((pred-g)**2).sum(axis=-1))
    final=dist[:,:,-1]; mean=dist.mean(axis=2); mind=dist.min(axis=2); progress=dist[:,:,0]-dist[:,:,-1]
    latent_mean=pred.mean(axis=(2,3)); latent_std=pred.std(axis=(2,3))
    anorm=np.sqrt((actions**2).sum(axis=-1)); action_mean=anorm.mean(axis=2); action_std=anorm.std(axis=2)
    adiff=np.sqrt((np.diff(actions,axis=2)**2).sum(axis=-1)) if actions.shape[2]>1 else np.zeros_like(anorm[:,:,:1])
    action_smooth=adiff.mean(axis=2)
    X=[]; y=[]
    E,K=costs.shape
    for ep in range(E):
        c=costs[ep]; order=np.argsort(c,kind='stable'); ranks=np.empty(K,dtype=int); ranks[order]=np.arange(K); sorted_c=c[order]
        cz=(c-c.mean())/(c.std()+1e-6)
        for j in range(K):
            X.append([1.0, -c[j], c[j], ranks[j]/(K-1), cz[j], c[j]-sorted_c[0], c[j]-sorted_c[min(4,K-1)], final[ep,j], mean[ep,j], mind[ep,j], progress[ep,j], action_mean[ep,j], action_std[ep,j], action_smooth[ep,j], latent_mean[ep,j], latent_std[ep,j]])
            y.append(bool(labels[ep,j]))
    return np.asarray(X,dtype=np.float64),np.asarray(y,dtype=bool)

def fit_utility(train_seeds, epochs=1800, lr=0.03, l2=1e-3):
    Xs=[]; ys=[]
    for s in train_seeds:
        X,y=pool_features_for_seed(s); Xs.append(X); ys.append(y)
    X=np.concatenate(Xs); y=np.concatenate(ys)
    mean=X.mean(axis=0); std=X.std(axis=0)+1e-6; mean[0]=0; std[0]=1
    Z=(X-mean)/std; Z[:,0]=1
    yy=y.astype(float); pos=yy.sum(); neg=len(yy)-pos
    wt=np.ones(len(yy)); wt[y]=max(1.0,neg/max(pos,1.0)); wt/=wt.mean()
    w=np.zeros(Z.shape[1])
    for _ in range(epochs):
        p=1/(1+np.exp(-np.clip(Z@w,-40,40)))
        grad=Z.T@((p-yy)*wt)/len(yy); grad[1:]+=l2*w[1:]; w-=lr*grad
    return {'w':w.astype(np.float32),'mean':mean.astype(np.float32),'std':std.astype(np.float32)}

def model_rollout_cost(model, info_dict, candidates):
    # Copy of JEPA get_cost, but keeps predicted_emb/goal_emb for utility features.
    device=next(model.parameters()).device
    info={}
    for k,v in info_dict.items():
        info[k]=v.to(device) if torch.is_tensor(v) else v
    goal={k:v[:,0] for k,v in info.items() if torch.is_tensor(v)}
    goal['pixels']=goal['goal']
    for k in list(info.keys()):
        if k.startswith('goal_'):
            goal[k[len('goal_'):]]=goal.pop(k)
    goal.pop('action')
    goal=model.encode(goal)
    info['goal_emb']=goal['emb']
    info=model.rollout(info,candidates)
    cost=model.criterion(info)
    return cost, info['predicted_emb'], info['goal_emb']

def utility_score_torch(model_cost, pred, goal_emb, candidates, util):
    # Returns higher-is-better utility, shape (B,S). Features match pool scalar subset.
    B,S=model_cost.shape; K=S
    goal=goal_emb
    if goal.ndim==pred.ndim-1: goal=goal.unsqueeze(1)
    if goal.shape[1]==1 and pred.shape[1]!=1: goal=goal.expand(-1,pred.shape[1],-1,-1)
    goal=goal[...,-1:,:].expand_as(pred)
    dist=torch.sqrt(((pred-goal)**2).sum(dim=-1)+1e-12)
    final=dist[:,:,-1]; mean=dist.mean(dim=2); mind=dist.min(dim=2).values; progress=dist[:,:,0]-dist[:,:,-1]
    latent_mean=pred.mean(dim=(2,3)); latent_std=pred.std(dim=(2,3))
    anorm=torch.sqrt((candidates**2).sum(dim=-1)+1e-12); action_mean=anorm.mean(dim=2); action_std=anorm.std(dim=2)
    if candidates.shape[2]>1:
        adiff=torch.sqrt((torch.diff(candidates,dim=2)**2).sum(dim=-1)+1e-12); action_smooth=adiff.mean(dim=2)
    else:
        action_smooth=torch.zeros_like(action_mean)
    c=model_cost
    order=torch.argsort(c,dim=1,stable=True)
    ranks=torch.empty_like(order)
    ar=torch.arange(K,device=c.device)[None,:].expand(B,K)
    ranks.scatter_(1,order,ar)
    sorted_c=torch.gather(c,1,order)
    cz=(c-c.mean(dim=1,keepdim=True))/(c.std(dim=1,keepdim=True)+1e-6)
    feat=torch.stack([torch.ones_like(c), -c, c, ranks.float()/max(K-1,1), cz, c-sorted_c[:,0:1], c-sorted_c[:,min(4,K-1):min(4,K-1)+1], final, mean, mind, progress, action_mean, action_std, action_smooth, latent_mean, latent_std],dim=-1)
    mean=torch.tensor(util['mean'],device=c.device,dtype=feat.dtype); std=torch.tensor(util['std'],device=c.device,dtype=feat.dtype); w=torch.tensor(util['w'],device=c.device,dtype=feat.dtype)
    z=(feat-mean)/std; z[...,0]=1
    return torch.tensordot(z,w,dims=([-1],[0]))

@torch.inference_mode()
def get_calibrated_cem_candidates(model, prepared_base, action_dim, horizon, num_samples, topk, n_steps, seed, lamb, util, restarts=1):
    device='cuda'; num_envs=next(v for v in prepared_base.values() if torch.is_tensor(v)).shape[0]
    all_cands=[]; all_costs=[]; all_raw=[]; all_util=[]
    for restart in range(restarts):
        mean=torch.zeros(num_envs,horizon,action_dim,device=device); var=torch.ones(num_envs,horizon,action_dim,device=device); gen=torch.Generator(device=device).manual_seed(seed+restart*1009)
        final_cand=final_cal=final_raw=final_u=None
        for _ in range(n_steps):
            candidates=torch.randn(num_envs,num_samples,horizon,action_dim,generator=gen,device=device)
            candidates=candidates*var[:,None]+mean[:,None]; candidates[:,0]=mean
            prepared={k:v.clone() if torch.is_tensor(v) else v for k,v in prepared_base.items()}
            prepared=base.expand_info_for_candidates(prepared,num_envs,num_samples)
            raw_cost,pred,goal=model_rollout_cost(model,prepared,candidates)
            u=utility_score_torch(raw_cost,pred,goal,candidates,util)
            cal=raw_cost-float(lamb)*u
            _,idx=torch.topk(cal,k=topk,dim=1,largest=False)
            batch=torch.arange(num_envs,device=device)[:,None]
            elite=candidates[batch,idx]
            mean=elite.mean(dim=1); var=elite.std(dim=1)
            final_cand=candidates.detach().cpu(); final_cal=cal.detach().cpu(); final_raw=raw_cost.detach().cpu(); final_u=u.detach().cpu()
        order=torch.argsort(final_cal,dim=1)[:,:topk]
        batch_cpu=torch.arange(num_envs)[:,None]
        all_cands.append(final_cand[batch_cpu,order]); all_costs.append(final_cal[batch_cpu,order]); all_raw.append(final_raw[batch_cpu,order]); all_util.append(final_u[batch_cpu,order])
    cands=torch.cat(all_cands,dim=1); cal=torch.cat(all_costs,dim=1); raw=torch.cat(all_raw,dim=1); uu=torch.cat(all_util,dim=1)
    order=torch.argsort(cal,dim=1)[:,:topk]; batch=torch.arange(cands.shape[0])[:,None]
    return cands[batch,order], cal[batch,order], raw[batch,order], uu[batch,order]

def run_eval(split, train_seeds, val_seeds, lambdas, num_eval, num_samples, topk, cem_steps, seed, restarts):
    util=fit_utility(train_seeds)
    cfg=OmegaConf.load('./config/eval/pusht.yaml'); cfg.policy=POLICY; cfg.eval.num_eval=num_eval; OmegaConf.update(cfg,'world.max_episode_steps',2*cfg.eval.eval_budget,merge=True)
    dataset=base.get_dataset(cfg); process=base.get_process(cfg,dataset); valid_indices=base.get_valid_indices(cfg,dataset)
    # Use deterministic per requested seed; this is eval sampling seed, not bsl-related.
    rng=np.random.default_rng(seed); picked=np.sort(rng.choice(len(valid_indices)-1,size=num_eval,replace=False)); indices=valid_indices[picked]
    raw_info=base.build_info_dict(cfg,dataset,process,indices)
    transform={'pixels':base.img_transform(cfg),'goal':base.img_transform(cfg)}
    prepared_base=base.make_eval_like_info(raw_info,transform,process)
    world_tmp=swm.World(**OmegaConf.to_container(cfg.world,resolve=True),image_shape=(224,224))
    low=np.asarray(world_tmp.envs.action_space.low); low=low[0] if low.ndim>1 else low
    action_dim=int(np.prod(low.shape))*int(cfg.plan_config.action_block)
    model=base.load_model(cfg,cache_dir=None)
    rows=[]; cases=[]
    for lamb in lambdas:
        cands, cal, raw, uu=get_calibrated_cem_candidates(model,prepared_base,action_dim,int(cfg.plan_config.horizon),num_samples,topk,cem_steps,seed,lamb,util,restarts=restarts)
        plans=cands.numpy()
        labels=[]
        for rank in range(topk):
            metrics=eval_fixed_plans(cfg,dataset,process,indices,plans[:,rank])
            labels.append(np.asarray(metrics['episode_successes'],dtype=bool))
        labels=np.stack(labels,axis=1)
        first=[]
        for row in labels:
            hit=np.nonzero(row)[0]; first.append(int(hit[0]+1) if len(hit) else None)
        rows.append({'split':split,'lambda':float(lamb),'num_eval':num_eval,'num_samples':num_samples,'cem_steps':cem_steps,'top1_success':float(labels[:,0].mean()*100),'top3_success':float(labels[:,:min(3,topk)].any(axis=1).mean()*100),'top5_success':float(labels[:,:min(5,topk)].any(axis=1).mean()*100),'top10_success':float(labels[:,:min(10,topk)].any(axis=1).mean()*100),'top30_success':float(labels.any(axis=1).mean()*100),'episodes_with_success':int(labels.any(axis=1).sum()),'near_miss_count':int(((~labels[:,0]) & labels.any(axis=1)).sum()),'mean_cal_cost':float(cal[:,0].mean()),'mean_raw_cost':float(raw[:,0].mean()),'mean_utility':float(uu[:,0].mean())})
        for i,fr in enumerate(first):
            if (not labels[i,0]) and labels[i].any() and len(cases)<200:
                cases.append({'split':split,'lambda':float(lamb),'eval_i':int(i),'dataset_index':int(indices[i]),'first_success_rank':fr,'top1_raw_cost':float(raw[i,0]),'top1_cal_cost':float(cal[i,0]),'top1_utility':float(uu[i,0])})
    del model; torch.cuda.empty_cache()
    return rows,cases

def write_csv(path,rows):
    if not rows: return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open('w',newline='') as f:
        import csv
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--num-eval',type=int,default=20); ap.add_argument('--num-samples',type=int,default=150); ap.add_argument('--cem-steps',type=int,default=15); ap.add_argument('--topk',type=int,default=10); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--restarts',type=int,default=1); ap.add_argument('--lambdas',default='0,0.1,0.2,0.5,1.0,2.0'); ap.add_argument('--outdir',default=str(OUT)); args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    lambdas=[float(x) for x in args.lambdas.split(',')]
    all_rows=[]; all_cases=[]
    for split,(train,val) in SPLITS.items():
        rows,cases=run_eval(split,train,val,lambdas,args.num_eval,args.num_samples,args.topk,args.cem_steps,args.seed,args.restarts)
        all_rows.extend(rows); all_cases.extend(cases)
    write_csv(out/'moda_only_calibrated_cem_lambda_sweep.csv',all_rows); write_csv(out/'moda_only_calibrated_cem_case_studies.csv',all_cases)
    # Aggregate by lambda.
    agg=[]
    for lamb in lambdas:
        rs=[r for r in all_rows if abs(r['lambda']-lamb)<1e-9]
        if not rs: continue
        agg.append({'lambda':lamb,'top1_success':float(np.mean([r['top1_success'] for r in rs])),'top3_success':float(np.mean([r['top3_success'] for r in rs])),'top5_success':float(np.mean([r['top5_success'] for r in rs])),'top10_success':float(np.mean([r['top10_success'] for r in rs])),'top30_success':float(np.mean([r['top30_success'] for r in rs])),'near_miss_count':int(sum(r['near_miss_count'] for r in rs))})
    best=max(agg,key=lambda r:r['top1_success']) if agg else None
    report={'settings':vars(args),'policy':POLICY,'rows':all_rows,'aggregate':agg,'best':best}
    (out/'moda_only_calibrated_cem.json').write_text(json.dumps(report,indent=2)+'\n')
    md=['# MoDA-Only Calibrated CEM Cost Injection','', 'This is planner-level cost injection, not final candidate reranking. Only stateroll/MoDA planner is used; no bsl fallback or fixed/harmed metrics. CEM elite selection uses `J_cal = J_raw - lambda * U_theta(candidate)`.','', '## Lambda Sweep','', '|lambda|top1|top3|top5|top10|top30|near-miss|','|---:|---:|---:|---:|---:|---:|---:|']
    for r in agg:
        md.append(f"|{r['lambda']}|{r['top1_success']:.2f}|{r['top3_success']:.2f}|{r['top5_success']:.2f}|{r['top10_success']:.2f}|{r['top30_success']:.2f}|{r['near_miss_count']}|")
    if best: md += ['', '## Verdict','', f"Best top1 is {best['top1_success']:.2f} at lambda={best['lambda']}. Raw lambda=0 top1 is {next((r['top1_success'] for r in agg if r['lambda']==0), float('nan')):.2f}."]
    (out/'moda_only_calibrated_cem.md').write_text('\n'.join(md)+'\n')
    print((out/'moda_only_calibrated_cem.md').read_text())

if __name__=='__main__': main()
