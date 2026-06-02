# 05. Success-conditioned residual proposal

对应文件：

```text
experiments/moda_only_learned_residual_proposal.py
experiments/moda_only_residual_confirm50_audit.py
```

## 这部分代码为什么重要

这是当前最重要的 MoDA-only 正向方向。

前面所有 post-hoc reranking 都有一个问题：CEM 已经生成了 final candidates，失败候选已经进入 rank0，再去重排往往分不清 success 和 near-miss failure。

residual proposal 换了思路：

不只改最后排序，而是直接改 action proposal。

## `moda_only_learned_residual_proposal.py`

这个脚本的核心目标是学习：

```text
delta_a = success_action - raw_rank0_failure_action
```

也就是说，对于 raw MoDA rank0 失败但同 episode 里存在成功 candidate 的样本，学习从失败 action 到成功 action 的方向。

### `train_data`

这个函数从 stateroll/MoDA candidate pool 里构造训练样本。

它遍历 train seeds。对于每个 episode：

1. 找 raw cost 最低的 rank0 candidate；
2. 检查 rank0 是否失败；
3. 检查同 episode 是否存在成功 candidate；
4. 找 raw cost 最低的成功 candidate；
5. 构造特征 `x`；
6. 构造 residual target `y = success_action - rank0_action`。

这一步是 residual proposal 的监督信号来源。

### `feature_np`

这个函数构造离线候选特征。

特征包括：

- raw cost；
- negative raw cost；
- local rank；
- within-episode z-score；
- cost gap to rank0 / top5；
- trajectory final / mean / min distance；
- progress；
- latent mean/std；
- action norm/std；
- action sequence 本身。

这组特征描述的是：“当前 raw rank0 failure 长什么样，以及它和候选池 cost geometry 的关系”。

### `fit_ridge`

这个函数用 ridge regression 学 residual。

这里没有用复杂神经网络，是因为当时目标是快速验证 candidate generation correction 是否有信号。ridge 的好处是稳定、可解释、不容易过拟合到很夸张。

### `online_feature`

训练时特征来自离线 candidate pool。评估时，CEM 在线生成 raw candidates，所以必须重新构造同样格式的 feature。

`online_feature` 的作用就是在线计算 raw cost、trajectory distance、progress、action statistics，并拼成和训练时一致的 feature。

### `score_candidates`

这个函数对候选 action 做 world model rollout，得到：

```text
raw cost
calibrated utility
plan score = raw cost - lambda * utility
```

也就是说，shifted candidates 不是盲目使用，而是重新经过 world model cost / calibrated cost 评估。

### `run_split`

这是主流程：

1. 用 train seeds 训练 residual model；
2. 加载 utility calibration；
3. 在线 CEM 生成 raw MoDA top-k candidates；
4. 用 residual model 预测 `delta_a`；
5. 对 raw top candidates 做 shift；
6. 把 raw candidates 和 shifted candidates 合并；
7. 用 raw cost / calibrated cost 排序；
8. 用真实 rollout label 评估 top1/top3/top5/oracle/near-miss。

汇报时这一段可以画成流程图：

```text
raw MoDA CEM candidates
        ↓
detect raw rank0 feature
        ↓
predict residual delta_a
        ↓
generate shifted candidates
        ↓
evaluate by raw/calibrated cost
        ↓
select MoDA-only action
```

## `moda_only_residual_confirm50_audit.py`

这个脚本用于确认 residual proposal 的结果是否稳定。

为什么需要它？

因为 medium20 曾经出现过：

```text
raw 60 -> residual 65
```

但 confirm50 结果变成：

```text
raw 46 -> residual 49/50
```

所以必须判断：是方法没用，还是 eval set 不一样。

### `choose_indices`

这个函数固定随机种子抽 eval indices。

它的意义是防止 medium20 和 confirm50 被误解成同一批 episode。实际上，不同 `num_eval` 会重新采样，不一定是“前 20 个 + 新 30 个”。

### `paired_metrics`

这个函数计算 residual 相对 raw MoDA 的：

```text
fixed
harmed
net
both_success
both_fail
```

注意这里不是 bsl fixed/harmed，而是 raw MoDA vs residual MoDA 的 paired comparison，所以仍然是 MoDA-only。

### subset audit

脚本把 confirm50 拆成：

```text
first20
added30
all50
```

并在同一批 indices 上扫：

```text
scale = 0.1, 0.2, 0.25, 0.3, 0.4, 0.5
```

这一步得出最终稳定结论：

```text
scale 0.25: raw 46 -> residual 49, near-miss 14 -> 11
scale 0.4/0.5: raw 46 -> residual 50, near-miss 14 -> 10
```

## 这部分最终结论

residual proposal 不是成熟到可以说 stable 65+，但它是目前唯一真正 MoDA-only 的正向方向。

它说明：

MoDA 的问题不是完全没有成功候选，而是 raw proposal 容易落到 near-miss failure 附近。通过学习 success-conditioned residual，可以从 candidate generation 层面减少 near-miss failure。

