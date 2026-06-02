# 03. MoDA-only 诊断代码

对应文件：

```text
experiments/moda_only_intra_episode_audit.py
experiments/moda_only_planner_in_loop_calibrated_cem.py
experiments/moda_only_action_sensitive_contrastive.py
experiments/moda_only_search_scaling.py
```

## 这部分代码解决什么问题

MoDA candidate pool 的初步结果显示：oracle 不低，但 top1 不强。也就是说，MoDA 有成功候选，但 raw cost 不会选。

这部分代码的目标是找根因：到底是 candidate pool 不够，还是 candidate scoring 不行，还是后处理 reranking 本身不可行。

## `moda_only_intra_episode_audit.py`

这是最关键的诊断脚本。

它做了几件事：

1. 计算 global candidate AUC；
2. 训练 episode-only baseline；
3. 训练 candidate-only baseline；
4. 计算 intra-episode AUC；
5. 统计 success candidate 能否压过 raw rank0 failure。

为什么这很重要？

因为之前 BCE / calibrated score 的 global AUC 看起来很高，但 top1 不涨。这个脚本证明，高 AUC 很多来自 episode difficulty leakage。也就是说，模型知道某个 episode 容不容易成功，但不知道同一个 episode 里哪个 candidate 应该排第一。

汇报时可以这样讲：

> 这个 audit 把问题从“是不是 loss 没调好”推进到“同 episode 内候选特征是否可分”。结果说明 post-hoc feature-level reranking 不是稳定解。

## `moda_only_planner_in_loop_calibrated_cem.py`

这个脚本尝试把 calibrated utility 注入 CEM 内部。

原始 CEM 用：

```text
J_raw
```

这个脚本尝试：

```text
J_plan = J_raw - lambda * U_theta
```

也就是说，不是最后才 rerank，而是在 CEM elite selection 和 final action selection 阶段都使用 calibrated cost。

结果是有小幅提升，但不够强。它的意义是诊断：

> 只把 calibrated score 注入 CEM，仍然不能稳定解决 MoDA-only top1。

## `moda_only_action_sensitive_contrastive.py`

这个脚本尝试让 success candidate 和 near-miss failure 的 embedding 分开。

它的想法是：如果 raw rank0 failure 和 success candidate 在 latent/action feature 上太接近，就通过 contrastive objective 增加 separation。

但结果 top1 没有改善，甚至下降。因此它不是主方法，而是负结果：

> 表征分离 proxy 指标改善，不一定带来 planning top1 改善。

## `moda_only_search_scaling.py`

这个脚本检查是不是搜索预算不够。

它改变 candidates、CEM steps、top-k 等设置，观察 oracle 和 top1 是否随搜索规模增加而提升。

结论是：单纯扩大搜索预算不能稳定修好 MoDA-only planner。

## 这部分最终结论

这几条诊断共同说明：

1. MoDA 有 candidate coverage；
2. raw cost 和 success 不对齐；
3. post-hoc reranking 不够；
4. planner-in-loop cost injection 有小幅信号但不稳定；
5. 真正要改的是 candidate generation / proposal。

