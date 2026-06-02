# MoDA / LeWM 技术路径汇报讲稿

## 一句话版本
MoDA 不是没有候选价值，而是 raw cost 不能稳定把成功候选排到 top1。bsl-integrated 能涨分，但不是 MoDA-only；真正 MoDA-only 的正信号来自 success-conditioned residual proposal，它通过修正 action proposal 减少 near-miss failure，带来一致的 paired improvement，但还没稳定到可以宣称 standalone 65+。

## 讲法
1. 我们先把结果分成三类，避免混淆：bsl-relative integration、AUC 假阳性、MoDA-only residual proposal。
2. bsl-integrated 的 81 -> 83.17 说明 MoDA 有互补候选，但方法形态仍是强 baseline 保底，所以不能作为 MoDA-only 主结果。
3. MoDA-only 的后处理 reranker 不成立：episode-only AUC 也能到 0.692，但 intra-episode AUC 是 0.5；candidate-only intra-episode AUC 也只有 0.559，所以高 AUC 很多是 episode difficulty leakage。
4. 因此我们转向 candidate generation correction，而不是 final rerank。
5. residual proposal 的做法是：在 raw rank0 失败但存在成功候选的 episode 中，学习 action residual delta_a = success_action - raw_rank0_action；评估时生成 shifted candidates，再用 raw/calibrated cost 选择。
6. medium20 原始结果 raw 60 -> residual 65，但 paired audit 发现 65 不稳定；confirm50 上 raw 46 -> residual 49/50，near-miss 14 -> 10，说明方向有相对改善但绝对值受 eval set 难度影响很大。
7. 当前最稳结论：post-hoc rerank 死路；candidate generation correction 是唯一 MoDA-only 正向方向。下一步做 residual gate 和 scale selection。

## 不能说的话
- 不能说 PAC-MoDA 已经是稳定 MoDA-only 65+ planner。
- 不能把 bsl-integrated 83.17 当成 MoDA-only 结果。
- 不能只报 AUC 作为 planning success。

## 可以说的话
- MoDA candidate coverage 有价值，但 cost-success alignment 差。
- 后处理 feature-level reranking 在 intra-episode 层面不可辨识。
- success-conditioned residual proposal 是一个更合理的 MoDA-only candidate generation correction，当前显示 consistent paired gain 和 near-miss reduction。
