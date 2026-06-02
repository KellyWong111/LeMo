# LeWM + MoDA 技术路径汇报讲稿

## 1. 开场
我这次汇报主要讲 LeWM 和 MoDA 的结合路径。LeWM 本身是 latent world model 加 CEM planning 的控制框架，MoDA 的优势是通过 depth-aware attention 做跨层信息检索。我做的工作不是简单把一个模块塞进去，而是从结构接入、训练、planning candidate pool、失败诊断，一直推进到 action proposal correction。

## 2. 任务边界
LeWM 官方或相关 checkpoint 不是只有 PushT，还包括 Cube、TwoRooms、Reacher 等任务。但我当前跑得最深的是 PushT。PushT 有连续动作、接触动力学和明显 near-miss failure，所以适合作为 MoDA-for-planning 的诊断环境。这里不能说我们已经在所有任务上验证了 MoDA，只能说框架有多任务基础，当前系统诊断主要在 PushT。

## 3. LeWM 和 MoDA
LeWM 的核心流程是：observation 经过 encoder 变成 latent state，predictor 在 latent 空间 rollout 未来状态，CEM planner 采样 action sequence 并通过 cost 选出最终 action。MoDA 的理论优势是 mixture-of-depth attention，也就是当前层可以通过 depth cache 检索前面层的 K/V 信息。这个机制放到 LeWM 里，目标是增强 predictor 的 state rollout 表达能力。

## 4. 代码接入
我主要实现了 `moda_module.py` 和 `moda_module_exact.py`。其中 exact 版本尽量保留 LeWM 原来的 conditional block 和 AdaLN-zero conditioning，只把 attention kernel 换成 MoDA attention。`MoDAAttentionExact` 在没有 cache 时退回普通 causal attention，在有 cached K/V 时调用 MoDA kernel。`MoDATransformerExact` 负责逐层收集 K/V，构造 depth cache。

训练入口包括 `train_encoder_moda.py`、`train_moda.py`、`train_moda_exact.py`。这些脚本把 MoDA predictor/encoder 接进 LeWM 的训练 pipeline，产出后续 planning evaluation 使用的 checkpoint。

## 5. Planning 诊断
接入后我们进入 planning evaluation，生成 stateroll/MoDA candidate pool。核心发现是：MoDA candidate pool 里不是没有成功候选，oracle 不低；但 raw MoDA cost 不能稳定把成功候选选到 top1。也就是说，问题不只是 coverage，而是 cost-success alignment。

一开始我尝试 BCE、calibration、selector、reranker。global AUC 看起来能涨，但 `moda_only_intra_episode_audit.py` 证明这很多是 episode difficulty leakage。episode-only 特征 AUC 都能到 0.692，但 intra-episode AUC 是 0.5；candidate-only intra-episode AUC 也只有 0.559。所以模型学到的是这个 episode 难不难，而不是同一个 episode 里哪个 candidate 应该排第一。

## 6. Baseline-safe integration
我也做过 PAC-MoDA / risk-controlled integration。它的逻辑是 baseline planner 保底，MoDA 只在 gate 判断为低风险、有机会修 baseline failure 时介入。这条线可以把系统 top1 从 81 左右提升到 82/83，说明 MoDA 有辅助候选价值。但它依赖 baseline fallback，所以不能当 MoDA-only standalone planner 结果。

## 7. Residual Proposal
当前最重要的 MoDA-only 方向是 success-conditioned residual proposal。它不再做 final rerank，而是直接修正 action proposal。具体来说，对于 raw rank0 失败但同 episode 有成功候选的样本，学习 `delta_a = success_action - raw_rank0_action`。评估时对 raw top candidates 生成 shifted candidates，再用 raw/calibrated cost 重新评估选择。

代码上，`train_data` 负责构造 residual target，`feature_np` 和 `online_feature` 负责构造离线/在线一致的特征，`fit_ridge` 学 residual，`score_candidates` 重新 rollout 打分，`run_split` 做完整评估。`moda_only_residual_confirm50_audit.py` 用固定 indices 做 paired audit，避免把不同 eval set 的 absolute top1 误读成方法变化。

## 8. 结果和口径
medium20 曾经有 raw 60 到 residual 65 的结果，但 confirm50 后更稳的结论是 raw 46 到 residual 49/50，near-miss 从 14 降到 10/11。所以不能说 MoDA-only 已经稳定 65+，但可以说 residual proposal 是目前唯一真正 MoDA-only 的正向方向：它有 consistent paired gain，并且能减少 near-miss failure。

## 9. 最终结论
一句话总结：MoDA 的候选覆盖和 depth/state rollout 有价值，但 raw cost 与 planning success 不完全对齐。我们系统排除了只靠 AUC calibration 和 final rerank 的路线，最后发现更合理的方向是从 action proposal generation 层面做 success-conditioned residual correction。
