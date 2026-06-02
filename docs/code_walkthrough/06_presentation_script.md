# 06. 汇报代码时的讲稿

可以按下面这段讲：

我这个项目最开始是把 MoDA 接到 LeWM 的 world model predictor 里。LeWM 本身是 latent world model + CEM planning，核心是用 latent rollout 评估 action sequence 的 cost。MoDA 的理论优势是通过 mixture-of-depth attention 做跨 depth 的信息检索，所以我先实现了 `moda_module.py` 和 `moda_module_exact.py`，其中 exact 版本保留 LeWM 的 conditional block 和 AdaLN-zero，只把 attention 换成 official MoDA Triton kernel。

接入以后，我通过 `train_encoder_moda.py`、`train_moda.py`、`train_moda_exact.py` 跑 MoDA encoder / predictor 训练。然后进入 planning 评估阶段，生成 stateroll / MoDA candidate pool。这个阶段的关键发现是：MoDA candidate pool 里确实有成功候选，oracle 不低，但 raw MoDA cost 不会稳定选 top1。

一开始我尝试做 planning-aware calibration，比如 BCE、rank-preserve、selector、detector。global AUC 看起来有提升，但 `moda_only_intra_episode_audit.py` 证明这个 AUC 很多来自 episode difficulty leakage。真正同一个 episode 内 success candidate 和 near-miss failure 的可分性很弱，所以 post-hoc reranking 不能稳定救 MoDA-only top1。

同时我也做了 PAC-MoDA / risk-controlled integration。`risk_controlled_moda_integration.py` 这条线能让系统级 top1 从 baseline 81 左右提升到 82/83，但它依赖 baseline fallback，所以只能作为 MoDA 的辅助价值证明，不能当 MoDA-only 主方法。

后面我尝试进入 planner 内部，比如 `moda_only_planner_in_loop_calibrated_cem.py` 把 calibrated utility 注入 CEM cost，`moda_only_action_sensitive_contrastive.py` 尝试拉开 success 和 near-miss failure embedding。但这些路线都只是小幅或负向诊断，没有稳定提升 MoDA-only top1。

最后目前最有价值的是 `moda_only_learned_residual_proposal.py`。这条线不再做 final rerank，而是做 candidate generation correction。具体来说，对于 raw rank0 失败但同 episode 有成功候选的样本，我学习 `delta_a = success_action - raw_rank0_failure_action`，然后在 evaluation 时生成 shifted candidates，再用 raw/calibrated cost 重新评估。

最后用 `moda_only_residual_confirm50_audit.py` 做 paired audit。medium20 曾经有 raw 60 到 residual 65 的结果，但 confirm50 后更稳的结论是 raw 46 到 residual 49/50，near-miss 14 降到 10/11。所以我不会说 MoDA-only 已经稳定 65+，但可以说 residual proposal 是目前唯一真正 MoDA-only 的正向方向，证明从 action proposal generation 层面修正 MoDA 是有希望的。

一句话总结：

MoDA 的优势是候选覆盖和 depth/state rollout，但 raw cost 与 planning success 不对齐。我们系统排除了单纯 AUC calibration 和 final rerank，最后发现 success-conditioned residual proposal 是当前最可信的 MoDA-only candidate-generation correction。

