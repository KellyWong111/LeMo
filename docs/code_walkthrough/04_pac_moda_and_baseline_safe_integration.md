# 04. PAC-MoDA 和 baseline-safe integration

对应文件：

```text
experiments/pac_moda_native_calibration_report.py
experiments/risk_controlled_moda_integration.py
experiments/pac_moda_v2_selector_v3_detector_gate.py
experiments/pac_moda_v2_full_experiments_n100_corrected.py
```

## 这部分代码解决什么问题

MoDA-only 不稳定以后，我们也做过 system-level integration：baseline planner 保底，MoDA 作为辅助候选源，只在高置信机会场景介入。

这部分结果能提升系统 top1，但必须讲清楚：它不是 MoDA-only。

## `pac_moda_native_calibration_report.py`

这个脚本用于整理 MoDA-native calibration report。

它的关键作用是把两类指标分开：

第一类：global MoDA candidate utility ranking。  
这里只能比较 raw stateroll cost、BCE calibrated utility、rank-preserve utility。

第二类：localized activation regime。  
这里看 detector gate、selector-v3、localized raw cost 是否能在某些机会区域里安全使用。

为什么要分开？

因为 localized raw cost 不是提升了 global raw-cost ranking，而是改变了 raw cost 的适用区域。这个区分能避免报告逻辑被质疑。

## `risk_controlled_moda_integration.py`

这个脚本代表 baseline-safe integration。

它的逻辑是：

1. baseline planner 先给出安全默认 action；
2. MoDA/stateroll 提供候选；
3. gate 判断当前 episode 是否适合切到 MoDA；
4. 如果风险低，就使用 MoDA；
5. 否则保持 baseline。

指标包括：

```text
baseline top1
gated integration top1
fixed
harmed
net
selected MoDA rate
```

这条线最好的地方是系统表现能涨，比如 81 到 82/83 左右。但它依赖 baseline fallback，所以汇报时要定位为：

> MoDA auxiliary candidate source with risk-controlled integration.

而不是：

> MoDA-only planner solved.

## `pac_moda_v2_selector_v3_detector_gate.py`

这个脚本属于 selector-v3 / detector gate 路线。

它用 detector 找 baseline 可能失败且 MoDA 有机会的区域，再局部使用 MoDA score。

这条线解释了为什么 bsl-relative 结果能涨：不是 MoDA 全局接管，而是在局部 opportunity regime 里补 baseline 的失败。

## `pac_moda_v2_full_experiments_n100_corrected.py`

这是更完整的 n100 corrected experiment 脚本，用于汇总多 seed、多 threshold、多 ablation。

它主要服务于论文/报告中的 system-integration 结果，不是 MoDA-only 主线。

## 这部分最终结论

这部分可以证明：

MoDA 有辅助价值。  
MoDA candidate 和 baseline 有互补性。  
risk-controlled integration 能带来系统级提升。

但不能证明：

MoDA 自己已经成为稳定 standalone planner。

