# LeWM + MoDA 代码模块导读

这份目录是汇报用代码讲解，不改原始 `.py`。讲法按项目技术路径组织，而不是逐行解释。

建议阅读顺序：

1. `01_moda_structure.md`  
   讲 MoDA 怎么接入 LeWM predictor / encoder。

2. `02_training_entries.md`  
   讲训练入口如何把 MoDA 配置、模型和数据接起来。

3. `03_moda_only_diagnostics.md`  
   讲为什么 AUC calibration / post-hoc reranking 不够。

4. `04_pac_moda_and_baseline_safe_integration.md`  
   讲 bsl-relative integration 为什么有效，但不能当 MoDA-only 主线。

5. `05_residual_proposal.md`  
   讲当前最重要的 MoDA-only 方向：success-conditioned residual proposal。

6. `06_presentation_script.md`  
   汇报时可以照着讲的简短版本。

