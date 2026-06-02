# MoDA 原始思路 vs 当前 DADP 实现

## 一句话结论

当前代码 **不是完整的 MoDA 复现**，而是：

> **受 MoDA 启发的、面向 world model predictor 的 depth-augmented 改写版**

如果对外口径是“完全复现 MoDA”，那现在这版不严谨。  
如果口径是“借鉴 MoDA 的跨层检索思想，适配到 LeWM predictor”，那是成立的。

---

## 对照表

| 维度 | 原始 MoDA 思路 | 当前 DADP 实现 | 结论 |
|---|---|---|---|
| 核心目标 | 解决深层网络中的跨层信息访问问题 | 解决 world model predictor 中的 information dilution | **问题不同，灵感一致** |
| 主要应用对象 | 更偏通用 Transformer / LLM 风格场景 | LeWM 的 latent predictor | **是适配，不是原封搬运** |
| 跨层检索 | 有，核心思想之一 | 有，`depth_k_cache / depth_v_cache` | **保留了核心思想** |
| 统一 softmax | 是 MoDA 很关键的一点 | `unified` 模式中实现了 | **保留了最关键机制** |
| 深度证据来源 | 跨层 K/V 检索 | 浅层同位置 K/V cache | **思路一致，具体场景化了** |
| 时序证据 | 原始 attention 路径 | 本层 causal sequence attention | **保留** |
| 时序+深度联合竞争 | 是 | `scores_seq` 与 `scores_depth` 拼接后统一 softmax | **是当前最像 MoDA 的部分** |
| group-aware / chunk-aware 机制 | 原味 MoDA 常有这类设计或高效实现 | 当前没有 | **缺失** |
| Triton / 自定义高效 kernel | 原味实现通常是重点 | 当前没有 | **缺失** |
| FlashAttention 兼容 | 原味路线通常很关注效率 | 当前 `unified` 路径会退回手写 score fusion | **缺失/退化** |
| 长序列优化 | 原味 MoDA 更有意义 | 当前任务 `history_size=3`，序列极短 | **场景不一样** |
| Depth 访问范围 | 可能更一般、更复杂 | 通过 `depth_start_layer` 和 `max_depth_layers` 控制 | **做了简化且更可消融** |
| 控制组 | 不一定是重点 | `none` 是很干净的 matched control | **这是 DADP 的实验优点** |
| 多融合消融 | 原始 MoDA 不一定提供这一整套 | `none/unified/additive/gated/residual` | **这是当前实现的额外研究价值** |
| 参数位置 | 原味实现服务原任务 | 当前只改 predictor，不改 encoder/loss/planner | **更利于归因分析** |
| 论文口径 | “MoDA 方法本身” | “MoDA-inspired DADP for world models” | **必须分清** |

---

## 当前代码里，最像 MoDA 的部分

这几处是当前实现里最应该保留、也最能对外说“受 MoDA 启发”的地方：

1. `DepthAugmentedAttention.forward`
   - 文件：[moda_module.py](/home/internship/wm_transfer_lab/LeWM_src/le-wm-main/moda_module.py#L64)
   - 作用：保留本层 `q/k/v`，同时引入浅层 `depth_k_cache / depth_v_cache`

2. `DepthAugmentedAttention._fuse`
   - 文件：[moda_module.py](/home/internship/wm_transfer_lab/LeWM_src/le-wm-main/moda_module.py#L105)
   - 作用：同时计算 `scores_seq` 与 `scores_depth`

3. `unified` 模式
   - 文件：[moda_module.py](/home/internship/wm_transfer_lab/LeWM_src/le-wm-main/moda_module.py#L119)
   - 作用：把时序证据和深度证据放在同一个 softmax 里竞争

这部分是当前实现真正“有 MoDA 味道”的地方。

---

## 当前代码里，不足以叫“完整复现”的部分

如果你老师和师兄的预期是：

> “把 MoDA 原方法完整复现，再嫁接到 LeWM”

那当前这版还差下面这些关键点：

### 1. 没有原味高效实现

当前 `unified` 路径是手写：

- `scores_seq = q @ k^T`
- `scores_depth = q @ dk`
- `cat`
- `softmax`

这意味着：

- 不是原味的高效 kernel 路线
- 也不是完整的 FlashAttention 兼容实现

### 2. 没有 group/chunk 级别设计

如果原始 MoDA 里强调：

- group-aware
- chunk-aware
- 长序列高效检索

那当前 LeWM 版本没有这些。

### 3. 没有“原方法数值一致性”证明

如果要叫完整复现，通常至少要有：

- naive reference
- optimized implementation
- forward/backward 数值一致性测试

当前这版没有做到这一步。

### 4. 任务场景不同

原味 MoDA 更适合在：

- 更深网络
- 更长序列
- 更重 attention 计算

的环境里体现优势。

当前 LeWM PushT 设置是：

- `history_size = 3`
- predictor `depth = 6`
- `max_depth_layers = 2`

所以它更像一个 **针对短序列 world model 的研究适配版**。

---

## 为什么当前 DADP 仍然有研究价值

虽然它不是完整 MoDA 复现，但它不是“没价值的残次版”。

它的价值在于：

1. 它保留了最关键的跨层检索思想；
2. 它把这个思想变成了 world model predictor 的研究问题；
3. 它内置了更干净的控制组与消融空间；
4. 它只改 predictor，便于做因果归因。

所以它更像：

> 一个受 MoDA 启发、但服务于 world model 机制研究的问题驱动实现。

---

## 如果老师和师兄想要“完整复现”，下一步该补什么

如果目标改成：

> **先完整复现 MoDA，再讨论如何接到 LeWM**

那建议补下面这些内容：

### A. 方法一致性

- 对照原始 MoDA 仓库，把 attention 路径逐项对齐
- 明确哪些 projection、哪些 cache 逻辑、哪些归一化步骤与原版一致

### B. 工程一致性

- 补原版 kernel / Triton 路线
- 补 group-aware / chunk-aware 逻辑
- 补长序列下的效率实现

### C. 正确性验证

- 写 naive reference
- 写 optimized version
- 做 forward/backward 数值对齐测试

### D. 原任务复现

- 先在原始任务或原论文场景上拿到相近结果
- 再迁移到 LeWM

只有做到这些，才更接近“完整复现”。

---

## 推荐对外口径

### 如果当前版本继续推进

建议说：

> We are developing a MoDA-inspired depth-augmented predictor for planning-oriented world models.

不要说：

> We fully reproduced MoDA inside LeWM.

### 如果之后补齐原版实现

那时才更适合说：

> We first reproduced MoDA, then adapted it to world model predictors.

---

## 最终判断

### 现在这版可以怎么定义

- **不是完整 MoDA 复现**
- **是 MoDA-inspired DADP**
- **是一个有研究价值的 world-model 适配版**

### 现在这版不应该怎么定义

- 不应说成“完整复现 MoDA”
- 不应说成“原汁原味 MoDA 已经接进 LeWM”

如果你后面要跟老师和师兄对口径，我建议最稳的一句是：

> 当前版本已经保留了 MoDA 的跨层统一竞争思想，但工程与方法层面仍属于 world-model 场景下的改写版，而不是原始 MoDA 的完整复现。
