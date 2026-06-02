# 01. MoDA 结构接入代码

对应文件：

```text
code/moda_module_exact.py
code/moda_module.py
```

## 这部分代码解决什么问题

LeWM 原本是 latent world model + CEM planning。它先把当前 observation 编码成 latent state，再用 predictor rollout 未来 latent，最后用 cost 选择 action。我们引入 MoDA 的第一步，就是把原来的 predictor attention 替换成带 depth retrieval 能力的 MoDA attention。

这部分代码的核心目标不是直接改 planner，而是先改 world model 的 representation / rollout 机制。也就是说，先让 LeWM 的 predictor 能使用 MoDA 的跨 depth 信息，再看 planning candidate pool 是否变好。

## `moda_module_exact.py`

这是最重要的结构实现。它的设计原则是：尽量保留 LeWM 原来的 predictor 结构，只替换 attention kernel。

文件顶部会尝试从这些位置导入 MoDA Triton kernel：

```text
MODA_TRITON_ROOT
/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton
/home/internship/wm_transfer_lab/MoDA/libs/moda_triton
```

这样做是为了兼容不同机器环境：本机、5090 服务器、实习服务器都能找到 MoDA kernel。

## `MoDAAttentionExact`

这个类是 MoDA attention 的核心。

它先像普通 transformer attention 一样，把输入 `x` 做 LayerNorm，再线性投影成 `q/k/v`。区别在于 forward 里有两条路径：

第一条路径：如果没有 `cached_k/cached_v`，就退回普通 causal attention。这保证模型在没有 depth cache 的时候仍然能正常跑，行为接近原始 LeWM predictor。

第二条路径：如果传入了 `cached_k/cached_v`，就调用 `parallel_moda` 或 `parallel_moda_chunk_visible`。这一步就是 MoDA 的核心：当前层的 query 不只看当前层自己的 key/value，还能访问前面层累积下来的 depth cache。

汇报时可以这样说：

> 这个模块保留了 LeWM 原始 attention layout，但在有 depth cache 时切换到 official MoDA Triton kernel，从而让 predictor 支持跨层 depth retrieval。

## `MoDAConditionalBlockExact`

这个类对应 LeWM 原来的 conditional transformer block。

它保留了 AdaLN-zero conditioning，也就是通过 condition `c` 生成：

```text
shift_msa, scale_msa, gate_msa,
shift_mlp, scale_mlp, gate_mlp
```

然后分别调制 attention 和 MLP 分支。`gate_msa` 和 `gate_mlp` 控制残差分支强度。这个设计让 MoDA block 可以接到 LeWM 原来的 conditioning 机制里，而不是重写整个 predictor。

汇报时重点：

> 我没有把 LeWM predictor 整个推倒重写，而是保留 conditional block / AdaLN-zero，只把 attention 子模块换成 MoDA attention。

## `MoDATransformerExact`

这个类把多个 `MoDAConditionalBlockExact` 堆起来。

它的关键逻辑是 `_build_depth_cache`。每一层 forward 后会产生当前层的 `cur_k/cur_v`，后续层可以把前面层的 K/V stack 起来形成 depth cache。

`cache_window` 控制最多使用最近多少层的 cache；`detach_cache` 控制 cache 是否断梯度。这个设计用于控制显存和训练稳定性。

汇报时可以这样概括：

> MoDATransformerExact 负责把每层 attention 的 K/V 收集成 depth cache，并在后续层调用 MoDA kernel。这样 predictor 不再只是固定深度的 sequential transformer，而是可以跨 depth 检索历史层信息。

## `moda_module.py`

这个文件是较早或 fallback 版本的 MoDA 实现。它的作用是保留 MoDA 接入 LeWM 的非 exact-kernel 路径。汇报时不用逐段展开，只需要说明：

> `moda_module.py` 是 MoDA 接入的通用实现，`moda_module_exact.py` 是后面更贴近 official Triton kernel 的强实现。

