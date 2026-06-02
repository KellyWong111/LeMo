# LeWM 上的 Exact MoDA：现在这版到底在做什么

这份文档专门解释当前 `LeWM_src/le-wm-main` 里这条 `exact MoDA` 线的原理。

目标只有两个：

1. 原版 LeWM 的 predictor 到底在做什么？
2. 现在这版加了 MoDA 之后，和原版的本质区别是什么？

## 1. 一句话先说清楚

原版 LeWM 是一个从像素端到端训练的 JEPA world model。

它的 predictor 做的事情是：

- 输入历史帧的 latent embedding
- 再输入动作 embedding 作为条件
- 用一个带因果掩码的 transformer 去预测下一步 latent embedding

现在这版 exact MoDA 做的事情是：

- 保留 LeWM 原本的训练框架、损失、AdaLN-zero 条件结构
- 但是把 predictor 里的注意力机制，改成了 MoDA 的“时间 attention + 深度 attention 的统一 softmax”

## 2. 原版 LeWM 的 predictor 是什么

原版训练入口是：

- `train.py`

原版 predictor 类是：

- `module.py -> ARPredictor`

原版 `ARPredictor` 的核心结构非常直接：

1. 给输入 latent 序列加位置编码
2. 丢进一个 `Transformer(..., block_class=ConditionalBlock)`
3. 每一层 `ConditionalBlock` 都做：
   - 时间维度上的 causal self-attention
   - MLP
   - 用动作条件 `c` 做 AdaLN-zero 调制

也就是说，原版 LeWM 的 predictor 本质上还是一个“只沿着时间维度看过去”的 transformer。

### 原版 LeWM 的信息流

你可以把它想成：

```text
历史 token -> 多层因果 transformer -> 预测下一步 latent
```

这里每个 token 在每一层都会问一个问题：

```text
我现在应该看哪些更早的时间步？
```

但是它不会显式问另一个问题：

```text
我需不需要回头看一下更浅层同一个位置的表示？
```

这就是原版 LeWM 的限制点：  
它只有“时间维”的注意力，没有“深度维”的注意力。

## 3. MoDA 加进去之后，多了什么

MoDA 的全称是 Mixture-of-Depth Attention。

它引入的核心思想是：

- 深层 predictor layer 不仅可以看过去的 token
- 还可以看更浅层 predictor layer 在“同一个时间位置”上的表示

所以注意力不再只是：

- 跨时间看历史

而是变成同时考虑两种来源：

- 时间上的历史信息
- 深度上的浅层表示

这就是 MoDA 的关键变化。

## 4. 现在这版 exact MoDA 的核心原理

当前实现里，每一层 MoDA attention 都会算两组 logits：

### 4.1 时间 attention logits

这部分和普通 causal attention 一样：

- 当前 query 去和当前层的 key 做点积
- 只允许看当前及以前的时间步

也就是标准的 sequence causal attention。

### 4.2 深度 attention logits

这部分是新增的：

- 当前 query 去和之前所有层缓存下来的 key 做点积
- 但只看“同一个时间位置”的跨层缓存

也就是说，如果当前在时间位置 `t`：

- 不只是看 `t` 之前的 token
- 还会看过去各层在 `t` 这个位置留下来的表示

### 4.3 最重要的一步：统一 softmax

当前 exact MoDA 不是先分别算两条分支再融合。

它做的是：

```text
softmax([sequence_logits, depth_logits])
```

也就是说：

- 时间证据和深度证据先拼在一起
- 然后共享同一个 softmax

这意味着两类信息是在一个概率分布里直接竞争的。

模型会自己决定：

- 这次更该相信时间上的过去
- 还是更该相信浅层的同位置信息

这就是 MoDA 最核心的地方。

## 5. 为什么这版叫 exact MoDA

因为它不是之前那种 DADP 式的实验性 fusion 了，而是尽量按原始 MoDA 机制来还原。

现在这版“忠实还原”主要体现在 4 点：

### 5.1 用的是 unified softmax

不是 additive
不是 gated
不是 residual
不是两支算完再加

而是时间 logits 和深度 logits 一起进同一个 softmax。

### 5.2 depth cache 的布局按 MoDA 原始格式来

当前缓存格式是：

```text
[B, T * L, H, D]
```

其中：

- `B`：batch size
- `T`：序列长度
- `L`：缓存的历史层数
- `H`：head 数
- `D`：每个 head 的维度

而且排列顺序是 position-major：

```text
[pos0_layer0, pos0_layer1, ..., pos1_layer0, pos1_layer1, ...]
```

这个顺序是为了和 MoDA 的预期一致。

### 5.3 用的是“全前层缓存”

现在不是只看最近 2 层、4 层这种小窗口。

而是：

- 第 1 层之后开始有 depth attention
- 更深的层能看到所有之前层的缓存

如果 predictor 深度是 64：

- layer 1 看 layer 0
- layer 2 看 layer 0,1
- ...
- layer 63 看 layer 0 到 62

### 5.4 LeWM 本身的条件结构保留了

这点非常重要：

这不是把 LeWM 整体换掉。

保留下来的东西包括：

- JEPA 训练方式
- LeWM 的 encoder / projector / loss
- AdaLN-zero 条件块
- action conditioning 方式

真正换掉的核心只是 predictor 里的 attention 机制。

所以这版更准确地说是：

```text
LeWM 框架 + MoDA predictor attention
```

而不是一个完全不同的新模型。

## 6. 和原版 LeWM 最本质的区别

最本质区别不是“参数更多了”或者“层数更深了”。

最本质区别是：

### 原版 LeWM

每层每个 token 只会问：

```text
我该看哪些更早的时间步？
```

### 现在的 exact MoDA LeWM

每层每个 token 会问：

```text
我现在应该：
1. 看时间上的过去？
还是
2. 看更浅层同位置留下的表示？
```

所以可以这样理解：

- 原版 LeWM 只有“时间维”的信息路由
- MoDA 版 LeWM 变成“时间维 + 深度维”的联合路由

## 7. 为什么这种改动可能有用

背后的直觉是：

- predictor 一旦很深，信息会在层层变换里被稀释
- 某些早层的局部几何结构、局部对齐信息，可能在深层不容易完整保留
- 如果深层只能依赖自己当前这层的时间 attention，它可能拿不到那些更“干净”的浅层信息

MoDA 的作用就是给深层一个额外通道：

- 允许它直接回看浅层在同一个位置上的表示

所以它试图解决的问题，本质上是：

```text
深层预测器中的 information dilution / information washing-out
```

## 8. 代码上现在对应哪些文件

### 原版 LeWM

- `train.py`
- `module.py`
- `ARPredictor`

### 现在的 exact MoDA 版

- `train_moda.py`
- `moda_module.py`
- `MoDAARPredictor`
- `config/train/lewm.yaml`
- `launch_moda64.sh`

## 9. 现在这版默认配置是什么

当前默认已经切成：

- `use_moda: true`
- `depth_start_layer: 1`
- `predictor.depth: 64`

也就是说，现在正式在跑的是：

- 一个 64 层的 LeWM predictor
- 从第 1 层之后开始使用 MoDA depth attention

## 10. 它不是什么

当前这版不是：

- 稀疏 attention 实验
- additive/gated/residual 的 fusion 消融
- 只缓存 2 层的小技巧
- 之前那批 DADP 风格的实验变体

这些都已经不是当前主线了。

## 11. 最容易记住的理解方式

你可以把两者差异记成一句话：

### 原版 LeWM

```text
只沿时间看。
```

### 现在 exact MoDA LeWM

```text
同时沿时间和深度看，
而且两者在一个 softmax 里竞争。
```

这就是它和原版最大的概念差异。

## 12. 最后一句总结

原版 LeWM 是一个“时间因果 transformer predictor”。

现在这版 exact MoDA LeWM 则是在不改变 LeWM 整体训练框架的前提下，把 predictor 改成了：

- 能看过去时间步
- 也能看之前层同位置表示
- 并且用统一 softmax 在两类证据之间做选择

这就是当前这版模型的核心原理，也是它和原版 LeWM 的本质区别。
