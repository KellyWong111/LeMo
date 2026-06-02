# 02. 训练入口代码

对应文件：

```text
code/train_encoder_moda.py
code/train_moda.py
code/train_moda_exact.py
```

## 这部分代码解决什么问题

MoDA module 写好以后，还需要接入 LeWM 的训练流程。这些训练入口负责读取 config、构造数据集、初始化模型、加载 MoDA predictor/encoder，然后执行训练。

这部分属于项目最早阶段：验证 MoDA 结构能不能作为 LeWM 的 encoder/predictor 组件正常训练。

## `train_encoder_moda.py`

这是把 MoDA 接入 encoder/predictor 训练的主要入口之一。

它的作用是：

1. 读取 LeWM 训练配置；
2. 构造 PushT / world model 数据；
3. 初始化带 MoDA 的 encoder 或 predictor；
4. 启动训练；
5. 保存 checkpoint。

汇报时重点不是每个训练参数，而是它在项目里的位置：

> 这一步完成了 MoDA 到 LeWM training pipeline 的结构接入，使得后续 planning evaluation 可以使用 MoDA checkpoint。

## `train_moda.py`

这是较早的 MoDA 训练入口。它主要用于验证 MoDA module 是否能在 LeWM 框架下跑通。

它对应的是 early prototype 阶段：先确认 MoDA 结构能训练、能产出 checkpoint，再进入 planning diagnostics。

## `train_moda_exact.py`

这个入口对应 exact MoDA kernel 路径，也就是更接近 official MoDA Triton kernel 的训练版本。

它和 `moda_module_exact.py` 对应，用于训练 exact-kernel MoDA predictor。

汇报时可以这样讲：

> 训练入口分为早期 MoDA prototype 和 exact-kernel MoDA 两类。早期入口用于快速验证结构可训练，exact 入口用于更严谨地复现 official MoDA kernel 路径。

## 这部分和后续实验的关系

这些训练脚本本身不是最终方法贡献，但它们是整个项目的基础。后面所有 candidate pool、stateroll、PAC-MoDA、residual proposal 都依赖前面训练出来的 MoDA/LeWM checkpoint。

