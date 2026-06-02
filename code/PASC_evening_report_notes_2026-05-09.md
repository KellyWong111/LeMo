# PASC 晚汇报讲稿（2026-05-09）

## 1. 开场结论
我现在把问题重新收敛了一下：不是某个 subspace、ranking loss 或 reranker 没调好，而是 prediction representation 和 planning representation 没有被显式对齐。

gate07 没有在 prediction loss 上训练崩。val pred_loss 从 ep2 的 0.00446 降到 ep16 的 0.00238。但是 planning success 不同步提升：ep2/ep3 是 75%，ep9 掉到 40%，ep17 也是 45%。所以更准确的说法不是“训练崩了”，而是 prediction objective 继续变好，但 planning-facing behavior 漂移。

## 2. 为什么不是单纯 subspace/ranking 问题
我们之前做 subspace、spread、ranking，是假设问题主要是候选动作在 latent 里分不开，导致 CEM 选错。

但 top-k oracle 诊断显示不是这么简单。gate07 ep4 下：
- s300/top30: top1 50%，oracle 65%，gap +15；但仍有 7/20 no-hit。
- s600/top60: top1 60%，oracle 75%，gap +15；仍有 5/20 no-hit。
- s1000/top100: top1 60%，oracle 75%，gap +15；仍有 5/20 no-hit。

这说明有一部分确实是 reranking 能救的 misrank case，但很多失败是 top-k 里面根本没有成功候选。subspace/reranker 只能重排已有候选，不能解决 no-hit。

## 3. CEM 参数不是简单调参
CEM sweep 说明 search budget 和 checkpoint 的 cost landscape 有交互：
- gate07 ep2: default 75%，strong CEM 65%。
- gate07 ep3: default 75%，strong CEM 60%。
- gate07 ep4: default 65%，strong CEM seed42 80%，但 seed43/44 都是 55%。
- gate07 ep16: default 65%，strong CEM 55%。
- pred6 ep3: default 65%，strong CEM 75%。

所以不是 samples 越大越好。如果 checkpoint 的 planning cost landscape 是友好的，strong CEM 能找到更好动作；如果 landscape 有偏，strong CEM 会更认真地优化错误目标。

## 4. 当前方法主线：PASC
我想把后续方法定义成 Planning-Aware Search Calibration，中文叫面向规划的搜索校准。

核心不是“调 CEM 参数”，而是：不用 prediction loss 单独选择模型，也不对所有 checkpoint 固定同一套 CEM，而是用 planning diagnostics 联合选择 checkpoint 和 search regime。

诊断指标包括：
- val pred_loss
- default / strong CEM success
- top-k oracle success 和 oracle gap
- candidate trajectory spread / top-k spread
- top2 margin / cost std

可以定义一个 planning-friendly score：
S_plan = alpha * OracleSuccess + beta * TopKSpread + gamma * CostStd - eta * OracleGap

然后用 S_plan 选择 checkpoint，再为这个 checkpoint 选择匹配的 CEM regime。

## 5. 如果师兄问：representation 和 CEM 不应该连着吗？
答：应该连着，这正是现在的问题。当前训练 loss 只保证 prediction representation 好，但没有显式保证 CEM rollout 里的 planning representation 好。实验上 pred_loss 降低但 planning success 漂移，说明这个连接目前是隐式且不稳定的。PASC 的目的就是把这个连接显式量化，用 planning diagnostics 来选 checkpoint 和 search regime。

## 6. 如果师兄问：是不是训练崩了？
答：如果说 pred_loss 训练崩，目前没有证据，val pred_loss 是下降的。但如果说 planning behavior 崩/漂移，是有证据的。更准确说是 prediction training 没崩，planning-facing representation/cost landscape 漂移。

## 7. 如果师兄问：subspace 还做不做？
答：subspace 不是完全错，但它现在不是主解。它适合解决 top-k 里有成功候选但 CEM 排错的 case。top-k oracle 显示还有大量 no-hit case，所以必须先解决 candidate generation / search calibration / rollout reliability。后面如果 oracle gap 很大，再回到 reranker/subspace。

## 8. 下一步实验
1. 补齐 ep2/3/4/9/16/17 的 planning diagnostics 表：pred_loss、default success、strong success、oracle、spread、margin、cost std。
2. 看这些 planning diagnostics 和 success 的相关性，确定 S_plan。
3. 做 checkpoint × CEM grid，不再默认所有 checkpoint 用同一个 CEM。
4. 对最优组合做 multi-seed，避免只报 seed42。
