# DADP Signal Experiments

This project track is framed around a world-model-specific problem:

- Deep latent predictors suffer **information dilution**
- Planning needs **continued access to shallow geometry/contact evidence**
- The method is a **Depth-Augmented Dynamics Predictor (DADP)**
- MoDA is treated as an inspiration for cross-layer retrieval, not the paper's core claim

## Minimal Matrix

The first goal is not SOTA. It is to test whether the new predictor family shows
an early planning signal under matched training conditions.

1. `none`:
   control predictor with no depth memory
2. `unified`:
   full method, one softmax over sequence + depth evidence

Matched settings:

- Task: `pusht`
- GPU: single GPU
- Predictor depth: repo default (`cfg.predictor.depth`)
- Epochs: `5` by default for fast signal
- Batch size: `42`
- Seed: `0`
- Depth retrieval start: `1`
- Max depth layers: `2`

## Why This Matrix

This is the fastest way to answer the only question that matters right now:

> Does direct access to shallow same-timestep evidence improve early world-model
> learning relative to a matched control predictor?

If the answer is yes, the next tier of experiments is:

1. `additive`
2. `gated`
3. `residual`
4. predictor depth ablation: `4 / 6 / 8 / 12`

## Decision Rule

Promote DADP to the main method track only if at least one of the following is
visible against the `none` control:

1. better early `pred_loss`
2. better early planning success after checkpoint eval
3. more stable scaling when predictor depth is increased

If the early signal is weak, do not spend 100 epochs on this variant.
