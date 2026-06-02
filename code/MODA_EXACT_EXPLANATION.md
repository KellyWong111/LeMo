# Exact MoDA on LeWM: What It Does and How It Differs from Original LeWM

This note explains the current `exact MoDA` branch in `LeWM_src/le-wm-main` in plain engineering terms.

It is written to answer two questions:

1. What is the original LeWM predictor doing?
2. What exactly changed after replacing it with MoDA?

## 1. Original LeWM in one sentence

Original LeWM is a JEPA world model trained from pixels.

The predictor takes:
- latent image embeddings from the history frames
- action embeddings as conditioning

and predicts the next latent embedding.

The official training entrypoint is:
- `train.py`

The original predictor class is:
- `module.py -> ARPredictor`

The official README says the standard training command is:

```bash
python train.py data=pusht
```

## 2. Original LeWM predictor: the core idea

The original predictor is a standard causal transformer with AdaLN-zero conditioning.

Concretely:

- `ARPredictor` adds positional embeddings to the latent sequence.
- It passes the sequence through `Transformer(..., block_class=ConditionalBlock)`.
- Each `ConditionalBlock` does:
  - causal self-attention over the token sequence
  - MLP update
  - AdaLN-zero modulation using the action-conditioned signal `c`

Important point:

- In original LeWM, each layer only attends over the sequence dimension.
- A token at time `t` can look backward in time.
- It cannot explicitly look back into earlier predictor layers at the same position.

So the information path is:

```text
history tokens -> causal transformer layers -> next latent prediction
```

There is no explicit cross-layer memory.

## 3. What "exact MoDA" adds

MoDA stands for Mixture-of-Depth Attention.

The key change is:

- deeper predictor layers do not only attend over previous tokens in time
- they also attend over cached representations from earlier layers at the same token position

So attention is no longer only "time-wise".
It becomes a mixture of:

- temporal attention
- depth attention

This is why the file is called `moda_module.py`.

## 4. The exact attention change

In the current exact-MoDA implementation, each predictor layer computes:

- normal causal attention scores over the current sequence
- depth attention scores over cached K/V from previous layers

Then these two score groups are concatenated and normalized with one single softmax.

That is the core MoDA idea.

In symbols:

```text
softmax([sequence_logits, depth_logits])
```

This matters because:

- it makes temporal evidence and depth evidence compete in one shared probability distribution
- the model can decide whether the current token should rely more on past time steps or on shallower-layer features at the same time step

This is different from:

- additive fusion
- gated fusion
- residual fusion

Those alternatives combine branches after separate processing.
MoDA instead mixes them inside the attention normalization itself.

## 5. Why this is called "faithful" or "exact"

The current version was rewritten to match the original MoDA mechanism, not the earlier DADP-style experiments.

The main faithfulness points are:

### 5.1 Unified softmax

The implementation uses one unified softmax over temporal and depth logits.

This is the central MoDA behavior.

### 5.2 Depth cache is flattened in MoDA format

Cached keys/values are stored as:

```text
[B, T * L, H, D]
```

where:

- `B` = batch
- `T` = sequence length
- `L` = number of cached previous layers
- `H` = attention heads
- `D` = head dimension

And the flattening order is position-major:

```text
[pos0_layer0, pos0_layer1, ..., pos1_layer0, pos1_layer1, ...]
```

This matches how MoDA expects depth memory to be laid out.

### 5.3 Full previous-layer cache

Once depth attention starts, each deeper layer can see all previous predictor layers, not just a tiny truncated subset.

So if the predictor depth is 64:

- layer 1 can see layer 0
- layer 2 can see layers 0 and 1
- ...
- layer 63 can see layers 0..62

### 5.4 LeWM conditioning is preserved

The LeWM-specific AdaLN-zero conditioning structure is still kept.

So this is not "replace LeWM with another model".
It is:

- keep LeWM's overall JEPA training recipe
- keep LeWM's action-conditioned predictor block structure
- replace the predictor attention mechanism with MoDA-style depth-augmented attention

## 6. What changed in code

### Original path

- `train.py`
- `module.py`
- predictor class: `ARPredictor`

### Current exact-MoDA path

- `train_moda.py`
- `moda_module.py`
- predictor class: `MoDAARPredictor`

And `train_moda.py` is now intentionally locked to MoDA-only runs:

- if someone tries `use_moda=false`, it raises an error
- that prevents accidental fallback to the old control path inside this entrypoint

## 7. Current default config

The current config is set to:

- `use_moda: true`
- `depth_start_layer: 1`
- `predictor.depth: 64`

So the model being trained now is:

- a 64-layer LeWM predictor
- with MoDA depth attention enabled from layer 1 onward

## 8. The easiest mental model

Think of original LeWM as:

```text
At each layer, each token only asks:
"Which earlier time steps should I attend to?"
```

Think of exact MoDA LeWM as:

```text
At each deeper layer, each token asks:
"Should I attend to earlier time steps,
or should I reuse useful representations from earlier layers
at this same position?"
```

So MoDA gives the predictor an extra axis of information routing:

- original LeWM: route across time
- MoDA LeWM: route across time and depth

## 9. Why this could help

The intuition is:

- in a deep predictor, useful information may get diluted or transformed across layers
- earlier layers may retain geometric or local structure that later layers partially wash out
- depth attention gives later layers a direct path back to those earlier representations

This is the hypothesis being tested.

## 10. What this is not

The current branch is not:

- sparse attention ablation
- additive/gated/residual post-fusion
- tiny 2-layer depth cache trick
- the earlier DADP-style experiment family

Those older paths were retired.

## 11. Short comparison table

| Aspect | Original LeWM | Current exact MoDA LeWM |
|---|---|---|
| Training entry | `train.py` | `train_moda.py` |
| Predictor | `ARPredictor` | `MoDAARPredictor` |
| Attention over time | Yes | Yes |
| Attention over previous layers | No | Yes |
| Unified temporal+depth softmax | No | Yes |
| AdaLN-zero conditioning | Yes | Yes |
| Current default predictor depth | 6 in old runs | 64 now |

## 12. File map

Current files to read:

- `module.py`
- `train.py`
- `moda_module.py`
- `train_moda.py`
- `config/train/lewm.yaml`
- `launch_moda64.sh`

## 13. Bottom line

Original LeWM predicts the next latent using a causal transformer over time.

Current exact-MoDA LeWM keeps the LeWM training recipe and conditioning structure, but changes the predictor so that deeper layers can jointly attend to:

- past tokens in time
- previous predictor layers at the same position

through one unified attention distribution.

That is the essential conceptual difference.
