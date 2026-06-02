# ARIS MoDA Diagnosis

## Scope

Goal: explain why the current exact-kernel MoDA line still fails to beat the clean `baseline64` line on PushT planning, even after fixing the earlier block-structure mismatch.

This note uses an ARIS-style loop:

1. collect only verified evidence
2. generate hypotheses
3. attack those hypotheses with the cheapest next experiments

## Verified Evidence

### Historical runs

- Historical MoDA result:
  - run: `pusht_moda64_kernel_clean`
  - checkpoint: `epoch 25`
  - eval config: `num_eval=50`, `num_samples=300`, `n_steps=30`, `topk=30`
  - planning: `88.0`

- Historical baseline result:
  - run: `pusht_real_3gpu_b42`
  - best known planning:
    - `epoch14=82.0`
    - `epoch30=70.0`
    - `epoch50=52.0`
    - `epoch70=48.0`
    - `epoch98=46.0`
    - `epoch99=52.0`

### Current clean runs on 5090

- Clean `baseline64`:
  - run: `pusht_baseline64_clean_5090_gpu1`
  - medium eval at `epoch17`:
    - `success_rate = 90.0`

- Earlier clean predictor-only MoDA:
  - run: `pusht_moda64_predictor_5090_gpu2`
  - medium eval at `epoch15`:
    - `success_rate = 85.0`

- Current exact-kernel MoDA:
  - run: `pusht_moda64_exact_5090_gpu2`
  - medium eval at `epoch12`:
    - `success_rate = 85.0`
  - heavy eval at `epoch12`:
    - `success_rate = 86.0`
  - medium eval at `epoch17`:
    - `success_rate = 80.0`
  - medium eval at `epoch19`:
    - `success_rate = 65.0`

## What Is Already Ruled Out

### 1. "The earlier failure was only because the module was wrong"

Ruled out.

Reason:

- The earlier `moda_module.py` line did have a real architectural mismatch:
  - it replaced LeWM's AdaLN-zero conditional block with a MoDA-style pre-norm block
- But after restoring a closer LeWM-style exact-kernel path in:
  - `train_moda_exact.py`
  - `moda_module_exact.py`
- Planning still does not beat `baseline64`

So the block mismatch was real, but not the only cause.

### 2. "The historical 88.0 was only due to heavier planning budget"

Ruled out.

Reason:

- exact-kernel `epoch12` medium: `85.0`
- exact-kernel `epoch12` heavy: `86.0`

Heavy budget helps a bit, but only by about `+1`.
That is not enough to explain the entire historical `88.0`.

## Ranked Hypotheses

### Hypothesis A

Historical `88.0` used a meaningfully different implementation from both:

- the earlier predictor-only `moda_module.py` line
- the current `moda_module_exact.py` line

Why this is plausible:

- The historical result lives on `6000ada`
- We have config evidence, but not yet a fully locked source snapshot for the exact code path that produced `88.0`
- A third implementation would immediately explain why neither current branch reproduces the old result cleanly

Priority: highest

### Hypothesis B

Depth cache semantics are unstable under LeWM's conditional predictor.

More concretely:

- each layer's K/V is produced after layer-specific AdaLN modulation
- those modulated K/V tensors are then cached and concatenated across layers
- the MoDA kernel may be seeing cross-layer memories that do not live in a consistent representation space

Why this is plausible:

- exact-kernel line improves only weakly at `epoch12`
- then degrades sharply by `epoch17` and `epoch19`
- this pattern is consistent with a mechanism that adds noisy cross-layer retrieval rather than useful depth memory

Priority: high

### Hypothesis C

Training hyperparameters or scheduler details differ in a way not yet accounted for.

Why this is weaker:

- The major visible items already align:
  - `batch_size=128`
  - `lr=5e-5`
  - `weight_decay=1e-3`
  - `predictor.depth=64`
  - `seed=3072`
- This can still matter, but currently looks less likely than A or B

Priority: medium

## Cheapest High-Value Experiments

### E1. Source-forensics on historical `88.0`

Question:

- Was historical `pusht_moda64_kernel_clean` produced by:
  - the current exact-kernel path
  - the earlier MoDA-style path
  - or a third path

Success criterion:

- recover the actual source snapshot or enough direct code evidence to classify the path unambiguously

Why first:

- This can kill or confirm Hypothesis A before wasting more GPU time

### E2. Exact-kernel with delayed depth cache

Change only:

- `depth_start_layer=2`, then maybe `4`

Keep fixed:

- checkpointing/eval config
- LeWM block structure
- planner settings

Question:

- Does pushing depth memory later reduce cross-layer noise and improve planning

Why second:

- This is the cheapest direct test of Hypothesis B

### E3. Exact-kernel depth ablation against control

Control:

- same exact-kernel code path, but effectively disable usable depth memory

Possible ways:

- `depth_start_layer` beyond active depth
- or an explicit no-depth flag if added cleanly

Question:

- Is the gain/loss coming from the MoDA cross-layer memory at all, or just from the surrounding predictor skeleton

### E4. Compare later exact checkpoints under heavy budget only if needed

Only after E1/E2.

Reason:

- Current exact trajectory already shows:
  - `epoch12 > epoch17 > epoch19`
- So blindly waiting longer is currently low-information

## Current Best Interpretation

The most defensible interpretation today is:

- `baseline64` is the strongest clean result we currently have
- exact-kernel MoDA is functioning, but its best tested checkpoint so far is only `85-86`
- historical `88.0` is real, but is not yet reproduced under the current exact path
- the two most serious explanations are:
  - historical code mismatch (Hypothesis A)
  - unstable depth-cache semantics under AdaLN-conditioned LeWM blocks (Hypothesis B)

## Immediate Recommendation

Do these in order:

1. recover or classify the historical `88.0` source path
2. run `depth_start_layer` ablations on the current exact-kernel branch
3. only then decide whether more training time is justified

## Claims We Can Say Now

- "The earlier predictor-only MoDA branch changed the predictor architecture materially and was not a clean LeWM-kernel-only comparison."
- "Fixing that mismatch alone did not recover a win over `baseline64`."
- "Heavier planning budget improves the exact-kernel branch only slightly at `epoch12` (`85 -> 86`), so budget alone does not explain historical `88.0`."
- "The strongest current explanation is either source-path mismatch for the historical run or harmful depth-cache behavior in the LeWM conditional block setting."

