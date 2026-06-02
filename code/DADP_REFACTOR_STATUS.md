# DADP Refactor Status

This note summarizes the current refactor from a low-value "MoDA plugged into
LeWM" variant into a problem-driven method track for world models.

## 1. What Was Changed

The project direction was changed from:

- "replace LeWM predictor with a MoDA-like block and see if it trains"

to:

- "study and mitigate information dilution in deep world model predictors"

Concretely, I did four things:

1. Reframed the method:
   the core object is now a **Depth-Augmented Dynamics Predictor (DADP)**,
   rather than a one-off "MoDA + LeWM" combination.
2. Refactored the predictor code:
   the code now supports a **family of depth-aware predictor variants** for
   controlled ablation, instead of one hard-coded MoDA swap.
3. Stabilized the training entrypoint:
   the training script now supports quick smoke tests with `num_workers=0`,
   so we can validate new variants in minutes instead of committing to
   multi-hour runs immediately.
4. Started the new experiment track:
   rather than spending 20 hours on a weak story, the server is now running
   short signal experiments designed around the new paper narrative.

## 2. How The Code Was Refactored

### 2.1 Predictor Layer

The old predictor path was replaced by a more general depth-aware formulation.

Instead of only one MoDA-style block, the predictor now supports:

- `none`
- `unified`
- `additive`
- `gated`
- `residual`

These correspond to different ways of combining:

- standard causal sequence attention
- shallow same-timestep depth memory

This matters because the paper is no longer "we used MoDA".
It is now:

- does direct access to shallow geometric evidence help a deep latent
  world-model predictor during planning-oriented learning?

### 2.2 Training Entrypoint

The training script was rewritten so that:

- LeWM's encoder remains unchanged
- LeWM's loss remains unchanged
- the planner/eval interface remains unchanged
- only the latent predictor is replaced

This is important experimentally, because it isolates the claim:

- if results improve, the gain is attributable to the predictor design,
  not to unrelated encoder or loss changes

### 2.3 Smoke-Test Support

I added a small robustness fix to the dataloader path:

- when `num_workers=0`, `prefetch_factor` is now set safely

This allows 1-epoch / few-batch tests to run cleanly.
That is important because we should reject weak ideas quickly, instead of
waiting 20 hours to discover they were poorly framed.

## 3. Current Code Locations

### Main Refactored Files

- `/home/internship/wm_transfer_lab/LeWM_src/le-wm-main/moda_module.py`
- `/home/internship/wm_transfer_lab/LeWM_src/le-wm-main/train_moda.py`

### Experiment Notes / Launchers

- `/home/internship/wm_transfer_lab/LeWM_src/le-wm-main/EXPERIMENTS_DADP_SIGNAL.md`
- `/home/internship/wm_transfer_lab/LeWM_src/le-wm-main/launch_dadp_signal.sh`
- `/home/internship/wm_transfer_lab/LeWM_src/le-wm-main/launch_dadp_stage2.sh`

### Baseline / Logs / Outputs

- baseline checkpoints:
  `/home/internship/wm_transfer_lab/lewm_cache/pusht_real_3gpu_b42/`
- active DADP logs:
  `/home/internship/wm_transfer_lab/lewm_dadp_none_e5.log`
  `/home/internship/wm_transfer_lab/dadp_signal_runner.log`
  `/home/internship/wm_transfer_lab/dadp_signal_stage2.log`

## 4. What The New Method Actually Is

The method can be described as:

> A depth-augmented latent dynamics predictor that allows deep predictor layers
> to access shallow same-timestep evidence while preserving causal sequence
> attention over time.

Operationally, this means:

- the predictor still models temporal latent dynamics
- but each deep layer can also retrieve shallower features from the same token
  position
- this gives the model direct access to lower-level geometry/contact evidence
  during deep rollout

That is the real research object now.
MoDA is an inspiration for cross-layer retrieval, but not the full story.

## 5. Why This Is Better Than The Old "MoDA + LeWM" Story

The old story was weak because it sounded like:

- we took a mechanism from another line of work
- inserted it into LeWM
- hoped training metrics would look better

That is not a strong top-tier paper contribution by itself.

The new story is stronger because it defines:

1. a world-model-specific problem:
   **information dilution in deep predictors**
2. a world-model-specific hypothesis:
   planning needs persistent access to shallow geometry/contact evidence
3. a method family:
   multiple depth-memory fusion designs, not one arbitrary implementation
4. a clean experiment axis:
   matched predictor-only ablations under the same encoder/loss/planner

## 6. Innovation Points

These are the current innovation points worth emphasizing.

### 6.1 Problem Innovation

The central claim is not "MoDA works in LeWM".
The central claim is:

- **deep world model predictors suffer planning-oriented information dilution**

This reframes the bottleneck from generic architecture design to a concrete
 failure mode in latent world-model rollout.

### 6.2 Method Innovation

The predictor is not just deeper.
It is **depth-augmented**:

- temporal attention handles latent dynamics across time
- depth retrieval preserves access to shallow same-position evidence

This is exactly the kind of mechanism that makes sense for planning:

- shallow layers often encode geometry, edges, contact cues, alignment
- deep layers encode more abstract rollout logic
- planning needs both, not only the deepest abstraction

### 6.3 Experimental Innovation

The code now directly supports paper-quality ablations:

- `none` vs `unified`
- `additive`
- `gated`
- `residual`
- later: predictor depth ablation

This is important because the research claim can now be tested structurally,
not only through one model-vs-model comparison.

### 6.4 Practical Innovation

The method is implemented as a predictor-side change only:

- encoder unchanged
- loss unchanged
- planner interface unchanged

This makes it easier to argue:

- the improvement comes from controlling information flow inside the predictor,
  rather than from broad system-level retuning

## 7. Current Experiment Status

On `8card-6000ada`, the current schedule is:

1. `none`, 5 epochs
2. `unified`, 5 epochs
3. `additive`, 5 epochs
4. `gated`, 5 epochs

This is a signal-seeking stage, not a final benchmark stage.
The purpose is to decide whether the new method direction deserves:

- longer training
- planner evaluation
- predictor depth scaling experiments

## 8. Recommended Paper Framing

Recommended working name:

- **Depth-Augmented Dynamics Predictor (DADP)**

Recommended core message:

- World-model planning quality is limited not only by model capacity, but by
  information dilution inside deep latent predictors.
- We address this by making shallow evidence directly reachable during deep
  rollout.

MoDA credit should be handled as:

- inspiration for cross-layer retrieval
- not the entire paper contribution

## 9. Bottom Line

The codebase is no longer set up for a low-value "MoDA plugin" story.
It is now set up for a stronger research question:

- identify information dilution in world-model predictors
- test depth-augmented predictor designs
- validate with matched planning-oriented experiments

That is the right direction if the goal is a serious paper rather than a
 superficial architecture transplant.
