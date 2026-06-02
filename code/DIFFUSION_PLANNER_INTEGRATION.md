# Diffusion Planner Integration Notes

This repository currently evaluates LeWM with:

- world model cost: `swm.policy.AutoCostModel`
- solver/planner: `stable_worldmodel.solver.CEMSolver`
- policy wrapper: `stable_worldmodel.policy.WorldModelPolicy`

The first integration step is to replace the solver slot with a local
`DiffusionPlannerSolver` without changing the encoder/predictor/cost stack.

## Current status

- Added `diffusion_planner.DiffusionPlannerSolver`
- Added Hydra solver config at `config/eval/solver/diffusion.yaml`
- The solver matches the `stable_worldmodel.solver.Solver` protocol:
  - `configure(action_space, n_envs, config)`
  - `solve(info_dict, init_action=None)`
  - returns `outputs["actions"]` with shape `(num_envs, horizon, action_dim * action_block)`

## Important caveat

The current implementation is only a diffusion-style planning scaffold:

- it uses truncated iterative proposal refinement
- it reranks proposals with the LeWM world-model cost
- it does **not** yet load or train a learned diffusion policy checkpoint

So this is not yet a true DiffusionDrive reproduction. It is the clean
integration path that lets us:

1. replace CEM at the same solver interface
2. plug in a learned planner later
3. compare:
   - CEM
   - diffusion-only planner
   - diffusion + world-model rerank

## Next steps

1. Add a learned planner network and checkpoint loading.
2. Add a trainer on expert action sequences conditioned on current state + goal.
3. Run eval parity checks on PushT with the same `PlanConfig`.
