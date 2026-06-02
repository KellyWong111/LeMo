---
tags:
- world-model
- planning
- robotics
- pusht
- moda
- lewm
license: mit
---

# LeWM + MoDA Planning Experiments

This repository packages the code used for LeWM + MoDA planning diagnostics on PushT, including MoDA candidate-pool calibration, baseline-safe integration, and MoDA-only residual proposal experiments.

## What is included

- `code/`: LeWM source tree with MoDA-related modules and configs.
- `experiments/`: standalone analysis and experiment scripts used for PAC-MoDA / MoDA-only planning studies.
- `docs/`: local technical path report and talk-track notes.
- `manifests/local_artifacts_manifest.txt`: manifest of local artifacts available on the original workstation.
- `artifacts/`: intentionally left small in this code release. Large candidate pools/checkpoints should be downloaded or copied separately.

## Key scripts

Most recent MoDA-only and PAC-MoDA scripts are in `experiments/`:

- `moda_only_learned_residual_proposal.py`: success-conditioned residual proposal correction.
- `moda_only_residual_confirm50_audit.py`: paired-index audit and residual scale sensitivity.
- `moda_only_intra_episode_audit.py`: diagnostic for global AUC vs intra-episode discriminability.
- `moda_only_planner_in_loop_calibrated_cem.py`: planner-in-the-loop calibrated CEM attempt.
- `moda_only_action_sensitive_contrastive.py`: contrastive diagnostic head.
- `risk_controlled_moda_integration.py`: baseline-safe opportunity-aware MoDA integration.
- `pac_moda_native_calibration_report.py`: MoDA-native calibration report generator.

## Main technical conclusion

MoDA candidate coverage is useful, but raw MoDA cost is poorly aligned with planning success. Post-hoc candidate reranking improves global AUC but does not reliably improve MoDA-only top1 because intra-episode candidate discriminability is weak. The most promising MoDA-only direction is success-conditioned residual proposal correction, which modifies action proposal generation instead of only reranking final candidates.

Conservative current result summary:

- bsl-relative integration can improve system-level top1, but it depends on a strong baseline fallback and should not be presented as MoDA-only.
- AUC-only calibration gains are not enough because they can reflect episode difficulty leakage.
- Learned residual proposal gives consistent paired improvement and near-miss reduction, but the absolute top1 is not yet a stable standalone 65+ result.

## Environment

The original remote environment used Python 3.10 and CUDA GPUs. A frozen dependency snapshot is provided under `code/requirements_frozen.txt` and related `requirements_frozen_v*.txt` files.

A minimal setup pattern is:

```bash
conda create -n lewm-moda python=3.10 -y
conda activate lewm-moda
pip install -r code/requirements_frozen.txt
```

For MuJoCo / headless evaluation:

```bash
export MUJOCO_GL=egl
export PYTHONPATH=$PWD/code:$PWD/code/wm_experiment_scripts:$PWD/experiments
```

Some scripts expect Stable World Model / PushT assets. Set:

```bash
export STABLEWM_HOME=/path/to/.stable_worldmodel
```

## Artifacts layout

Large artifacts are not copied into this small code release by default. The original local artifact root was:

```text
/Users/wangyijing/lewm_migration_bundle/wm_runs
```

For runnable experiments, place or symlink artifacts as:

```text
artifacts/wm_runs/
  stateroll_normalbudget_candidate_pool_s300_steps30_n100/
    proposal_data/
    raw_rollout_npz/
  bsl_normalbudget_candidate_pool_s300_steps30_n100/
    proposal_data/
    raw_rollout_npz/
  pac_moda_v2_full_n100_20260529/
  rpn_residual_proposal/
  ...
```

Then set:

```bash
export LEWM_WM_RUNS=$PWD/artifacts/wm_runs
```

Note: several historical scripts contain absolute paths from the original remote workstation. If running on a new machine, either create a compatible symlink or patch the `ROOT` constants in scripts to use `LEWM_WM_RUNS`.

## Smoke checks

A lightweight import / structure check:

```bash
bash scripts/run_smoke.sh
```

A full residual audit requires candidate pools and a working world model evaluation environment. See:

```bash
bash scripts/run_residual_audit_example.sh
```

## Recommended Hugging Face split

For a clean public release, use two repositories:

1. Code repo: this directory.
2. Artifact repo: selected checkpoints, candidate pools, and result CSV/JSON files.

Do not upload the full 113G local migration bundle unless needed. It contains many intermediate and failed experiments.
