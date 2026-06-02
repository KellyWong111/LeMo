#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/code:$ROOT_DIR/code/wm_experiment_scripts:$ROOT_DIR/experiments:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LEWM_WM_RUNS="${LEWM_WM_RUNS:-$ROOT_DIR/artifacts/wm_runs}"
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable_worldmodel}"
echo "ROOT_DIR=$ROOT_DIR"
echo "PYTHONPATH=$PYTHONPATH"
echo "LEWM_WM_RUNS=$LEWM_WM_RUNS"
echo "STABLEWM_HOME=$STABLEWM_HOME"
