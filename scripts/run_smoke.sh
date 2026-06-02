#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/env.sh"
${PYTHON:-python3} - <<'PY'
from pathlib import Path
root = Path.cwd()
print('Smoke check: repository files')
for p in [
    'code/moda_module.py',
    'code/config/eval/pusht.yaml',
    'experiments/moda_only_learned_residual_proposal.py',
    'experiments/moda_only_residual_confirm50_audit.py',
    'experiments/risk_controlled_moda_integration.py',
]:
    path = root / p
    print(f'{p}:', 'OK' if path.exists() else 'MISSING')
print('Done. Full planning evaluation requires PushT assets, checkpoints, and candidate pools.')
PY
