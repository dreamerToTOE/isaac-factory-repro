#!/usr/bin/env bash
# Run only with this project's isolated Python; preserve all existing installations.
set -euo pipefail
: "${ISAACSIM_PATH:=/home/ubuntu2004/isaacsim-4.5.0}"
: "${FACTORY_VENV:?Set FACTORY_VENV to the isolated Python 3.10 environment}"
unset PYTHONPATH LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
# Upstream setup scripts reference unset variables.
set +u
source "$ISAACSIM_PATH/setup_conda_env.sh"
set -u
# Prefer isolated pinned learning packages over the simulator's bundled archives.
export PYTHONPATH="$FACTORY_VENV/lib/python3.10/site-packages:$PYTHONPATH"
export LD_PRELOAD="$ISAACSIM_PATH/kit/libcarb.so"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$FACTORY_VENV/bin/python" "$script_dir/python_entry.py" "$@"
