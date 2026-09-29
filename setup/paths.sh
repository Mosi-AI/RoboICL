# Source from Bash. Existing overrides are preserved.
export ROBOICL_CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export ROBOICL_DATA_ROOT="${ROBOICL_DATA_ROOT:-$ROBOICL_CODE_ROOT/data}"
export ROBOICL_RESULTS_ROOT="${ROBOICL_RESULTS_ROOT:-$ROBOICL_CODE_ROOT/results}"
export PYTHONPATH="$ROBOICL_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
