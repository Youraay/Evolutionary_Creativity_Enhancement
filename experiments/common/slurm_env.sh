# Environment of the Slurm batch scripts; sourced, not executed.
#
# The dispatcher submits from the repository root, so SLURM_SUBMIT_DIR is the project directory unless
# PROJECT_DIR is set. Settings come from <repo>/.env (see .env.example); the Python environment is
# activated from VENV_PATH (default <repo>/.venv).

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR" || exit 1
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env"
    set +a
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck disable=SC1091
source "${VENV_PATH:-$PROJECT_DIR/.venv}/bin/activate"

echo "[INFO] $(date '+%F %T') | host $(hostname) | python $(command -v python) | project $PROJECT_DIR"
nvidia-smi --query-gpu=index,name,clocks.sm,power.draw,clocks_event_reasons.hw_power_brake_slowdown \
    --format=csv 2>/dev/null || true
