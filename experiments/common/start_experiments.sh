#!/bin/bash
# =====================================================================================
# Run one experiment under your own Slurm user, e.g. to share the GPU quota. Every user gets their
# own job database, results and logs:
#
#     <RESULTS_ROOT>/<experiment>_<user>/
#         jobs.db            job queue (created with experiments.<experiment>.create_jobs)
#         results/           run directories
#         logs/              slurm-<jobid>.out/.err
#         dispatcher.log     output of the dispatcher
#         submit_job.sh      generated Slurm batch script
#
# Usage (on a login node, from anywhere):
#     experiments/common/start_experiments.sh <experiment> start         create the database (if missing), start the dispatcher in tmux
#     experiments/common/start_experiments.sh <experiment> status        progress, Slurm jobs, dispatcher
#     experiments/common/start_experiments.sh <experiment> stop          stop the dispatcher (running Slurm jobs continue)
#     experiments/common/start_experiments.sh <experiment> keepalive     start the dispatcher only if it is not running (for cron)
#     experiments/common/start_experiments.sh <experiment> install-cron  run keepalive every 10 minutes via crontab
#
# <experiment> is a package below experiments/, e.g. fitness_embedding, tournament_size or long_runs.
# Requirements: member of the Slurm account; the models are in the shared Hugging Face cache, the jobs
# run offline. Configuration via environment variables (defaults below).
# =====================================================================================
set -euo pipefail

SCRIPT_PATH=$(readlink -f "$0")
PROJECT=$(dirname "$(dirname "$(dirname "$SCRIPT_PATH")")")
EXPERIMENT=${1:-}
[ -n "$EXPERIMENT" ] && [ -f "$PROJECT/experiments/$EXPERIMENT/worker.py" ] || {
    sed -n '2,23p' "$SCRIPT_PATH" | sed 's/^# \{0,1\}//'; exit 1; }
PREFIX=$(echo "$EXPERIMENT" | tr '[:lower:]' '[:upper:]')
PYTHON=${VENV_PATH:-$PROJECT/.venv}/bin/python
NODE=${EXP_NODE-}                  # e.g. gpu31-021; empty = any node
ACCOUNT=${EXP_ACCOUNT:-dldevel}
PARTITION=${EXP_PARTITION:-gpu2}
TIME_LIMIT=${EXP_TIME:-15:00:00}   # experiments with longer runs set their own limit in the dispatcher
RESULTS_ROOT=${RESULTS_ROOT:-/work/dldevel/public/Evolutionary_Novelty_Enhancement_Results}

USER_NAME=${USER:-$(id -un)}
BASE=${EXP_BASE:-$RESULTS_ROOT/${EXPERIMENT}_$USER_NAME}
DB=$BASE/jobs.db
RESULTS=$BASE/results
LOGDIR=$BASE/logs
DISPATCHER_LOG=$BASE/dispatcher.log
SUBMIT=$BASE/submit_job.sh
TMUX_SESSION=${EXPERIMENT}_$USER_NAME
MAX_JOBS=${EXP_MAX_RUNNING_JOBS:-4}   # QOS gpu2 allows 4 running jobs per user

umask 002   # results readable and deletable by the group

# --- Environment of dispatcher, create_jobs and (via sbatch --export=ALL) workers ----------------
# Set explicitly so that a .env of another user cannot override anything: python-dotenv keeps
# variables that are already set.
export RESULTS_ROOT
export "${PREFIX}_DB_FILE=$DB"
export "${PREFIX}_RESULT_DIR=$RESULTS"
export "${PREFIX}_SUBMIT_SCRIPT=$SUBMIT"
export MAX_RUNNING_JOBS=$MAX_JOBS
export HF_CACHE=${HF_CACHE:-/work/dldevel/public/Evolutionary_Diffusion_Enhancement/huggingface}
export HF_HUB_OFFLINE=1
export HF_TOKEN=                    # offline, no token needed; prevents taking one from .env
export DEVICE=cuda
export POPULATION_SIZE=60
export NUM_GENERATIONS=30
export POLL_INTERVAL=30
export PYTHONPATH=$PROJECT
export PYTHONUNBUFFERED=1
ENV_VARS="RESULTS_ROOT ${PREFIX}_DB_FILE ${PREFIX}_RESULT_DIR ${PREFIX}_SUBMIT_SCRIPT MAX_RUNNING_JOBS HF_CACHE
          HF_HUB_OFFLINE HF_TOKEN DEVICE POPULATION_SIZE NUM_GENERATIONS POLL_INTERVAL PYTHONPATH PYTHONUNBUFFERED"

die() { echo "[ERROR] $*" >&2; exit 1; }

dispatcher_running() {
    # only the Python process itself counts, not shells whose command line contains the text
    pgrep -u "$USER_NAME" -f "^[^ ]*python[0-9.]* (-u )?-m experiments\.${EXPERIMENT}\.dispatch" >/dev/null 2>&1
}

db_query() {
    "$PYTHON" - "$DB" "$1" <<'EOF'
import sqlite3, sys
connection = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
for row in connection.execute(sys.argv[2]):
    print("|".join("" if v is None else str(v) for v in row))
EOF
}

unfinished_jobs() {
    [ -r "$DB" ] || { echo 0; return; }
    db_query "SELECT COUNT(*) FROM jobs WHERE status IN ('OPEN','RETRY','CLAIMED','RUNNING')"
}

check_prerequisites() {
    [ -x "$PYTHON" ] || die "$PYTHON is not executable (set VENV_PATH)"
    "$PYTHON" -c "import torch, diffusers, transformers" 2>/dev/null \
        || die "The Python environment ($PYTHON) cannot import torch/diffusers/transformers"
    command -v sbatch >/dev/null || die "sbatch not found (run on a login node)"
    command -v tmux >/dev/null || die "tmux not found"
    sacctmgr -n -P show assoc user="$USER_NAME" account="$ACCOUNT" format=account 2>/dev/null | grep -q "$ACCOUNT" \
        || die "User $USER_NAME has no Slurm account $ACCOUNT"
}

write_submit_script() {
    mkdir -p "$LOGDIR"
    cat > "$SUBMIT" <<EOF
#!/bin/bash
#SBATCH --job-name=${EXPERIMENT}_$USER_NAME
#SBATCH --output=$LOGDIR/slurm-%j.out
#SBATCH --error=$LOGDIR/slurm-%j.err
#SBATCH --partition=$PARTITION
${NODE:+#SBATCH --nodelist=$NODE}
#SBATCH --gres=gpu:1
#SBATCH --account=$ACCOUNT
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=$TIME_LIMIT
# Generated by start_experiments.sh. The environment comes from the dispatcher via --export=ALL;
# deliberately no 'source .env'.

cd "$PROJECT"
echo "[INFO] Host: \$(hostname) | User: \$(id -un) | Job ID: \$1"
nvidia-smi --query-gpu=index,name,clocks_event_reasons.hw_power_brake_slowdown --format=csv,noheader
"$PYTHON" -m experiments.${EXPERIMENT}.worker "\$1"
EOF
    chmod +x "$SUBMIT"
}

create_db_if_missing() {
    mkdir -p "$BASE" "$RESULTS" "$LOGDIR"
    if [ -e "$DB" ]; then
        echo "[INFO] Job database exists: $DB (reused)"
    else
        (cd "$PROJECT" && "$PYTHON" -m "experiments.${EXPERIMENT}.create_jobs" --db "$DB") \
            || die "Could not create the job database"
    fi
}

start_dispatcher() {
    if dispatcher_running; then
        echo "[INFO] Dispatcher is already running (tmux session: $TMUX_SESSION)"
        return 0
    fi
    local cmd="cd '$PROJECT' && '$PYTHON' -u -m experiments.${EXPERIMENT}.dispatch 2>&1 | tee -a '$DISPATCHER_LOG'"
    echo "[INFO] $(date '+%F %T') dispatcher start ($1) on $(hostname) as $USER_NAME" >> "$DISPATCHER_LOG"
    # Pass the environment explicitly: an already running tmux server would otherwise keep its old one.
    local exports=""
    for v in $ENV_VARS; do exports+="export $v='${!v}'; "; done
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        tmux new-window -t "$TMUX_SESSION" -c "$PROJECT" "bash -lc \"umask 002; $exports $cmd\""
    else
        tmux new-session -d -s "$TMUX_SESSION" -c "$PROJECT" "bash -lc \"umask 002; $exports $cmd\""
    fi
    sleep 5
    dispatcher_running || die "The dispatcher did not start, see $DISPATCHER_LOG"
    echo "[INFO] Dispatcher started (tmux session: $TMUX_SESSION, log: $DISPATCHER_LOG)"
}

cmd_status() {
    echo "== Experiment: $EXPERIMENT | user: $USER_NAME | base: $BASE"
    if [ -r "$DB" ]; then
        echo "== Jobs per status"
        db_query "SELECT status, COUNT(*) FROM jobs GROUP BY status ORDER BY status" \
            | awk -F'|' '{printf "   %-8s %s\n",$1,$2}'
        local failed
        failed=$(db_query "SELECT id||': '||substr(coalesce(last_error,''),1,200) FROM jobs WHERE status IN ('FAILED','RETRY')")
        [ -n "$failed" ] && { echo "== Failed:"; echo "$failed" | sed 's/^/   /'; }
    else
        echo "   (no job database yet, run '$SCRIPT_PATH $EXPERIMENT start' first)"
    fi
    echo "== Slurm"
    squeue -u "$USER_NAME" -o "   %A %j %T %M %N %r" | head -20
    if dispatcher_running; then echo "== Dispatcher: running (tmux attach -t $TMUX_SESSION)"; else echo "== Dispatcher: NOT RUNNING"; fi
}

case "${2:-}" in
    start)
        check_prerequisites
        write_submit_script
        create_db_if_missing
        start_dispatcher "start"
        echo "[INFO] Status: $SCRIPT_PATH $EXPERIMENT status"
        echo "[INFO] Recommended against unnoticed dispatcher crashes: $SCRIPT_PATH $EXPERIMENT install-cron"
        ;;
    keepalive)
        if ! dispatcher_running && [ "$(unfinished_jobs)" -gt 0 ] 2>/dev/null; then
            write_submit_script
            start_dispatcher "keepalive"
        fi
        ;;
    stop)
        if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
            tmux kill-session -t "$TMUX_SESSION"
            echo "[INFO] Dispatcher stopped. Running Slurm jobs continue; 'start' resumes."
        else
            echo "[INFO] No tmux session $TMUX_SESSION"
        fi
        ;;
    status)
        cmd_status
        ;;
    install-cron)
        line="5-55/10 * * * * $SCRIPT_PATH $EXPERIMENT keepalive >> $BASE/keepalive.log 2>&1"
        if crontab -l 2>/dev/null | grep -qF "$SCRIPT_PATH $EXPERIMENT keepalive"; then
            echo "[INFO] Cron entry exists already"
        else
            (crontab -l 2>/dev/null; echo "$line") | crontab -
            echo "[INFO] Cron installed: $line"
            echo "[INFO] Note: cron runs only on this login node ($(hostname))."
        fi
        ;;
    *)
        sed -n '2,23p' "$SCRIPT_PATH" | sed 's/^# \{0,1\}//'
        exit 1
        ;;
esac
