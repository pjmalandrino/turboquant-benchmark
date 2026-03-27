#!/bin/zsh
# =============================================================
# TurboQuant KV Cache Experiment Runner
# =============================================================
# Usage: ./run_experiment.sh <config> [NUM_RUNS]
#   config:   baseline | turbo3 | turbo4
#   NUM_RUNS: number of runs (default: 1)
#
# Examples:
#   ./run_experiment.sh baseline      # 1 run with f16 KV cache
#   ./run_experiment.sh turbo3        # 1 run with turbo3 KV cache
#   ./run_experiment.sh turbo3 3      # 3 runs with turbo3 KV cache
# =============================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="$SCRIPT_DIR/models/google_gemma-3-4b-it-Q4_K_M.gguf"
PROMPT="$SCRIPT_DIR/prompt.txt"
LLAMA_CLI="$SCRIPT_DIR/build/bin/llama-completion"
RESULTS_DIR="$SCRIPT_DIR/results"

# Common params
NGL=99
CTX=8192
N_PREDICT=512
SEED=42
TEMP=0

# ----- parse args -----

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <baseline|turbo3|turbo4> [NUM_RUNS]"
    exit 1
fi

CONFIG_NAME="$1"
NUM_RUNS="${2:-1}"

case "$CONFIG_NAME" in
    baseline)
        CACHE_FLAGS="--cache-type-k f16 --cache-type-v f16"
        ;;
    turbo3)
        CACHE_FLAGS="--cache-type-k turbo3 --cache-type-v turbo3"
        ;;
    turbo4)
        CACHE_FLAGS="--cache-type-k turbo4 --cache-type-v turbo4"
        ;;
    *)
        echo "ERROR: Unknown config '$CONFIG_NAME'. Use: baseline, turbo3, or turbo4"
        exit 1
        ;;
esac

# ----- helpers -----

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ----- pre-flight checks -----

echo ""
echo "============================================="
echo "  TurboQuant Experiment — $CONFIG_NAME"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="
echo ""

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model not found at $MODEL"
    exit 1
fi

if [[ ! -x "$LLAMA_CLI" ]]; then
    echo "ERROR: llama-cli not found or not executable at $LLAMA_CLI"
    exit 1
fi

if [[ ! -f "$PROMPT" ]]; then
    echo "ERROR: Prompt file not found at $PROMPT"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

log "Config:     $CONFIG_NAME"
log "Cache:      $CACHE_FLAGS"
log "Model:      $(basename "$MODEL")"
log "Prompt:     $(wc -w < "$PROMPT") words"
log "Runs:       $NUM_RUNS"
log "Seed:       $SEED"
log "Temp:       $TEMP"
log "Context:    $CTX"
log "N_predict:  $N_PREDICT"
echo ""

# ----- run experiments -----

for run_num in $(seq 1 "$NUM_RUNS"); do
    run_id="${CONFIG_NAME}_run${run_num}"

    log "========== RUN $run_num/$NUM_RUNS [$run_id] =========="

    output_file="$RESULTS_DIR/${run_id}_output.txt"
    memory_file="$RESULTS_DIR/${run_id}_memory.txt"
    cmd_file="$RESULTS_DIR/${run_id}_command.txt"

    # Build the command
    full_cmd="$LLAMA_CLI -m $MODEL -ngl $NGL -c $CTX -n $N_PREDICT -f $PROMPT --seed $SEED --temp $TEMP -no-cnv $CACHE_FLAGS"
    echo "$full_cmd" > "$cmd_file"
    log "Command saved to: $cmd_file"
    echo ""
    echo "  $full_cmd"
    echo ""

    # Capture system memory BEFORE
    log "Capturing memory snapshot BEFORE run..."
    {
        echo "=== SYSTEM MEMORY BEFORE ==="
        echo "--- vm_stat ---"
        vm_stat
        echo ""
        echo "--- memory_pressure ---"
        memory_pressure 2>/dev/null | head -5 || echo "(memory_pressure not available)"
    } > "$memory_file" 2>&1

    # Run in foreground — all output visible in terminal and saved to file
    log "Starting inference... (output below)"
    echo "---------------------------------------------"

    eval "$full_cmd" 2>&1 | tee "$output_file"
    exit_code=${pipestatus[1]:-$?}

    echo ""
    echo "---------------------------------------------"

    # Capture system memory AFTER
    log "Capturing memory snapshot AFTER run..."
    {
        echo ""
        echo "=== SYSTEM MEMORY AFTER ==="
        echo "--- vm_stat ---"
        vm_stat
        echo ""
        echo "--- memory_pressure ---"
        memory_pressure 2>/dev/null | head -5 || echo "(memory_pressure not available)"
    } >> "$memory_file" 2>&1

    # Summary
    echo ""
    log "Run completed — exit code: $exit_code"
    log "Output file:  $output_file"
    log "Memory file:  $memory_file"

    if [[ $exit_code -ne 0 ]]; then
        log "WARNING: Non-zero exit code!"
    fi

    # Extract key metrics
    echo ""
    log "=== KEY METRICS ==="
    grep -E "(common_perf_print|llama_kv_cache:|llama_memory_breakdown)" "$output_file" 2>/dev/null || \
        echo "  (no metrics found — check $output_file)"
    echo ""

    # Pause between runs if doing multiple
    if [[ $run_num -lt $NUM_RUNS ]]; then
        log "Pausing 3s before next run..."
        sleep 3
    fi
done

# ----- done -----

echo ""
echo "============================================="
log "All $NUM_RUNS run(s) for '$CONFIG_NAME' complete"
echo "  Results in: $RESULTS_DIR/"
echo "============================================="
echo ""
echo "Files generated for this config:"
ls -la "$RESULTS_DIR/${CONFIG_NAME}_"* 2>/dev/null
echo ""
echo "To run the next config:"
if [[ "$CONFIG_NAME" == "baseline" ]]; then
    echo "  ./run_experiment.sh turbo3"
elif [[ "$CONFIG_NAME" == "turbo3" ]]; then
    echo "  ./run_experiment.sh turbo4"
fi
