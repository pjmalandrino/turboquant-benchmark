#!/usr/bin/env bash
# =============================================================
# TurboQuant KV Cache Experiment Runner — v2
# =============================================================
# Usage: ./run_experiment.sh <config> [options]
#
#   config:   baseline | q8_0 | q4_0 | turbo3 | turbo4
#
# Options:
#   -r NUM_RUNS     Number of runs (default: 3)
#   -p PROMPT       Prompt to use: short | medium | long (default: medium)
#   -m MODEL        Model file path (default: auto-detect in models/)
#
# Examples:
#   ./run_experiment.sh baseline
#   ./run_experiment.sh turbo3 -r 1 -p short
#   ./run_experiment.sh q8_0 -p long -m models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
# =============================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_BIN="$SCRIPT_DIR/build/bin/llama-completion"
RESULTS_DIR="$SCRIPT_DIR/results"

# Defaults
NUM_RUNS=3
PROMPT_NAME="medium"
MODEL=""

# Common inference params
NGL=99
CTX=8192
N_PREDICT=512
SEED=42
TEMP=0

# ----- parse args -----

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <baseline|q8_0|q4_0|turbo3|turbo4> [-r NUM_RUNS] [-p short|medium|long] [-m MODEL_PATH]"
    exit 1
fi

CONFIG_NAME="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r) NUM_RUNS="$2"; shift 2 ;;
        -p) PROMPT_NAME="$2"; shift 2 ;;
        -m) MODEL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Resolve cache flags
case "$CONFIG_NAME" in
    baseline) CACHE_FLAGS="--cache-type-k f16 --cache-type-v f16" ;;
    q8_0)     CACHE_FLAGS="--cache-type-k q8_0 --cache-type-v q8_0" ;;
    q4_0)     CACHE_FLAGS="--cache-type-k q4_0 --cache-type-v q4_0" ;;
    turbo3)   CACHE_FLAGS="--cache-type-k turbo3 --cache-type-v turbo3" ;;
    turbo4)   CACHE_FLAGS="--cache-type-k turbo4 --cache-type-v turbo4" ;;
    *) echo "ERROR: Unknown config '$CONFIG_NAME'. Use: baseline, q8_0, q4_0, turbo3, turbo4"; exit 1 ;;
esac

# Resolve prompt
PROMPT_FILE="$SCRIPT_DIR/prompts/${PROMPT_NAME}.txt"
if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "ERROR: Prompt file not found: $PROMPT_FILE"
    echo "  Available: $(ls "$SCRIPT_DIR/prompts/"*.txt 2>/dev/null | xargs -I{} basename {} .txt | tr '\n' ' ')"
    exit 1
fi

# Resolve model (auto-detect if not specified)
if [[ -z "$MODEL" ]]; then
    MODEL_COUNT=$(find "$SCRIPT_DIR/models" -name "*.gguf" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$MODEL_COUNT" -eq 0 ]]; then
        echo "ERROR: No .gguf files found in models/"
        exit 1
    elif [[ "$MODEL_COUNT" -eq 1 ]]; then
        MODEL=$(find "$SCRIPT_DIR/models" -name "*.gguf" | head -1)
    else
        echo "ERROR: Multiple models found. Specify one with -m:"
        find "$SCRIPT_DIR/models" -name "*.gguf" -exec basename {} \;
        exit 1
    fi
fi

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model not found at $MODEL"
    exit 1
fi

if [[ ! -x "$LLAMA_BIN" ]]; then
    echo "ERROR: llama-completion not found at $LLAMA_BIN"
    echo "  Build first: cmake -B build -DGGML_METAL=ON && cmake --build build"
    exit 1
fi

# ----- helpers -----

log() { echo "[$(date '+%H:%M:%S')] $*"; }

MODEL_SHORT="$(basename "$MODEL" .gguf)"

# Results go in a subfolder per model
RUN_RESULTS_DIR="$RESULTS_DIR/${MODEL_SHORT}/${PROMPT_NAME}"
mkdir -p "$RUN_RESULTS_DIR"

# ----- banner -----

echo ""
echo "============================================="
echo "  TurboQuant Experiment v2"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="
echo ""
log "Config:     $CONFIG_NAME"
log "Cache:      $CACHE_FLAGS"
log "Model:      $MODEL_SHORT"
log "Prompt:     $PROMPT_NAME ($(wc -w < "$PROMPT_FILE" | tr -d ' ') words)"
log "Runs:       $NUM_RUNS"
log "Seed:       $SEED | Temp: $TEMP | Ctx: $CTX | Predict: $N_PREDICT"
log "Output dir: $RUN_RESULTS_DIR"
echo ""

# ----- run experiments -----

for run_num in $(seq 1 "$NUM_RUNS"); do
    run_id="${CONFIG_NAME}_run${run_num}"

    log "========== RUN $run_num/$NUM_RUNS [$run_id] =========="

    output_file="$RUN_RESULTS_DIR/${run_id}_output.txt"
    memory_file="$RUN_RESULTS_DIR/${run_id}_memory.txt"
    cmd_file="$RUN_RESULTS_DIR/${run_id}_command.txt"

    # Build the command (use relative model path in saved command for portability)
    full_cmd="$LLAMA_BIN -m $MODEL -ngl $NGL -c $CTX -n $N_PREDICT -f $PROMPT_FILE --seed $SEED --temp $TEMP -no-cnv $CACHE_FLAGS"

    # Save portable command (relative paths)
    echo "./build/bin/llama-completion -m models/$(basename "$MODEL") -ngl $NGL -c $CTX -n $N_PREDICT -f prompts/${PROMPT_NAME}.txt --seed $SEED --temp $TEMP -no-cnv $CACHE_FLAGS" > "$cmd_file"

    log "Command:"
    echo "  $(cat "$cmd_file")"
    echo ""

    # Capture system memory BEFORE
    log "Memory snapshot BEFORE..."
    {
        echo "=== SYSTEM MEMORY BEFORE ==="
        vm_stat 2>/dev/null || true
        echo ""
    } > "$memory_file" 2>&1

    # Run in foreground — all output visible
    log "Starting inference..."
    echo "---------------------------------------------"

    eval "$full_cmd" 2>&1 | tee "$output_file"
    exit_code=${PIPESTATUS[0]:-$?}

    echo ""
    echo "---------------------------------------------"

    # Capture system memory AFTER
    {
        echo ""
        echo "=== SYSTEM MEMORY AFTER ==="
        vm_stat 2>/dev/null || true
    } >> "$memory_file" 2>&1

    # Summary
    log "Exit code: $exit_code"

    if [[ $exit_code -ne 0 ]]; then
        log "WARNING: Non-zero exit code!"
    fi

    # Extract key metrics (compact)
    echo ""
    log "--- METRICS ---"
    grep -E "^(common_perf_print|llama_kv_cache:|llama_memory_breakdown)" "$output_file" 2>/dev/null | \
        grep -vE "(Dumping|kv_overrides|metadata)" || \
        echo "  (no metrics found)"
    echo ""

    # Pause between runs
    if [[ $run_num -lt $NUM_RUNS ]]; then
        log "Cooling down 5s..."
        sleep 5
    fi
done

# ----- summary -----

echo ""
echo "============================================="
log "Done: $NUM_RUNS run(s) of '$CONFIG_NAME' on $MODEL_SHORT ($PROMPT_NAME prompt)"
echo "============================================="
echo ""
echo "Results in: $RUN_RESULTS_DIR/"
ls "$RUN_RESULTS_DIR/${CONFIG_NAME}_"* 2>/dev/null
echo ""

# Suggest next step
case "$CONFIG_NAME" in
    baseline) echo "Next: ./run_experiment.sh q8_0 -p $PROMPT_NAME -m $MODEL" ;;
    q8_0)     echo "Next: ./run_experiment.sh q4_0 -p $PROMPT_NAME -m $MODEL" ;;
    q4_0)     echo "Next: ./run_experiment.sh turbo3 -p $PROMPT_NAME -m $MODEL" ;;
    turbo3)   echo "Next: ./run_experiment.sh turbo4 -p $PROMPT_NAME -m $MODEL" ;;
    turbo4)   echo "All configs tested!" ;;
esac
