#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCNN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# shellcheck source=/dev/null
source "$SCRIPT_DIR/demo_config.sh"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/benchmark_presets.sh"

CUSTOM_MODEL_PARAM="$MODEL_PARAM"
CUSTOM_MODEL_BIN="$MODEL_BIN"
CUSTOM_INPUT_SPECS=("${INPUT_SPECS[@]}")
CUSTOM_OUTPUT_SPECS=("${OUTPUT_SPECS[@]}")

USE_PRESET_RESOLVED="$USE_BENCHMARK_PRESET"

if [[ "$USE_BENCHMARK_PRESET" == "ON" ]]; then
    demo_benchmark_apply_preset "$BENCHMARK_PRESET"

    if [[ -n "$CUSTOM_MODEL_PARAM" ]]; then
        MODEL_PARAM="$CUSTOM_MODEL_PARAM"
    fi
    if [[ -n "$CUSTOM_MODEL_BIN" ]]; then
        MODEL_BIN="$CUSTOM_MODEL_BIN"
    fi
    if [[ ${#CUSTOM_INPUT_SPECS[@]} -gt 0 ]]; then
        INPUT_SPECS=("${CUSTOM_INPUT_SPECS[@]}")
    fi
    if [[ ${#CUSTOM_OUTPUT_SPECS[@]} -gt 0 ]]; then
        OUTPUT_SPECS=("${CUSTOM_OUTPUT_SPECS[@]}")
    fi
fi

BUILD_DIR="$NCNN_ROOT/${BUILD_DIR_PREFIX}-${TARGET_PLATFORM}"
EXEC="$BUILD_DIR/demo/demo_inference_test"

if [[ ! -f "$EXEC" ]]; then
    echo "Error: executable not found at $EXEC"
    echo "Please build first by editing demo_config.sh and running ./demo/scripts/build_demo.sh"
    exit 1
fi

if [[ -z "$MODEL_PARAM" ]]; then
    echo "Error: MODEL_PARAM is empty in demo_config.sh"
    exit 1
fi

if [[ "$MODEL_PARAM" != /* ]]; then
    MODEL_PARAM="$NCNN_ROOT/$MODEL_PARAM"
fi

if [[ -n "$MODEL_BIN" && "$MODEL_BIN" != /* ]]; then
    MODEL_BIN="$NCNN_ROOT/$MODEL_BIN"
fi

if [[ -n "$MODEL_BIN" && ! -f "$MODEL_BIN" ]]; then
    case "$MODEL_PARAM" in
        "$NCNN_ROOT"/benchmark/*.param)
            echo "Warning: benchmark param detected and bin not found, fallback to param-only mode"
            echo "  missing bin: $MODEL_BIN"
            MODEL_BIN=""
            ;;
        *)
            echo "Error: model bin not found: $MODEL_BIN"
            exit 1
            ;;
    esac
fi

CMD=(
    "$EXEC"
    --param "$MODEL_PARAM"
    --loops "$TEST_LOOPS"
    --warmup "$WARMUP_LOOPS"
    --threads "$NUM_THREADS"
    --backend "$TARGET_BACKEND"
    --save-dir "$SAVE_DIR"
)

if [[ -n "$MODEL_BIN" ]]; then
    CMD+=(--bin "$MODEL_BIN")
fi

for spec in "${INPUT_SPECS[@]}"; do
    [[ -n "$spec" ]] && CMD+=(--input "$spec")
done

for spec in "${OUTPUT_SPECS[@]}"; do
    [[ -n "$spec" ]] && CMD+=(--output "$spec")
done

echo "=== ncnn demo run ==="
echo "  Binary:   $EXEC"
echo "  Backend:  $TARGET_BACKEND"
echo "  Param:    $MODEL_PARAM"
echo "  Bin:      $MODEL_BIN"
echo "  Save dir: $SAVE_DIR"
if [[ "$USE_PRESET_RESOLVED" == "ON" ]]; then
    echo "  Preset:   $BENCHMARK_PRESET"
else
    echo "  Preset:   disabled"
fi
echo ""

"${CMD[@]}"
