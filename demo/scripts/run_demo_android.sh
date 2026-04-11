#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCNN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ADB_BIN="${ADB_BIN:-adb}"
ADB_BIN="/mnt/c/Environment/platform-tools/adb.exe"

if ! "$ADB_BIN" version >/dev/null 2>&1; then
    echo "Error: adb not found or not runnable: $ADB_BIN"
    exit 1
fi

# shellcheck source=/dev/null
source "$SCRIPT_DIR/demo_config.sh"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/benchmark_presets.sh"

CUSTOM_MODEL_PARAM="$MODEL_PARAM"
CUSTOM_MODEL_BIN="$MODEL_BIN"
CUSTOM_INPUT_SPECS=("${INPUT_SPECS[@]}")
CUSTOM_OUTPUT_SPECS=("${OUTPUT_SPECS[@]}")

USE_PRESET_RESOLVED="$USE_BENCHMARK_PRESET"

if [[ "$TARGET_PLATFORM" != "android" ]]; then
    echo "Error: TARGET_PLATFORM must be android in demo_config.sh"
    exit 1
fi

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

BUILD_DIR="$NCNN_ROOT/${BUILD_DIR_PREFIX}-${TARGET_PLATFORM}-${ANDROID_ABI}"
EXEC_LOCAL="$BUILD_DIR/demo/demo_inference_test"

if [[ ! -f "$EXEC_LOCAL" ]]; then
    echo "Error: Android binary not found at $EXEC_LOCAL"
    echo "Run ./demo/scripts/build_demo.sh after setting TARGET_PLATFORM=android"
    exit 1
fi

REMOTE_DIR="$ANDROID_WORKDIR"
REMOTE_EXEC="$REMOTE_DIR/demo_inference_test"
REMOTE_SAVE_DIR="$REMOTE_DIR/output"
REMOTE_PARAM=""
REMOTE_BIN=""

if [[ -z "$MODEL_PARAM" ]]; then
    echo "Error: MODEL_PARAM is empty"
    exit 1
fi

if [[ "$MODEL_PARAM" != /* ]]; then
    MODEL_PARAM="$NCNN_ROOT/$MODEL_PARAM"
fi

REMOTE_PARAM="$REMOTE_DIR/$(basename "$MODEL_PARAM")"

if [[ -n "$MODEL_BIN" ]]; then
    if [[ "$MODEL_BIN" != /* ]]; then
        MODEL_BIN="$NCNN_ROOT/$MODEL_BIN"
    fi

    if [[ ! -f "$MODEL_BIN" ]]; then
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

    if [[ -n "$MODEL_BIN" ]]; then
        REMOTE_BIN="$REMOTE_DIR/$(basename "$MODEL_BIN")"
    fi
fi

"$ADB_BIN" -s "$ANDROID_SERIAL" shell mkdir -p "$REMOTE_DIR"
"$ADB_BIN" -s "$ANDROID_SERIAL" shell mkdir -p "$REMOTE_SAVE_DIR"
"$ADB_BIN" -s "$ANDROID_SERIAL" push "$EXEC_LOCAL" "$REMOTE_EXEC"
"$ADB_BIN" -s "$ANDROID_SERIAL" push "$MODEL_PARAM" "$REMOTE_PARAM"
if [[ -n "$REMOTE_BIN" ]]; then
    "$ADB_BIN" -s "$ANDROID_SERIAL" push "$MODEL_BIN" "$REMOTE_BIN"
fi
"$ADB_BIN" -s "$ANDROID_SERIAL" shell chmod +x "$REMOTE_EXEC"

ANDROID_INPUT_ARGS=()
for spec in "${INPUT_SPECS[@]}"; do
    [[ -z "$spec" ]] && continue

    name="${spec%%:*}"
    rest="${spec#*:}"

    if [[ "$rest" == "$spec" ]]; then
        ANDROID_INPUT_ARGS+=(--input "$spec")
        continue
    fi

    file_part="${rest%%:*}"
    shape_part="${rest#*:}"
    if [[ "$shape_part" == "$rest" ]]; then
        shape_part=""
    fi

    if [[ -n "$file_part" ]]; then
        local_file="$file_part"
        if [[ "$local_file" != /* ]]; then
            local_file="$NCNN_ROOT/$local_file"
        fi
        remote_file="$REMOTE_DIR/$(basename "$local_file")"
        "$ADB_BIN" -s "$ANDROID_SERIAL" push "$local_file" "$remote_file"
        if [[ -n "$shape_part" ]]; then
            ANDROID_INPUT_ARGS+=(--input "$name:$remote_file:$shape_part")
        else
            ANDROID_INPUT_ARGS+=(--input "$name:$remote_file")
        fi
    else
        ANDROID_INPUT_ARGS+=(--input "$spec")
    fi
done

ANDROID_OUTPUT_ARGS=()
for spec in "${OUTPUT_SPECS[@]}"; do
    [[ -n "$spec" ]] && ANDROID_OUTPUT_ARGS+=(--output "$spec")
done

CMD=(
    "$REMOTE_EXEC"
    --param "$REMOTE_PARAM"
    --loops "$TEST_LOOPS"
    --warmup "$WARMUP_LOOPS"
    --threads "$NUM_THREADS"
    --backend "$TARGET_BACKEND"
    --save-dir "$REMOTE_SAVE_DIR"
)

if [[ -n "$REMOTE_BIN" ]]; then
    CMD+=(--bin "$REMOTE_BIN")
fi

CMD+=("${ANDROID_INPUT_ARGS[@]}")
CMD+=("${ANDROID_OUTPUT_ARGS[@]}")

echo "=== android demo run ==="
echo "  serial:    $ANDROID_SERIAL"
echo "  remote:    $REMOTE_DIR"
if [[ "$USE_PRESET_RESOLVED" == "ON" ]]; then
    echo "  preset:    $BENCHMARK_PRESET"
else
    echo "  preset:    disabled"
fi
echo "  backend:   $TARGET_BACKEND"
echo ""

"$ADB_BIN" -s "$ANDROID_SERIAL" shell "cd '$REMOTE_DIR' && $(printf '%q ' "${CMD[@]}")"
