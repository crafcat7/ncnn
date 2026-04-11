#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCNN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# shellcheck source=/dev/null
source "$SCRIPT_DIR/demo_config.sh"

PLATFORM="${TARGET_PLATFORM}"
BACKEND="${TARGET_BACKEND}"
BUILD_DIR="${BUILD_DIR_PREFIX}-${PLATFORM}"
NCNN_VULKAN="OFF"

if [[ "$BACKEND" == "vulkan" ]]; then
    NCNN_VULKAN="ON"
fi

COMMON_ARGS=(
    -DNCNN_VULKAN="$NCNN_VULKAN"
    -DNCNN_BUILD_DEMO=ON
    -DNCNN_BUILD_BENCHMARK=OFF
    -DNCNN_BUILD_EXAMPLES=OFF
    -DNCNN_BUILD_TOOLS=OFF
    -DNCNN_BUILD_TESTS=OFF
    -DNCNN_SIMPLEOCV=ON
)

if [[ "${LIGHTWEIGHT_BUILD}" == "ON" ]]; then
    COMMON_ARGS+=(
        -DNCNN_OPENMP=OFF
        -DNCNN_AVX=OFF
        -DNCNN_AVX2=OFF
        -DNCNN_AVX512=OFF
        -DNCNN_FMA=OFF
        -DNCNN_XOP=OFF
        -DNCNN_INT8=OFF
        -DNCNN_BF16=OFF
    )
fi

if [[ "$PLATFORM" == "android" ]]; then
    COMMON_ARGS+=(
        -DNCNN_RUNTIME_CPU=ON
    )
elif [[ "${LIGHTWEIGHT_BUILD}" == "ON" ]]; then
    COMMON_ARGS+=(
        -DNCNN_RUNTIME_CPU=OFF
    )
fi

echo "=== ncnn demo build ==="
echo "  Platform:     $PLATFORM"
echo "  Backend:      $BACKEND"
echo "  Build type:   $BUILD_TYPE"
echo "  Build dir:    $BUILD_DIR"
echo "  Build jobs:   $BUILD_JOBS"
echo "  Lightweight:  $LIGHTWEIGHT_BUILD"
echo ""

mkdir -p "$NCNN_ROOT/$BUILD_DIR"

case "$PLATFORM" in
    android)
        if [[ -z "$ANDROID_NDK_ROOT" || ! -d "$ANDROID_NDK_ROOT" ]]; then
            echo "Error: ANDROID_NDK_ROOT is not configured in demo_config.sh"
            exit 1
        fi

        BUILD_DIR="${BUILD_DIR_PREFIX}-${PLATFORM}-${ANDROID_ABI}"

        cmake -S "$NCNN_ROOT" -B "$NCNN_ROOT/$BUILD_DIR" \
              -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
              -DANDROID_ABI="$ANDROID_ABI" \
              -DANDROID_PLATFORM="$ANDROID_API" \
              -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
              "${COMMON_ARGS[@]}"
        ;;

    linux|macos)
        cmake -S "$NCNN_ROOT" -B "$NCNN_ROOT/$BUILD_DIR" \
              -G "Unix Makefiles" \
              -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
              "${COMMON_ARGS[@]}"
        ;;

    windows)
        cmake -S "$NCNN_ROOT" -B "$NCNN_ROOT/$BUILD_DIR" \
              -G "Visual Studio 17 2022" -A x64 \
              -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
              "${COMMON_ARGS[@]}"
        ;;

    *)
        echo "Error: unsupported TARGET_PLATFORM=$PLATFORM"
        exit 1
        ;;
esac

cmake --build "$NCNN_ROOT/$BUILD_DIR" --target demo_inference_test --parallel "$BUILD_JOBS"

echo ""
echo "Build done: $NCNN_ROOT/$BUILD_DIR/demo/demo_inference_test"
