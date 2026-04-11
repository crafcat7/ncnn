#!/usr/bin/env bash

# Edit this file to change build and run behavior.

# build config
TARGET_PLATFORM="android"   # linux | macos | android | windows
TARGET_BACKEND="cpu"        # cpu | vulkan
BUILD_TYPE="Release"
BUILD_DIR_PREFIX="build-demo"
BUILD_JOBS=1

# android config
ANDROID_NDK_ROOT="${ANDROID_NDK:-}"
ANDROID_ABI="arm64-v8a"
ANDROID_API="android-21"

# lightweight build switches for WSL or low-memory environments
LIGHTWEIGHT_BUILD="ON"

# run config
USE_BENCHMARK_PRESET="ON"
BENCHMARK_PRESET="shufflenet"

MODEL_PARAM=""
MODEL_BIN=""
TEST_LOOPS=20
WARMUP_LOOPS=5
NUM_THREADS=1
SAVE_DIR="demo/output/run1"

# android deploy + run config
ANDROID_SERIAL="4569513f"
ANDROID_WORKDIR="/data/local/tmp/ncnn-demo"

# input format: name:file:shape
# shape examples: 224  /  224,224  /  224,224,3
INPUT_SPECS=(
)

# leave empty to use all model outputs
OUTPUT_SPECS=(
)
