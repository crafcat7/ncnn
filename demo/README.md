# ncnn Demo Inference Test

Standalone inference testing tool for ncnn, with timing statistics, memory tracking, and input/output persistence.
All build and run parameters are configured by editing shell scripts instead of typing long command lines.

## Structure

```
demo/
├── CMakeLists.txt            CMake build configuration
├── executor/                 ncnn executor layer (decoupled from test logic)
│   ├── run_config.h         Run configuration structures
│   ├── ncnn_executor.h/cpp  Thin wrapper around ncnn Net/Extractor
│   ├── memory_tracker.h/cpp TrackingAllocator for memory statistics
│   └── io_utils.h/cpp       Input/output utilities (tensor dump, image load)
├── test/                    Test orchestration layer
│   ├── inference_test_case.h/cpp
│   └── inference_test_main.cpp   CLI entry point
├── scripts/
│   ├── build_demo.sh        Linux/macOS build script
│   ├── build_demo.cmd       Windows build script
│   └── run_demo.sh          Run wrapper
└── README.md
```

## Architecture

The codebase is split into two layers with a clean interface:

- **executor** (`NcnnExecutor`): Directly wraps ncnn `Net`/`Extractor`, manages lifecycle, exposes `run_once()`. No test logic here.
- **test** (`InferenceTestCase`): Orchestrates warmup, multi-loop timing, memory stats, and result saving. Depends only on the executor interface.

## Build

### 1. Edit Config

Edit `demo/scripts/demo_config.sh`:

```bash
TARGET_PLATFORM="linux"
TARGET_BACKEND="cpu"
BUILD_JOBS=1

USE_BENCHMARK_PRESET="ON"
BENCHMARK_PRESET="squeezenet"

# Or disable preset and use custom model paths
MODEL_PARAM="/path/to/model.param"
MODEL_BIN="/path/to/model.bin"
SAVE_DIR="demo/output/run1"

INPUT_SPECS=(
  "data::227,227,3"
)
```

### 2. Build

### Linux / macOS

```bash
./demo/scripts/build_demo.sh
```

### Android

```bash
./demo/scripts/build_demo.sh
```

Android binary output path:

```bash
build-demo-android-arm64-v8a/demo/demo_inference_test
```

### Windows

```cmd
demo\scripts\build_demo.cmd --platform windows --backend cpu
```

### 3. Run

```bash
./demo/scripts/run_demo.sh
```

### 4. Run On Android Device

This workflow is fixed to device `adb -s 4569513f`.

Example Android config:

```bash
TARGET_PLATFORM="android"
TARGET_BACKEND="vulkan"
ANDROID_SERIAL="4569513f"
ANDROID_ABI="arm64-v8a"
USE_BENCHMARK_PRESET="ON"
BENCHMARK_PRESET="mobilenet_v2"
```

```bash
./demo/scripts/run_demo_android.sh
```

## Config Items

| Option | Description |
|--------|-------------|
| `TARGET_PLATFORM` | `linux` / `macos` / `android` / `windows` |
| `TARGET_BACKEND` | `cpu` / `vulkan` |
| `BUILD_JOBS` | Build parallelism, default `1`, recommended for WSL |
| `USE_BENCHMARK_PRESET` | `ON` means reuse ncnn benchmark built-in model presets |
| `BENCHMARK_PRESET` | Example: `squeezenet`, `mobilenet`, `resnet50`, `yolov4-tiny` |
| `MODEL_PARAM` | Model `.param` path |
| `MODEL_BIN` | Model `.bin` path |
| `TEST_LOOPS` | Repeated inference count |
| `WARMUP_LOOPS` | Warmup count |
| `NUM_THREADS` | ncnn thread count |
| `SAVE_DIR` | Output directory |
| `INPUT_SPECS` | Input list, format `name:file:shape` |
| `OUTPUT_SPECS` | Output name list, empty means all outputs |
| `ANDROID_SERIAL` | Android device serial, default `4569513f` |
| `ANDROID_WORKDIR` | Remote execution directory on device |

## Benchmark Presets

When `USE_BENCHMARK_PRESET="ON"`, `run_demo.sh` and `run_demo_android.sh` auto-fill:

- model param path from `benchmark/*.param`
- default input tensor name + shape based on `benchmark/benchncnn.cpp`
- no `.bin` file required, matching benchncnn behavior

Priority rule:

- if `MODEL_PARAM` / `MODEL_BIN` / `INPUT_SPECS` / `OUTPUT_SPECS` is explicitly configured in `demo_config.sh`, custom config wins
- benchmark preset is only applied when those custom fields are left empty

Example presets:

- `squeezenet`
- `mobilenet`
- `mobilenet_v2`
- `mobilenet_v3`
- `resnet18`
- `resnet50`
- `googlenet`
- `yolov4-tiny`
- `nanodet_m`
- `vision_transformer`
- `FastestDet`

### Input Format

- Image: `"input:photo.jpg:224,224,3"` — loads and resizes the image via `simpleocv`
- Raw tensor: `"input:data.bin:224,224,3"` — loads raw float32 binary
- Shape only: `"input::224,224,3"` — creates a constant-filled tensor

### Output

`--save-dir` produces:
- `stats.json` — timing latency (min/max/avg/p50/p90/p99) and peak memory
- `input_*.bin` / `input_*.json` — input tensor raw data and metadata
- `output_*.bin` / `output_*.json` — output tensor raw data and metadata

## Memory Statistics

CPU mode: Tracks `TrackingAllocator` wrapping `PoolAllocator` — reports peak bytes allocated including workspace.

Vulkan mode: Reports host-side allocator stats; GPU device memory tracking is not included in v1.

## WSL Notes

Recommended settings in `demo_config.sh`:

```bash
BUILD_JOBS=1
LIGHTWEIGHT_BUILD="ON"
```

If `adb` is not found inside WSL, either:

```bash
sudo apt install android-sdk-platform-tools
```

or add Windows adb into WSL `PATH` before running `run_demo_android.sh`.
