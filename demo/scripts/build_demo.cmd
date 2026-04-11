@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "NCNN_ROOT=%SCRIPT_DIR%..\.."

set "PLATFORM="
set "BACKEND=cpu"
set "BUILD_TYPE=Release"
set "ANDROID_NDK="
set "ANDROID_ABI=arm64-v8a"
set "ANDROID_API=android-21"
set "BUILD_DIR=build-demo-windows"
set "BUILD_JOBS=1"

:parse
if "%~1"=="" goto :done
if /i "%~1"=="--platform" (set "PLATFORM=%~2" & shift & shift & goto :parse)
if /i "%~1"=="--backend" (set "BACKEND=%~2" & shift & shift & goto :parse)
if /i "%~1"=="--build-type" (set "BUILD_TYPE=%~2" & shift & shift & goto :parse)
if /i "%~1"=="--ndk" (set "ANDROID_NDK=%~2" & shift & shift & goto :parse)
if /i "%~1"=="--abi" (set "ANDROID_ABI=%~2" & shift & shift & goto :parse)
if /i "%~1"=="--api" (set "ANDROID_API=%~2" & shift & shift & goto :parse)
if /i "%~1"=="--jobs" (set "BUILD_JOBS=%~2" & shift & shift & goto :parse)
if /i "%~1"=="--help" goto :usage
if /i "%~1"=="-h" goto :usage
echo Unknown option: %~1
:usage
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo   --platform android^|macos^|windows   Target platform
echo   --backend cpu^|vulkan              Backend (default: cpu)
echo   --build-type Debug^|Release          Build type (default: Release)
echo   --ndk ^<path^>                    Android NDK root
echo   --abi ^<abi^>                     Android ABI (default: arm64-v8a)
echo   --api ^<N^>                       Android API (default: android-21)
echo   --jobs ^<N^>                      Build jobs (default: 1)
echo.
echo Example:
echo   %~nx0 --platform windows --backend vulkan
exit /b 1

:done
if "%PLATFORM%"=="" (
    echo Error: --platform is required
    goto :usage
)

if "%BACKEND%"=="vulkan" (
    set "NCNN_VULKAN=ON"
) else (
    set "NCNN_VULKAN=OFF"
)

echo === ncnn demo build ===
echo   Platform:    %PLATFORM%
echo   Backend:     %BACKEND% (NCNN_VULKAN=%NCNN_VULKAN%)
echo   Build type:  %BUILD_TYPE%
echo   Build jobs:  %BUILD_JOBS%
echo.

if "%PLATFORM%"=="android" (
    if "%ANDROID_NDK%"=="" (
        if defined ANDROID_NDK_HOME (
            set "ANDROID_NDK=%ANDROID_NDK_HOME%"
        ) else (
            echo Error: ANDROID_NDK not set. Use --ndk ^<path^>
            exit /b 1
        )
    )
    if not exist "%ANDROID_NDK%\build\cmake\android.toolchain.cmake" (
        echo Error: android.toolchain.cmake not found in NDK
        exit /b 1
    )
    set "BUILD_DIR=%BUILD_DIR%-android-%ANDROID_ABI%"
    echo Building for Android %ANDROID_API% %ANDROID_ABI%
    echo   NDK: %ANDROID_NDK%
    mkdir "%BUILD_DIR%" 2>nul
    pushd "%BUILD_DIR%"
    cmake -G "Unix Makefiles" ^
          -DCMAKE_TOOLCHAIN_FILE="%ANDROID_NDK%\build\cmake\android.toolchain.cmake" ^
          -DANDROID_ABI="%ANDROID_ABI%" ^
          -DANDROID_PLATFORM="%ANDROID_API%" ^
          -DNCNN_VULKAN=%NCNN_VULKAN% ^
          -DNCNN_BUILD_DEMO=ON ^
          -DNCNN_BUILD_BENCHMARK=OFF ^
          -DNCNN_BUILD_EXAMPLES=OFF ^
          -DNCNN_BUILD_TOOLS=OFF ^
          -DNCNN_BUILD_TESTS=OFF ^
          -DNCNN_OPENMP=OFF ^
          -DNCNN_SIMPLEOCV=ON ^
          -DNCNN_RUNTIME_CPU=OFF ^
          -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
          "%NCNN_ROOT%"
    popd
    echo.
    echo Configured. Build with: cd %BUILD_DIR% ^&^& cmake --build . --parallel %BUILD_JOBS%
) else if "%PLATFORM%"=="macos" (
    set "BUILD_DIR=%BUILD_DIR%-macos"
    echo Building for macOS
    mkdir "%BUILD_DIR%" 2>nul
    pushd "%BUILD_DIR%"
    cmake -G "Unix Makefiles" ^
          -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
          -DNCNN_VULKAN=%NCNN_VULKAN% ^
          -DNCNN_BUILD_DEMO=ON ^
          -DNCNN_BUILD_BENCHMARK=OFF ^
          -DNCNN_BUILD_EXAMPLES=OFF ^
          -DNCNN_BUILD_TOOLS=OFF ^
          -DNCNN_BUILD_TESTS=OFF ^
          -DNCNN_OPENMP=OFF ^
          -DNCNN_SIMPLEOCV=ON ^
          -DNCNN_RUNTIME_CPU=OFF ^
          "%NCNN_ROOT%"
    popd
    echo.
    echo Configured. Build with: cd %BUILD_DIR% ^&^& cmake --build . --parallel %BUILD_JOBS%
) else if "%PLATFORM%"=="windows" (
    set "BUILD_DIR=%BUILD_DIR%-windows"
    echo Building for Windows
    mkdir "%BUILD_DIR%" 2>nul
    pushd "%BUILD_DIR%"
    cmake -G "Visual Studio 17 2022" -A x64 ^
          -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
          -DNCNN_VULKAN=%NCNN_VULKAN% ^
          -DNCNN_BUILD_DEMO=ON ^
          -DNCNN_BUILD_BENCHMARK=OFF ^
          -DNCNN_BUILD_EXAMPLES=OFF ^
          -DNCNN_BUILD_TOOLS=OFF ^
          -DNCNN_BUILD_TESTS=OFF ^
          -DNCNN_OPENMP=OFF ^
          -DNCNN_SIMPLEOCV=ON ^
          -DNCNN_RUNTIME_CPU=OFF ^
          "%NCNN_ROOT%"
    popd
    echo.
    echo Configured. Build with: cd %BUILD_DIR% ^&^& cmake --build . --parallel %BUILD_JOBS%
) else (
    echo Error: Unknown platform: %PLATFORM%
    exit /b 1
)

endlocal
