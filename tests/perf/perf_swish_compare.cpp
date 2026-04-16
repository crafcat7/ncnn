// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "benchmark.h"
#include "cpu.h"
#include "mat.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#if __ARM_NEON
#include <arm_neon.h>
#include "arm/arm_usability.h"
#include "arm/neon_mathfun.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "arm/neon_mathfun_fp16s.h"
#endif
#endif

#define PERF_WARMUP_COUNT  10
#define PERF_RUN_COUNT     20
#define PERF_TARGET_MIN_MS 5.0

struct PerfResult
{
    double time_min;
    double time_max;
    double time_avg;
    double time_median;
    int loop_count;
};

struct DiffResult
{
    float max_abs_err;
    float max_rel_err;
};

static void sort_doubles(double* arr, int n)
{
    for (int i = 1; i < n; i++)
    {
        double key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key)
        {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

static void print_perf_result(const char* tag, const PerfResult& result)
{
    if (result.loop_count > 1)
    {
        fprintf(stdout, "%-48s  min = %8.2f  max = %8.2f  avg = %8.2f  median = %8.2f  (x%d)\n",
                tag, result.time_min, result.time_max, result.time_avg, result.time_median, result.loop_count);
    }
    else
    {
        fprintf(stdout, "%-48s  min = %8.2f  max = %8.2f  avg = %8.2f  median = %8.2f\n",
                tag, result.time_min, result.time_max, result.time_avg, result.time_median);
    }
}

static void print_speedup(const PerfResult& legacy, const PerfResult& optimized)
{
    fprintf(stdout, "%-48s  min = %6.2fx  avg = %6.2fx  median = %6.2fx\n\n",
            "speedup(optimized/legacy)",
            legacy.time_min / optimized.time_min,
            legacy.time_avg / optimized.time_avg,
            legacy.time_median / optimized.time_median);
}

static void print_diff_result(const DiffResult& diff)
{
    fprintf(stdout, "%-48s  abs = %.8f  rel = %.8f\n",
            "numerical_diff(legacy vs optimized)",
            diff.max_abs_err, diff.max_rel_err);
}

static int calibrate_inner_loops(double warmup_min_ms)
{
    int inner_loops = 1;
    if (warmup_min_ms > 0 && warmup_min_ms < PERF_TARGET_MIN_MS)
    {
        while (inner_loops * warmup_min_ms < PERF_TARGET_MIN_MS)
            inner_loops *= 10;
    }
    return inner_loops;
}

static PerfResult run_benchmark(void (*fn)(void*, int), const void* src, size_t bytes, int elemcount)
{
    std::vector<unsigned char> buf(bytes);

    double warmup_min_ms = DBL_MAX;
    for (int i = 0; i < PERF_WARMUP_COUNT; i++)
    {
        memcpy(buf.data(), src, bytes);
        double t0 = ncnn::get_current_time();
        fn(buf.data(), elemcount);
        double t1 = ncnn::get_current_time();
        double t = t1 - t0;
        if (t < warmup_min_ms) warmup_min_ms = t;
    }

    int inner_loops = calibrate_inner_loops(warmup_min_ms);
    double times[PERF_RUN_COUNT];
    double time_sum = 0.f;
    double time_min_val = DBL_MAX;
    double time_max_val = -DBL_MAX;

    for (int i = 0; i < PERF_RUN_COUNT; i++)
    {
        double start = ncnn::get_current_time();

        for (int k = 0; k < inner_loops; k++)
        {
            memcpy(buf.data(), src, bytes);
            fn(buf.data(), elemcount);
        }

        double end = ncnn::get_current_time();
        double t = end - start;

        times[i] = t;
        time_sum += t;
        if (t < time_min_val) time_min_val = t;
        if (t > time_max_val) time_max_val = t;
    }

    sort_doubles(times, PERF_RUN_COUNT);

    PerfResult result;
    result.time_min = time_min_val;
    result.time_max = time_max_val;
    result.time_avg = time_sum / PERF_RUN_COUNT;
    result.time_median = PERF_RUN_COUNT % 2 == 0 ? (times[PERF_RUN_COUNT / 2 - 1] + times[PERF_RUN_COUNT / 2]) / 2.0 : times[PERF_RUN_COUNT / 2];
    result.loop_count = inner_loops;
    return result;
}

static DiffResult compare_fp32(const float* a, const float* b, int n)
{
    DiffResult diff = {0.f, 0.f};

    for (int i = 0; i < n; i++)
    {
        float abs_err = fabsf(a[i] - b[i]);
        float rel_err = abs_err / fmaxf(fabsf(a[i]), 1e-6f);
        if (abs_err > diff.max_abs_err) diff.max_abs_err = abs_err;
        if (rel_err > diff.max_rel_err) diff.max_rel_err = rel_err;
    }

    return diff;
}

#if NCNN_BF16
static DiffResult compare_bf16(const unsigned short* a, const unsigned short* b, int n)
{
    DiffResult diff = {0.f, 0.f};

    for (int i = 0; i < n; i++)
    {
        float av = ncnn::bfloat16_to_float32(a[i]);
        float bv = ncnn::bfloat16_to_float32(b[i]);
        float abs_err = fabsf(av - bv);
        float rel_err = abs_err / fmaxf(fabsf(av), 1e-6f);
        if (abs_err > diff.max_abs_err) diff.max_abs_err = abs_err;
        if (rel_err > diff.max_rel_err) diff.max_rel_err = rel_err;
    }

    return diff;
}
#endif

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static DiffResult compare_fp16(const __fp16* a, const __fp16* b, int n)
{
    DiffResult diff = {0.f, 0.f};

    for (int i = 0; i < n; i++)
    {
        float av = (float)a[i];
        float bv = (float)b[i];
        float abs_err = fabsf(av - bv);
        float rel_err = abs_err / fmaxf(fabsf(av), 1e-6f);
        if (abs_err > diff.max_abs_err) diff.max_abs_err = abs_err;
        if (rel_err > diff.max_rel_err) diff.max_rel_err = rel_err;
    }

    return diff;
}
#endif

#if __ARM_NEON
static void swish_legacy_fp32(void* data, int elemcount)
{
    float* ptr = (float*)data;
    float32x4_t one = vdupq_n_f32(1.f);
    int i = 0;

#if __aarch64__
    for (; i + 15 < elemcount; i += 16)
    {
        float32x4_t p0 = vld1q_f32(ptr);
        float32x4_t p1 = vld1q_f32(ptr + 4);
        float32x4_t p2 = vld1q_f32(ptr + 8);
        float32x4_t p3 = vld1q_f32(ptr + 12);
        p0 = div_ps(p0, vaddq_f32(one, exp_ps(vnegq_f32(p0))));
        p1 = div_ps(p1, vaddq_f32(one, exp_ps(vnegq_f32(p1))));
        p2 = div_ps(p2, vaddq_f32(one, exp_ps(vnegq_f32(p2))));
        p3 = div_ps(p3, vaddq_f32(one, exp_ps(vnegq_f32(p3))));
        vst1q_f32(ptr, p0);
        vst1q_f32(ptr + 4, p1);
        vst1q_f32(ptr + 8, p2);
        vst1q_f32(ptr + 12, p3);
        ptr += 16;
    }
#endif

    for (; i + 7 < elemcount; i += 8)
    {
        float32x4_t p0 = vld1q_f32(ptr);
        float32x4_t p1 = vld1q_f32(ptr + 4);
        p0 = div_ps(p0, vaddq_f32(one, exp_ps(vnegq_f32(p0))));
        p1 = div_ps(p1, vaddq_f32(one, exp_ps(vnegq_f32(p1))));
        vst1q_f32(ptr, p0);
        vst1q_f32(ptr + 4, p1);
        ptr += 8;
    }
    for (; i + 3 < elemcount; i += 4)
    {
        float32x4_t p = vld1q_f32(ptr);
        p = div_ps(p, vaddq_f32(one, exp_ps(vnegq_f32(p))));
        vst1q_f32(ptr, p);
        ptr += 4;
    }
    for (; i < elemcount; i++)
    {
        *ptr = *ptr / (1.f + expf(-*ptr));
        ptr++;
    }
}

static void swish_optimized_fp32(void* data, int elemcount)
{
    float* ptr = (float*)data;
    int i = 0;

#if __aarch64__
    for (; i + 15 < elemcount; i += 16)
    {
        float32x4_t p0 = vld1q_f32(ptr);
        float32x4_t p1 = vld1q_f32(ptr + 4);
        float32x4_t p2 = vld1q_f32(ptr + 8);
        float32x4_t p3 = vld1q_f32(ptr + 12);
        p0 = vmulq_f32(p0, sigmoid_ps(p0));
        p1 = vmulq_f32(p1, sigmoid_ps(p1));
        p2 = vmulq_f32(p2, sigmoid_ps(p2));
        p3 = vmulq_f32(p3, sigmoid_ps(p3));
        vst1q_f32(ptr, p0);
        vst1q_f32(ptr + 4, p1);
        vst1q_f32(ptr + 8, p2);
        vst1q_f32(ptr + 12, p3);
        ptr += 16;
    }
#endif

    for (; i + 7 < elemcount; i += 8)
    {
        float32x4_t p0 = vld1q_f32(ptr);
        float32x4_t p1 = vld1q_f32(ptr + 4);
        p0 = vmulq_f32(p0, sigmoid_ps(p0));
        p1 = vmulq_f32(p1, sigmoid_ps(p1));
        vst1q_f32(ptr, p0);
        vst1q_f32(ptr + 4, p1);
        ptr += 8;
    }
    for (; i + 3 < elemcount; i += 4)
    {
        float32x4_t p = vld1q_f32(ptr);
        p = vmulq_f32(p, sigmoid_ps(p));
        vst1q_f32(ptr, p);
        ptr += 4;
    }
    for (; i < elemcount; i++)
    {
        *ptr = *ptr / (1.f + expf(-*ptr));
        ptr++;
    }
}

#if NCNN_BF16
static void swish_legacy_bf16(void* data, int elemcount)
{
    unsigned short* ptr = (unsigned short*)data;
    float32x4_t one = vdupq_n_f32(1.f);
    int i = 0;

#if __aarch64__
    for (; i + 15 < elemcount; i += 16)
    {
        uint16x8_t p01 = vld1q_u16(ptr);
        uint16x8_t p23 = vld1q_u16(ptr + 8);
        float32x4_t p0 = bfloat2float(vget_low_u16(p01));
        float32x4_t p1 = bfloat2float(vget_high_u16(p01));
        float32x4_t p2 = bfloat2float(vget_low_u16(p23));
        float32x4_t p3 = bfloat2float(vget_high_u16(p23));
        p0 = div_ps(p0, vaddq_f32(one, exp_ps(vnegq_f32(p0))));
        p1 = div_ps(p1, vaddq_f32(one, exp_ps(vnegq_f32(p1))));
        p2 = div_ps(p2, vaddq_f32(one, exp_ps(vnegq_f32(p2))));
        p3 = div_ps(p3, vaddq_f32(one, exp_ps(vnegq_f32(p3))));
        p01 = vcombine_u16(float2bfloat(p0), float2bfloat(p1));
        p23 = vcombine_u16(float2bfloat(p2), float2bfloat(p3));
        vst1q_u16(ptr, p01);
        vst1q_u16(ptr + 8, p23);
        ptr += 16;
    }
#endif

    for (; i + 7 < elemcount; i += 8)
    {
        uint16x8_t p = vld1q_u16(ptr);
        float32x4_t p0 = bfloat2float(vget_low_u16(p));
        float32x4_t p1 = bfloat2float(vget_high_u16(p));
        p0 = div_ps(p0, vaddq_f32(one, exp_ps(vnegq_f32(p0))));
        p1 = div_ps(p1, vaddq_f32(one, exp_ps(vnegq_f32(p1))));
        p = vcombine_u16(float2bfloat(p0), float2bfloat(p1));
        vst1q_u16(ptr, p);
        ptr += 8;
    }
    for (; i + 3 < elemcount; i += 4)
    {
        float32x4_t p = bfloat2float(vld1_u16(ptr));
        p = div_ps(p, vaddq_f32(one, exp_ps(vnegq_f32(p))));
        vst1_u16(ptr, float2bfloat(p));
        ptr += 4;
    }
    for (; i < elemcount; i++)
    {
        float v = ncnn::bfloat16_to_float32(*ptr);
        v = v / (1.f + expf(-v));
        *ptr = ncnn::float32_to_bfloat16(v);
        ptr++;
    }
}

static void swish_optimized_bf16(void* data, int elemcount)
{
    unsigned short* ptr = (unsigned short*)data;
    int i = 0;

#if __aarch64__
    for (; i + 15 < elemcount; i += 16)
    {
        uint16x8_t p01 = vld1q_u16(ptr);
        uint16x8_t p23 = vld1q_u16(ptr + 8);
        float32x4_t p0 = bfloat2float(vget_low_u16(p01));
        float32x4_t p1 = bfloat2float(vget_high_u16(p01));
        float32x4_t p2 = bfloat2float(vget_low_u16(p23));
        float32x4_t p3 = bfloat2float(vget_high_u16(p23));
        p0 = vmulq_f32(p0, sigmoid_ps(p0));
        p1 = vmulq_f32(p1, sigmoid_ps(p1));
        p2 = vmulq_f32(p2, sigmoid_ps(p2));
        p3 = vmulq_f32(p3, sigmoid_ps(p3));
        p01 = vcombine_u16(float2bfloat(p0), float2bfloat(p1));
        p23 = vcombine_u16(float2bfloat(p2), float2bfloat(p3));
        vst1q_u16(ptr, p01);
        vst1q_u16(ptr + 8, p23);
        ptr += 16;
    }
#endif

    for (; i + 7 < elemcount; i += 8)
    {
        uint16x8_t p = vld1q_u16(ptr);
        float32x4_t p0 = bfloat2float(vget_low_u16(p));
        float32x4_t p1 = bfloat2float(vget_high_u16(p));
        p0 = vmulq_f32(p0, sigmoid_ps(p0));
        p1 = vmulq_f32(p1, sigmoid_ps(p1));
        p = vcombine_u16(float2bfloat(p0), float2bfloat(p1));
        vst1q_u16(ptr, p);
        ptr += 8;
    }
    for (; i + 3 < elemcount; i += 4)
    {
        float32x4_t p = bfloat2float(vld1_u16(ptr));
        p = vmulq_f32(p, sigmoid_ps(p));
        vst1_u16(ptr, float2bfloat(p));
        ptr += 4;
    }
    for (; i < elemcount; i++)
    {
        float v = ncnn::bfloat16_to_float32(*ptr);
        v = v / (1.f + expf(-v));
        *ptr = ncnn::float32_to_bfloat16(v);
        ptr++;
    }
}
#endif

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static void swish_legacy_fp16sa(void* data, int elemcount)
{
    __fp16* ptr = (__fp16*)data;
    float16x8_t one = vdupq_n_f16(1.f);
    int i = 0;

    for (; i + 31 < elemcount; i += 32)
    {
        float16x8_t p0 = vld1q_f16(ptr);
        float16x8_t p1 = vld1q_f16(ptr + 8);
        float16x8_t p2 = vld1q_f16(ptr + 16);
        float16x8_t p3 = vld1q_f16(ptr + 24);
        p0 = vdivq_f16(p0, vaddq_f16(one, exp_ps_f16(vnegq_f16(p0))));
        p1 = vdivq_f16(p1, vaddq_f16(one, exp_ps_f16(vnegq_f16(p1))));
        p2 = vdivq_f16(p2, vaddq_f16(one, exp_ps_f16(vnegq_f16(p2))));
        p3 = vdivq_f16(p3, vaddq_f16(one, exp_ps_f16(vnegq_f16(p3))));
        vst1q_f16(ptr, p0);
        vst1q_f16(ptr + 8, p1);
        vst1q_f16(ptr + 16, p2);
        vst1q_f16(ptr + 24, p3);
        ptr += 32;
    }
    for (; i + 15 < elemcount; i += 16)
    {
        float16x8_t p0 = vld1q_f16(ptr);
        float16x8_t p1 = vld1q_f16(ptr + 8);
        p0 = vdivq_f16(p0, vaddq_f16(one, exp_ps_f16(vnegq_f16(p0))));
        p1 = vdivq_f16(p1, vaddq_f16(one, exp_ps_f16(vnegq_f16(p1))));
        vst1q_f16(ptr, p0);
        vst1q_f16(ptr + 8, p1);
        ptr += 16;
    }
    for (; i + 7 < elemcount; i += 8)
    {
        float16x8_t p = vld1q_f16(ptr);
        p = vdivq_f16(p, vaddq_f16(one, exp_ps_f16(vnegq_f16(p))));
        vst1q_f16(ptr, p);
        ptr += 8;
    }
    for (; i + 3 < elemcount; i += 4)
    {
        float16x4_t p = vld1_f16(ptr);
        p = vdiv_f16(p, vadd_f16(vget_low_f16(one), exp_ps_f16(vneg_f16(p))));
        vst1_f16(ptr, p);
        ptr += 4;
    }
    for (; i < elemcount; i++)
    {
        __fp16 v = *ptr;
        v = v / (__fp16)(1.f + expf(-v));
        *ptr = v;
        ptr++;
    }
}

static void swish_optimized_fp16sa(void* data, int elemcount)
{
    __fp16* ptr = (__fp16*)data;
    int i = 0;

    for (; i + 31 < elemcount; i += 32)
    {
        float16x8_t p0 = vld1q_f16(ptr);
        float16x8_t p1 = vld1q_f16(ptr + 8);
        float16x8_t p2 = vld1q_f16(ptr + 16);
        float16x8_t p3 = vld1q_f16(ptr + 24);
        p0 = vmulq_f16(p0, sigmoid_ps_f16(p0));
        p1 = vmulq_f16(p1, sigmoid_ps_f16(p1));
        p2 = vmulq_f16(p2, sigmoid_ps_f16(p2));
        p3 = vmulq_f16(p3, sigmoid_ps_f16(p3));
        vst1q_f16(ptr, p0);
        vst1q_f16(ptr + 8, p1);
        vst1q_f16(ptr + 16, p2);
        vst1q_f16(ptr + 24, p3);
        ptr += 32;
    }
    for (; i + 15 < elemcount; i += 16)
    {
        float16x8_t p0 = vld1q_f16(ptr);
        float16x8_t p1 = vld1q_f16(ptr + 8);
        p0 = vmulq_f16(p0, sigmoid_ps_f16(p0));
        p1 = vmulq_f16(p1, sigmoid_ps_f16(p1));
        vst1q_f16(ptr, p0);
        vst1q_f16(ptr + 8, p1);
        ptr += 16;
    }
    for (; i + 7 < elemcount; i += 8)
    {
        float16x8_t p = vld1q_f16(ptr);
        p = vmulq_f16(p, sigmoid_ps_f16(p));
        vst1q_f16(ptr, p);
        ptr += 8;
    }
    for (; i + 3 < elemcount; i += 4)
    {
        float16x4_t p = vld1_f16(ptr);
        p = vmul_f16(p, sigmoid_ps_f16(p));
        vst1_f16(ptr, p);
        ptr += 4;
    }
    for (; i < elemcount; i++)
    {
        __fp16 v = *ptr;
        v = v / (__fp16)(1.f + expf(-v));
        *ptr = v;
        ptr++;
    }
}
#endif
#endif

static void fill_fp32(ncnn::Mat& m)
{
    float* ptr = (float*)m;
    for (int i = 0; i < (int)m.total(); i++)
    {
        ptr[i] = (float)((i % 251) - 125) * 0.03125f;
    }
}

#if NCNN_BF16
static void fill_bf16(std::vector<unsigned short>& v)
{
    for (size_t i = 0; i < v.size(); i++)
    {
        float x = (float)((int)(i % 251) - 125) * 0.03125f;
        v[i] = ncnn::float32_to_bfloat16(x);
    }
}
#endif

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static void fill_fp16(std::vector<__fp16>& v)
{
    for (size_t i = 0; i < v.size(); i++)
    {
        v[i] = (__fp16)((int)(i % 251) - 125) * (__fp16)0.03125f;
    }
}
#endif

#if __ARM_NEON
static void check_case_fp32(int w, int h, int c)
{
    ncnn::Mat src(w, h, c);
    fill_fp32(src);

    ncnn::Mat legacy = src.clone();
    ncnn::Mat optimized = src.clone();
    swish_legacy_fp32(legacy.data, (int)legacy.total());
    swish_optimized_fp32(optimized.data, (int)optimized.total());

    char label[128];
    snprintf(label, sizeof(label), "swish-fp32 [%d,%d,%d]", w, h, c);
    fprintf(stdout, "%s\n", label);
    print_diff_result(compare_fp32((const float*)legacy, (const float*)optimized, (int)legacy.total()));
}

#if NCNN_BF16
static void check_case_bf16(int w, int h, int c)
{
    std::vector<unsigned short> src((size_t)w * h * c);
    fill_bf16(src);

    std::vector<unsigned short> legacy(src);
    std::vector<unsigned short> optimized(src);
    swish_legacy_bf16(legacy.data(), (int)legacy.size());
    swish_optimized_bf16(optimized.data(), (int)optimized.size());

    char label[128];
    snprintf(label, sizeof(label), "swish-bf16 [%d,%d,%d]", w, h, c);
    fprintf(stdout, "%s\n", label);
    print_diff_result(compare_bf16(legacy.data(), optimized.data(), (int)legacy.size()));
}
#endif

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static void check_case_fp16sa(int w, int h, int c)
{
    std::vector<__fp16> src((size_t)w * h * c);
    fill_fp16(src);

    std::vector<__fp16> legacy(src);
    std::vector<__fp16> optimized(src);
    swish_legacy_fp16sa(legacy.data(), (int)legacy.size());
    swish_optimized_fp16sa(optimized.data(), (int)optimized.size());

    char label[128];
    snprintf(label, sizeof(label), "swish-fp16sa [%d,%d,%d]", w, h, c);
    fprintf(stdout, "%s\n", label);
    print_diff_result(compare_fp16(legacy.data(), optimized.data(), (int)legacy.size()));
}
#endif

static void bench_case_fp32(int w, int h, int c)
{
    ncnn::Mat src(w, h, c);
    fill_fp32(src);

    char label[128];
    snprintf(label, sizeof(label), "swish-fp32 [%d,%d,%d] legacy", w, h, c);
    PerfResult legacy = run_benchmark(swish_legacy_fp32, src.data, src.total() * sizeof(float), (int)src.total());
    print_perf_result(label, legacy);

    snprintf(label, sizeof(label), "swish-fp32 [%d,%d,%d] optimized", w, h, c);
    PerfResult optimized = run_benchmark(swish_optimized_fp32, src.data, src.total() * sizeof(float), (int)src.total());
    print_perf_result(label, optimized);

    print_speedup(legacy, optimized);
}

#if NCNN_BF16
static void bench_case_bf16(int w, int h, int c)
{
    std::vector<unsigned short> src((size_t)w * h * c);
    fill_bf16(src);

    char label[128];
    snprintf(label, sizeof(label), "swish-bf16 [%d,%d,%d] legacy", w, h, c);
    PerfResult legacy = run_benchmark(swish_legacy_bf16, src.data(), src.size() * sizeof(unsigned short), (int)src.size());
    print_perf_result(label, legacy);

    snprintf(label, sizeof(label), "swish-bf16 [%d,%d,%d] optimized", w, h, c);
    PerfResult optimized = run_benchmark(swish_optimized_bf16, src.data(), src.size() * sizeof(unsigned short), (int)src.size());
    print_perf_result(label, optimized);

    print_speedup(legacy, optimized);
}
#endif

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static void bench_case_fp16sa(int w, int h, int c)
{
    std::vector<__fp16> src((size_t)w * h * c);
    fill_fp16(src);

    char label[128];
    snprintf(label, sizeof(label), "swish-fp16sa [%d,%d,%d] legacy", w, h, c);
    PerfResult legacy = run_benchmark(swish_legacy_fp16sa, src.data(), src.size() * sizeof(__fp16), (int)src.size());
    print_perf_result(label, legacy);

    snprintf(label, sizeof(label), "swish-fp16sa [%d,%d,%d] optimized", w, h, c);
    PerfResult optimized = run_benchmark(swish_optimized_fp16sa, src.data(), src.size() * sizeof(__fp16), (int)src.size());
    print_perf_result(label, optimized);

    print_speedup(legacy, optimized);
}
#endif
#endif

int main()
{
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(1);

#if !__ARM_NEON
    fprintf(stdout, "perf_swish_compare requires ARM NEON\n");
    return 0;
#else
    check_case_fp32(56, 56, 64);
    check_case_fp32(224, 224, 3);
#if NCNN_BF16
    check_case_bf16(56, 56, 64);
    check_case_bf16(224, 224, 3);
#endif
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    check_case_fp16sa(56, 56, 64);
    check_case_fp16sa(224, 224, 3);
#endif

    fprintf(stdout, "\n");

    bench_case_fp32(56, 56, 64);
    bench_case_fp32(28, 28, 128);
    bench_case_fp32(224, 224, 3);
    bench_case_fp32(100000, 1, 1);
#if NCNN_BF16
    bench_case_bf16(56, 56, 64);
    bench_case_bf16(28, 28, 128);
    bench_case_bf16(224, 224, 3);
    bench_case_bf16(100000, 1, 1);
#endif
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    bench_case_fp16sa(56, 56, 64);
    bench_case_fp16sa(28, 28, 128);
    bench_case_fp16sa(224, 224, 3);
    bench_case_fp16sa(100000, 1, 1);
#endif

    return 0;
#endif
}
