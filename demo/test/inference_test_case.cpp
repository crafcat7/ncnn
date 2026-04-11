#include "inference_test_case.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sys/stat.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <unistd.h>
#endif

static void ensure_dir(const std::string& path)
{
    if (path.empty())
        return;
    size_t pos = path.find_last_of("/\\");
    if (pos != std::string::npos) {
        ensure_dir(path.substr(0, pos));
    }
#if defined(_WIN32) || defined(_WIN64)
    CreateDirectoryA(path.c_str(), NULL);
#else
    mkdir(path.c_str(), 0755);
#endif
}

static std::string get_model_name(const std::string& param_path)
{
    size_t pos = param_path.find_last_of("/\\");
    std::string name = (pos != std::string::npos) ? param_path.substr(pos + 1) : param_path;
    size_t dot = name.rfind(".param");
    if (dot != std::string::npos)
        name = name.substr(0, dot);
    return name;
}

static double percentile(std::vector<double>& arr, double p)
{
    if (arr.empty())
        return 0.0;
    std::sort(arr.begin(), arr.end());
    double idx = p * (arr.size() - 1);
    size_t lo = static_cast<size_t>(idx);
    size_t hi = (lo + 1 < arr.size()) ? lo + 1 : lo;
    double frac = idx - lo;
    return arr[lo] * (1.0 - frac) + arr[hi] * frac;
}

InferenceTestCase::InferenceTestCase(const RunConfig& config)
    : config_(config)
{
    report_.model_name = get_model_name(config_.param_path);
    report_.backend = (config_.backend == Backend::Vulkan) ? "vulkan" : "cpu";
    report_.test_loops = 0;
    report_.latency_min_ms = 0;
    report_.latency_max_ms = 0;
    report_.latency_avg_ms = 0;
    report_.latency_p50_ms = 0;
    report_.latency_p90_ms = 0;
    report_.latency_p99_ms = 0;
    report_.peak_memory_bytes = 0;
}

InferenceTestCase::~InferenceTestCase() = default;

int InferenceTestCase::setup()
{
    if (!config_.save_dir.empty()) {
        ensure_dir(config_.save_dir);
    }

    if (executor_.load(config_) != 0)
        return -1;

    return build_inputs();
}

int InferenceTestCase::build_inputs()
{
    if (!config_.inputs.empty()) {
        for (const auto& spec : config_.inputs) {
            ncnn::Mat mat;
            float fill = spec.fill_value.empty() ? 0.01f : spec.fill_value[0];

            if (!spec.file_path.empty()) {
                if (!create_tensor_from_file(spec.file_path, spec.shape, mat)) {
                    if (!create_tensor_from_shape(spec.shape, fill, mat)) {
                        fprintf(stderr, "[Test] Failed to create input for '%s'\n", spec.name.c_str());
                        return -1;
                    }
                }
            } else {
                if (!create_tensor_from_shape(spec.shape, fill, mat)) {
                    fprintf(stderr, "[Test] Failed to create shape for '%s'\n", spec.name.c_str());
                    return -1;
                }
            }

            TensorData td;
            td.name = spec.name;
            td.mat = mat;
            inputs_.push_back(std::move(td));
        }
    } else {
        for (const auto& name : executor_.input_names()) {
            TensorData td;
            td.name = name;
            td.mat = ncnn::Mat(1, 1, 3);
            td.mat.fill(0.01f);
            inputs_.push_back(std::move(td));
        }
    }
    return 0;
}

int InferenceTestCase::run()
{
    std::vector<double> latencies;
    size_t peak_mem = 0;
    MemoryStats mem_stats;

    for (int i = 0; i < config_.warmup_loops; ++i) {
        std::vector<TensorData> dummy_out;
        double lat;
        int ret = executor_.run_once(inputs_, dummy_out, &lat, nullptr);
        if (ret != 0) {
            fprintf(stderr, "[Test] Warmup loop %d failed\n", i);
            return ret;
        }
    }

    for (int i = 0; i < config_.test_loops; ++i) {
        std::vector<TensorData> out;
        double lat;
        int ret = executor_.run_once(inputs_, out, &lat, &mem_stats);
        if (ret != 0) {
            fprintf(stderr, "[Test] Test loop %d failed\n", i);
            return ret;
        }
        latencies.push_back(lat);
        if (mem_stats.peak_bytes > peak_mem)
            peak_mem = mem_stats.peak_bytes;

        if (!config_.save_dir.empty() && i == 0) {
            outputs_ = out;
        }

        fprintf(stderr, "  loop %3d: %.2f ms  peak_mem: %.2f KB\n",
                i, lat, mem_stats.peak_bytes / 1024.0);
    }

    report_.all_latencies_ms = latencies;
    report_.test_loops = (int)latencies.size();
    report_.peak_memory_bytes = peak_mem;

    double sum = 0;
    report_.latency_min_ms = latencies[0];
    report_.latency_max_ms = latencies[0];
    for (double lat : latencies) {
        sum += lat;
        report_.latency_min_ms = std::min(report_.latency_min_ms, lat);
        report_.latency_max_ms = std::max(report_.latency_max_ms, lat);
    }
    report_.latency_avg_ms = sum / latencies.size();

    std::vector<double> sorted_lat = latencies;
    report_.latency_p50_ms = percentile(sorted_lat, 0.50);
    report_.latency_p90_ms = percentile(sorted_lat, 0.90);
    report_.latency_p99_ms = percentile(sorted_lat, 0.99);

    if (!config_.save_dir.empty()) {
        save_inputs_outputs(inputs_, outputs_);
        save_results();
    }

    return 0;
}

int InferenceTestCase::save_results()
{
    std::string json_path = config_.save_dir + "/stats.json";
    std::string model_name = get_model_name(config_.param_path);

    save_run_stats_json(json_path, model_name,
                        report_.all_latencies_ms,
                        report_.peak_memory_bytes,
                        inputs_, outputs_,
                        report_.backend);

    fprintf(stderr, "\n=== Report ===\n");
    print_report();
    fprintf(stderr, "\nResults saved to: %s\n", config_.save_dir.c_str());
    return 0;
}

int InferenceTestCase::save_inputs_outputs(const std::vector<TensorData>& inputs,
                                          const std::vector<TensorData>& outputs)
{
    if (config_.save_dir.empty())
        return 0;

    for (size_t i = 0; i < inputs.size(); ++i) {
        char buf[256];
        snprintf(buf, sizeof(buf), "/input_%zu.bin", i);
        save_tensor_bin(config_.save_dir + buf, inputs[i].mat);

        snprintf(buf, sizeof(buf), "/input_%zu.json", i);
        save_tensor_json(config_.save_dir + buf, inputs[i].name, config_.save_dir + "/input_0.bin",
                         inputs[i].mat);
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
        char buf[256];
        snprintf(buf, sizeof(buf), "/output_%zu.bin", i);
        save_tensor_bin(config_.save_dir + buf, outputs[i].mat);

        snprintf(buf, sizeof(buf), "/output_%zu.json", i);
        save_tensor_json(config_.save_dir + buf, outputs[i].name, config_.save_dir + "/output_0.bin",
                         outputs[i].mat);
    }
    return 0;
}

void InferenceTestCase::print_report() const
{
    fprintf(stderr,
            "  Model:      %s\n"
            "  Backend:    %s\n"
            "  Loops:      %d\n"
            "  Latency:    min=%.2f  max=%.2f  avg=%.2f  p50=%.2f  p90=%.2f  p99=%.2f ms\n"
            "  Peak Memory: %.2f KB\n",
            report_.model_name.c_str(),
            report_.backend.c_str(),
            report_.test_loops,
            report_.latency_min_ms,
            report_.latency_max_ms,
            report_.latency_avg_ms,
            report_.latency_p50_ms,
            report_.latency_p90_ms,
            report_.latency_p99_ms,
            report_.peak_memory_bytes / 1024.0);
}
