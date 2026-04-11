#ifndef DEMO_INFERENCE_TEST_CASE_H
#define DEMO_INFERENCE_TEST_CASE_H

#include "../executor/run_config.h"
#include "../executor/ncnn_executor.h"
#include "../executor/io_utils.h"

#include <string>
#include <vector>

struct TestReport {
    std::string model_name;
    std::string backend;
    int test_loops;
    double latency_min_ms;
    double latency_max_ms;
    double latency_avg_ms;
    double latency_p50_ms;
    double latency_p90_ms;
    double latency_p99_ms;
    size_t peak_memory_bytes;
    std::vector<double> all_latencies_ms;
};

class InferenceTestCase {
public:
    InferenceTestCase(const RunConfig& config);
    ~InferenceTestCase();

    int setup();
    int run();
    const TestReport& get_report() const { return report_; }
    void print_report() const;

private:
    int build_inputs();
    int save_results();
    int save_inputs_outputs(const std::vector<TensorData>& inputs,
                            const std::vector<TensorData>& outputs);

    RunConfig config_;
    NcnnExecutor executor_;
    std::vector<TensorData> inputs_;
    std::vector<TensorData> outputs_;
    TestReport report_;
};

#endif // DEMO_INFERENCE_TEST_CASE_H
