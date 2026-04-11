#ifndef DEMO_RUN_CONFIG_H
#define DEMO_RUN_CONFIG_H

#include <string>
#include <vector>

struct TensorShape {
    std::string name;
    std::vector<int> dims;
};

struct InputSpec {
    std::string name;
    std::string file_path;
    std::vector<int> shape;
    std::vector<float> fill_value;
};

struct OutputSpec {
    std::string name;
};

enum class Backend {
    CPU,
    Vulkan
};

struct RunConfig {
    std::string param_path;
    std::string bin_path;
    std::vector<InputSpec> inputs;
    std::vector<OutputSpec> outputs;
    int num_threads = 1;
    int powersave = 2;
    Backend backend = Backend::CPU;
    int gpu_device = 0;
    bool light_mode = true;
    int warmup_loops = 5;
    int test_loops = 20;
    std::string save_dir;
};

#endif // DEMO_RUN_CONFIG_H
