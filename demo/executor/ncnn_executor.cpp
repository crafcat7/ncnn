#include "ncnn_executor.h"

#include "benchmark.h"
#include "cpu.h"
#include "datareader.h"

#include <algorithm>
#include <string.h>

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char* format, void* p) const
    {
        (void)format;
        (void)p;
        return 0;
    }

    virtual size_t read(void* buf, size_t size) const
    {
        memset(buf, 0, size);
        return size;
    }
};

NcnnExecutor::NcnnExecutor()
    : cpu_blob_tracker_(&cpu_blob_allocator_),
      cpu_workspace_tracker_(&cpu_workspace_allocator_)
{
}

NcnnExecutor::~NcnnExecutor()
{
    unload();
}

int NcnnExecutor::load(const RunConfig& config)
{
    unload();

    opt_ = ncnn::Option();
    opt_.num_threads = config.num_threads;
    opt_.lightmode = config.light_mode;

    ncnn::set_cpu_powersave(config.powersave);

    use_vulkan_ = (config.backend == Backend::Vulkan);
    backend_name_ = use_vulkan_ ? "vulkan" : "cpu";

#if NCNN_VULKAN
    if (use_vulkan_) {
        if (!ncnn::get_gpu_count()) {
            fprintf(stderr, "[Executor] Vulkan requested but no GPU available, falling back to CPU\n");
            use_vulkan_ = false;
            backend_name_ = "cpu";
        }
    }
#endif

    if (use_vulkan_) {
#if NCNN_VULKAN
        vkdev_ = ncnn::get_gpu_device(config.gpu_device);
        net_.set_vulkan_device(vkdev_);

        opt_.use_vulkan_compute = true;

        vk_blob_allocator_ = new ncnn::VkBlobAllocator(vkdev_);
        vk_staging_allocator_ = new ncnn::VkStagingAllocator(vkdev_);

        opt_.blob_vkallocator = vk_blob_allocator_;
        opt_.workspace_vkallocator = vk_blob_allocator_;
        opt_.staging_vkallocator = vk_staging_allocator_;
#endif
    } else {
        opt_.use_vulkan_compute = false;
        opt_.blob_allocator = &cpu_blob_tracker_;
        opt_.workspace_allocator = &cpu_workspace_tracker_;
    }

    net_.opt = opt_;

    if (!config.param_path.empty()) {
        if (net_.load_param(config.param_path.c_str()) != 0) {
            fprintf(stderr, "[Executor] Failed to load param: %s\n", config.param_path.c_str());
            return -1;
        }
    }

    if (!config.bin_path.empty()) {
        if (net_.load_model(config.bin_path.c_str()) != 0) {
            fprintf(stderr, "[Executor] Failed to load model: %s\n", config.bin_path.c_str());
            return -1;
        }
    } else {
        DataReaderFromEmpty dr;
        if (net_.load_model(dr) != 0) {
            fprintf(stderr, "[Executor] Failed to load empty model weights for param-only benchmark\n");
            return -1;
        }
    }

    const std::vector<const char*>& net_input_names = net_.input_names();
    const std::vector<const char*>& net_output_names = net_.output_names();

    input_names_.clear();
    for (size_t i = 0; i < net_input_names.size(); i++)
    {
        input_names_.push_back(net_input_names[i]);
    }

    output_names_.clear();
    if (!config.outputs.empty())
    {
        for (size_t i = 0; i < config.outputs.size(); i++)
        {
            output_names_.push_back(config.outputs[i].name);
        }
    }
    else
    {
        for (size_t i = 0; i < net_output_names.size(); i++)
        {
            output_names_.push_back(net_output_names[i]);
        }
    }

    loaded_ = true;
    fprintf(stderr, "[Executor] Loaded: param=%s bin=%s backend=%s inputs=%zu outputs=%zu\n",
            config.param_path.c_str(), config.bin_path.c_str(), backend_name_.c_str(),
            input_names_.size(), output_names_.size());

    return 0;
}

int NcnnExecutor::prepare_inputs(std::vector<TensorData>& inputs)
{
    for (auto& td : inputs) {
        if (td.mat.empty()) {
            fprintf(stderr, "[Executor] Input '%s' is empty\n", td.name.c_str());
            return -1;
        }
    }
    return 0;
}

int NcnnExecutor::extract_outputs(ncnn::Extractor& ex, std::vector<TensorData>& outputs)
{
    if (output_names_.empty()) {
        return 0;
    }

    for (const auto& name : output_names_) {
        ncnn::Mat out;
        int ret = ex.extract(name.c_str(), out);
        if (ret != 0) {
            fprintf(stderr, "[Executor] Failed to extract output '%s'\n", name.c_str());
            return ret;
        }
        TensorData td;
        td.name = name;
        td.mat = out;
        outputs.push_back(std::move(td));
    }
    return 0;
}

int NcnnExecutor::run_once(std::vector<TensorData>& inputs, std::vector<TensorData>& outputs,
                           double* latency_ms, MemoryStats* mem_stats)
{
    if (!loaded_) {
        fprintf(stderr, "[Executor] Not loaded\n");
        return -1;
    }

    if (prepare_inputs(inputs) != 0)
        return -1;

    cpu_blob_tracker_.reset_peak();
    cpu_workspace_tracker_.reset_peak();

    double start = ncnn::get_current_time();

    ncnn::Extractor ex = net_.create_extractor();
    ex.set_light_mode(opt_.lightmode);

    for (size_t i = 0; i < inputs.size(); ++i) {
        const char* input_name = inputs[i].name.empty() ? input_names_[i].c_str() : inputs[i].name.c_str();
        int ret = ex.input(input_name, inputs[i].mat);
        if (ret != 0) {
            fprintf(stderr, "[Executor] Failed to input '%s'\n", input_name);
            return ret;
        }
    }

    outputs.clear();
    int ret = extract_outputs(ex, outputs);

    double end = ncnn::get_current_time();
    *latency_ms = end - start;

    if (mem_stats) {
        MemoryStats s;
        s = cpu_blob_tracker_.get_stats();
        s.peak_bytes += cpu_workspace_tracker_.get_stats().peak_bytes;
        *mem_stats = s;
    }

    return ret;
}

void NcnnExecutor::unload()
{
    if (!loaded_)
        return;

    net_.clear();

#if NCNN_VULKAN
    delete vk_blob_allocator_;
    delete vk_staging_allocator_;
    vk_blob_allocator_ = nullptr;
    vk_staging_allocator_ = nullptr;
    vkdev_ = nullptr;
#endif

    input_names_.clear();
    output_names_.clear();
    loaded_ = false;
}
