#ifndef DEMO_NCNN_EXECUTOR_H
#define DEMO_NCNN_EXECUTOR_H

#include "run_config.h"
#include "memory_tracker.h"
#include "io_utils.h"

#include "allocator.h"
#include "net.h"

#include <string>

#if NCNN_VULKAN
#include "gpu.h"
#endif

class NcnnExecutor {
public:
    NcnnExecutor();
    ~NcnnExecutor();

    int load(const RunConfig& config);
    int run_once(std::vector<TensorData>& inputs, std::vector<TensorData>& outputs,
                 double* latency_ms, MemoryStats* mem_stats);
    void unload();

    const std::vector<std::string>& input_names() const { return input_names_; }
    const std::vector<std::string>& output_names() const { return output_names_; }
    bool is_vulkan() const { return use_vulkan_; }
    const std::string& backend_name() const { return backend_name_; }

private:
    int prepare_inputs(std::vector<TensorData>& inputs);
    int extract_outputs(ncnn::Extractor& ex, std::vector<TensorData>& outputs);

    ncnn::Net net_;
    ncnn::Option opt_;

    ncnn::UnlockedPoolAllocator cpu_blob_allocator_;
    ncnn::PoolAllocator cpu_workspace_allocator_;
    TrackingAllocator cpu_blob_tracker_;
    TrackingAllocator cpu_workspace_tracker_;

#if NCNN_VULKAN
    ncnn::VulkanDevice* vkdev_ = nullptr;
    ncnn::VkBlobAllocator* vk_blob_allocator_ = nullptr;
    ncnn::VkStagingAllocator* vk_staging_allocator_ = nullptr;
#endif

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    bool use_vulkan_ = false;
    std::string backend_name_;
    bool loaded_ = false;
};

#endif // DEMO_NCNN_EXECUTOR_H
