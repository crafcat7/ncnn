// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_COMMAND_H
#define NCNN_COMMAND_H

#include "platform.h"

#if NCNN_VULKAN

#include "mat.h"

#include <cstdint>

namespace ncnn {

class Pipeline;
#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 26
class ImportAndroidHardwareBufferPipeline;
#endif // __ANDROID_API__ >= 26
#endif // NCNN_PLATFORM_API
struct NCNN_EXPORT VkComputeCommandStatistics
{
    uint64_t dispatches;
    uint64_t pipeline_binds;
    uint64_t redundant_pipeline_binds;
    uint64_t descriptor_bindings;
    uint64_t push_constant_updates;
    uint64_t resource_barrier_calls;
    uint64_t buffer_resource_barriers;
    uint64_t image_resource_barriers;

    VkComputeCommandStatistics();
};

#define NCNN_VK_COMPUTE_OPT_PIPELINE_BIND_ELISION_BIT    0
#define NCNN_VK_COMPUTE_OPT_READONLY_BINDINGS_BIT        1
#define NCNN_VK_COMPUTE_OPT_BATCH_BUFFER_BARRIERS_BIT    2
#define NCNN_VK_COMPUTE_OPT_STACK_DESCRIPTOR_PAYLOAD_BIT 3

enum VkComputeOptimizationFlag : uint32_t
{
    VkComputeOptimizationPipelineBindElision = UINT32_C(1) << NCNN_VK_COMPUTE_OPT_PIPELINE_BIND_ELISION_BIT,
    VkComputeOptimizationReadonlyBindings = UINT32_C(1) << NCNN_VK_COMPUTE_OPT_READONLY_BINDINGS_BIT,
    VkComputeOptimizationBatchBufferBarriers = UINT32_C(1) << NCNN_VK_COMPUTE_OPT_BATCH_BUFFER_BARRIERS_BIT,
    VkComputeOptimizationStackDescriptorPayload = UINT32_C(1) << NCNN_VK_COMPUTE_OPT_STACK_DESCRIPTOR_PAYLOAD_BIT
};

class VkComputePrivate;
class NCNN_EXPORT VkCompute
{
public:
    explicit VkCompute(const VulkanDevice* vkdev);
    VkCompute(const VulkanDevice* vkdev, uint32_t optimization_flags);
    virtual ~VkCompute();

public:
    void record_upload(const Mat& src, VkMat& dst, const Option& opt);

    void record_download(const VkMat& src, Mat& dst, const Option& opt);

    void record_clone(const Mat& src, VkMat& dst, const Option& opt);

    void record_clone(const Mat& src, VkImageMat& dst, const Option& opt);

    void record_clone(const VkMat& src, Mat& dst, const Option& opt);

    void record_clone(const VkImageMat& src, Mat& dst, const Option& opt);

    void record_clone(const VkMat& src, VkMat& dst, const Option& opt);

    void record_clone(const VkImageMat& src, VkImageMat& dst, const Option& opt);

    void record_clone(const VkMat& src, VkImageMat& dst, const Option& opt);

    void record_clone(const VkImageMat& src, VkMat& dst, const Option& opt);

    void record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& bindings, const std::vector<vk_constant_type>& constants, const VkMat& dispatcher);

    // Record a buffer-only pipeline while preserving the access contract for
    // immutable/read-only storage-buffer bindings.  Entries in
    // readonly_buffer_bindings correspond to buffer_bindings and are non-zero
    // for bindings that the shader only reads.
    void record_pipeline_readonly(const Pipeline* pipeline, const std::vector<VkMat>& buffer_bindings, const std::vector<unsigned char>& readonly_buffer_bindings, const std::vector<vk_constant_type>& constants, const VkMat& dispatcher);

    void record_pipeline(const Pipeline* pipeline, const std::vector<VkImageMat>& bindings, const std::vector<vk_constant_type>& constants, const VkImageMat& dispatcher);

    void record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& buffer_bindings, const std::vector<VkImageMat>& image_bindings, const std::vector<vk_constant_type>& constants, const VkMat& dispatcher);
    void record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& buffer_bindings, const std::vector<VkImageMat>& image_bindings, const std::vector<vk_constant_type>& constants, const VkImageMat& dispatcher);
    void record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& buffer_bindings, const std::vector<VkImageMat>& image_bindings, const std::vector<vk_constant_type>& constants, const Mat& dispatcher);

#if NCNN_BENCHMARK
    void record_write_timestamp(uint32_t query);
#endif // NCNN_BENCHMARK

#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 26
    void record_import_android_hardware_buffer(const ImportAndroidHardwareBufferPipeline* pipeline, const VkImageMat& src, const VkMat& dst);
#endif // __ANDROID_API__ >= 26
#endif // NCNN_PLATFORM_API

    int submit();

    int wait();

    int submit_and_wait();

    int reset();

    uint64_t pending_dispatch_total() const;

    VkComputeCommandStatistics command_statistics() const;

#if NCNN_BENCHMARK
    int create_query_pool(uint32_t query_count);

    int get_query_pool_results(uint32_t first_query, uint32_t query_count, std::vector<uint64_t>& results);
#endif // NCNN_BENCHMARK

protected:
    const VulkanDevice* vkdev;

    void barrier_readwrite(const VkMat& binding);
    void barrier_readonly(const VkMat& binding);
    void barrier_readwrite(const VkImageMat& binding);
    void barrier_readonly(const VkImageMat& binding);

private:
    void record_pipeline_impl(const Pipeline* pipeline, const std::vector<VkMat>& buffer_bindings, const std::vector<VkImageMat>& image_bindings, const std::vector<vk_constant_type>& constants, const Mat& dispatcher, const std::vector<unsigned char>* readonly_buffer_bindings);

    VkComputePrivate* const d;
};

class VkTransferPrivate;
class NCNN_EXPORT VkTransfer
{
public:
    explicit VkTransfer(const VulkanDevice* vkdev);
    virtual ~VkTransfer();

public:
    void record_upload(const Mat& src, VkMat& dst, const Option& opt, bool flatten = true);

    int submit_and_wait();

    int reset();

    uint64_t pending_upload_total() const;

protected:
    const VulkanDevice* vkdev;

private:
    VkTransferPrivate* const d;
};

} // namespace ncnn

#endif // NCNN_VULKAN

#endif // NCNN_COMMAND_H
