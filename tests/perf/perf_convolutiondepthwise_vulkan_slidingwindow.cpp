// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "benchmark.h"
#include "command.h"
#include "gpu.h"
#include "layer.h"
#include "modelbin.h"

#include <stdio.h>
#include <vector>

static int benchmark_case(int w, int h, int c)
{
    ncnn::Layer* op_ref = ncnn::create_layer_vulkan("ConvolutionDepthWise");
    ncnn::Layer* op_new = ncnn::create_layer_vulkan("ConvolutionDepthWise");
    if (!op_ref || !op_new)
    {
        delete op_ref;
        delete op_new;
        return -1;
    }

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
    {
        delete op_ref;
        delete op_new;
        return -1;
    }

    op_ref->vkdev = vkdev;
    op_new->vkdev = vkdev;

    ncnn::ParamDict pd;
    pd.set(0, c);
    pd.set(1, 3);
    pd.set(2, 1);
    pd.set(3, 1);
    pd.set(4, 1);
    pd.set(5, 1);
    pd.set(6, c * 3 * 3);
    pd.set(7, c);

    std::vector<ncnn::Mat> weights(2);
    weights[0].create(c * 3 * 3);
    weights[0].fill(0.01f);
    weights[1].create(c);
    weights[1].fill(0.01f);

    ncnn::Option opt_ref;
    opt_ref.lightmode = true;
    opt_ref.use_packing_layout = true;
    opt_ref.use_shader_local_memory = false;
    opt_ref.use_vulkan_compute = true;
    opt_ref.use_fp16_packed = false;
    opt_ref.use_fp16_storage = false;
    opt_ref.use_fp16_arithmetic = false;

    ncnn::Option opt_new = opt_ref;
    opt_new.use_shader_local_memory = true;

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    opt_ref.blob_vkallocator = blob_vkallocator;
    opt_ref.workspace_vkallocator = blob_vkallocator;
    opt_ref.staging_vkallocator = staging_vkallocator;

    opt_new.blob_vkallocator = blob_vkallocator;
    opt_new.workspace_vkallocator = blob_vkallocator;
    opt_new.staging_vkallocator = staging_vkallocator;

    op_ref->load_param(pd);
    op_new->load_param(pd);

    ncnn::ModelBinFromMatArray mb_ref(weights.data());
    ncnn::ModelBinFromMatArray mb_new(weights.data());
    op_ref->load_model(mb_ref);
    op_new->load_model(mb_new);

    op_ref->create_pipeline(opt_ref);
    op_new->create_pipeline(opt_new);

    ncnn::VkWeightAllocator weight_vkallocator(vkdev);
    ncnn::VkWeightStagingAllocator weight_staging_vkallocator(vkdev);

    ncnn::Option opt_upload = opt_ref;
    opt_upload.blob_vkallocator = &weight_vkallocator;
    opt_upload.workspace_vkallocator = &weight_vkallocator;
    opt_upload.staging_vkallocator = &weight_staging_vkallocator;

    {
        ncnn::VkTransfer cmd(vkdev);
        op_ref->upload_model(cmd, opt_upload);
        cmd.submit_and_wait();
    }
    {
        ncnn::VkTransfer cmd(vkdev);
        op_new->upload_model(cmd, opt_upload);
        cmd.submit_and_wait();
    }

    ncnn::Mat input_cpu(w, h, c);
    input_cpu.fill(0.1f);

    ncnn::VkMat input_gpu;
    {
        ncnn::VkCompute cmd(vkdev);
        cmd.record_upload(input_cpu, input_gpu, opt_ref);
        cmd.submit_and_wait();
    }

    const int warmup = 20;
    const int loops = 100;
    const int runs = 20;

    for (int i = 0; i < warmup; i++)
    {
        ncnn::VkMat out_ref;
        ncnn::VkMat out_new;
        ncnn::VkCompute cmd(vkdev);
        op_ref->forward(input_gpu, out_ref, cmd, opt_ref);
        op_new->forward(input_gpu, out_new, cmd, opt_new);
        cmd.submit_and_wait();
    }

    double t_ref_sum = 0.f;
    double t_new_sum = 0.f;

    for (int r = 0; r < runs; r++)
    {
        {
            ncnn::VkCompute cmd(vkdev);
            ncnn::VkMat out_ref;
            for (int i = 0; i < loops; i++)
            {
                op_ref->forward(input_gpu, out_ref, cmd, opt_ref);
            }
            double t0 = ncnn::get_current_time();
            cmd.submit_and_wait();
            double t1 = ncnn::get_current_time();
            t_ref_sum += (t1 - t0) / loops;
        }

        {
            ncnn::VkCompute cmd(vkdev);
            ncnn::VkMat out_new;
            for (int i = 0; i < loops; i++)
            {
                op_new->forward(input_gpu, out_new, cmd, opt_new);
            }
            double t0 = ncnn::get_current_time();
            cmd.submit_and_wait();
            double t1 = ncnn::get_current_time();
            t_new_sum += (t1 - t0) / loops;
        }
    }

    double t_ref = t_ref_sum / runs;
    double t_new = t_new_sum / runs;
    double speedup = t_ref / t_new;

    fprintf(stdout, "dw3x3s1 pack4 %dx%dx%d  baseline=%.4fms  sliding=%.4fms  speedup=%.3fx\n", w, h, c, t_ref, t_new, speedup);

    op_ref->destroy_pipeline(opt_ref);
    op_new->destroy_pipeline(opt_new);
    delete op_ref;
    delete op_new;

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}

int main()
{
    ncnn::create_gpu_instance();

    int gpu_count = ncnn::get_gpu_count();
    if (gpu_count == 0)
    {
        fprintf(stderr, "no vulkan gpu available\n");
        ncnn::destroy_gpu_instance();
        return 0;
    }

    int ret = 0;
    ret |= benchmark_case(112, 112, 32);
    ret |= benchmark_case(56, 56, 64);
    ret |= benchmark_case(28, 28, 128);
    ret |= benchmark_case(14, 14, 256);
    ret |= benchmark_case(7, 7, 512);

    ncnn::destroy_gpu_instance();

    return ret;
}
