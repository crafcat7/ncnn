// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layer.h"
#include "testutil.h"

#if NCNN_VULKAN
static int test_convolutiondepthwise_vulkan_slidingwindow_case(int w, int h, int c, int bias)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, c);  // num_output
    pd.set(1, 3);  // kernel_w
    pd.set(2, 1);  // dilation_w
    pd.set(3, 1);  // stride_w
    pd.set(4, 1);  // pad_left
    pd.set(5, bias);
    pd.set(6, c * 3 * 3);
    pd.set(7, c); // group

    int activation_type = RAND() % 7;
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0);
    activation_params[1] = RandomFloat(0, 1);
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(c * 3 * 3);
    weights[1] = RandomMat(c);

    ncnn::Option opt_base;
    opt_base.num_threads = 1;
    opt_base.use_packing_layout = true;
    opt_base.use_vulkan_compute = true;
    opt_base.use_fp16_packed = false;
    opt_base.use_fp16_storage = false;
    opt_base.use_fp16_arithmetic = false;

    ncnn::Option opt_ref = opt_base;
    opt_ref.use_shader_local_memory = false;

    ncnn::Option opt_new = opt_base;
    opt_new.use_shader_local_memory = true;

    int typeindex = ncnn::layer_to_index("ConvolutionDepthWise");
    if (typeindex == -1)
        return -1;

    ncnn::Mat top_ref;
    ncnn::Mat top_new;

    int ret = test_layer_gpu(typeindex, pd, weights, opt_ref, a, top_ref, ncnn::Mat(), 0);
    if (ret != 0)
    {
        fprintf(stderr, "test_layer_gpu opt_ref failed w=%d h=%d c=%d bias=%d act=%d\n", w, h, c, bias, activation_type);
        return ret;
    }

    ret = test_layer_gpu(typeindex, pd, weights, opt_new, a, top_new, ncnn::Mat(), 0);
    if (ret != 0)
    {
        fprintf(stderr, "test_layer_gpu opt_new failed w=%d h=%d c=%d bias=%d act=%d\n", w, h, c, bias, activation_type);
        return ret;
    }

    ret = CompareMat(top_ref, top_new, 0.001f);
    if (ret != 0)
    {
        fprintf(stderr, "CompareMat failed w=%d h=%d c=%d bias=%d act=%d\n", w, h, c, bias, activation_type);
        return ret;
    }

    return 0;
}

static int test_convolutiondepthwise_vulkan_slidingwindow()
{
    int ret = 0
              || test_convolutiondepthwise_vulkan_slidingwindow_case(15, 7, 4, 0)
              || test_convolutiondepthwise_vulkan_slidingwindow_case(15, 7, 4, 1)
              || test_convolutiondepthwise_vulkan_slidingwindow_case(18, 17, 8, 0)
              || test_convolutiondepthwise_vulkan_slidingwindow_case(18, 17, 8, 1)
              || test_convolutiondepthwise_vulkan_slidingwindow_case(25, 33, 16, 0)
              || test_convolutiondepthwise_vulkan_slidingwindow_case(25, 33, 16, 1);

    return ret;
}
#endif // NCNN_VULKAN

int main()
{
    SRAND(7767517);

#if NCNN_VULKAN
    return test_convolutiondepthwise_vulkan_slidingwindow();
#else
    return 0;
#endif
}
