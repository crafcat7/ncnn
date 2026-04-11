#ifndef DEMO_IO_UTILS_H
#define DEMO_IO_UTILS_H

#include <string>
#include <vector>
#include "mat.h"

struct TensorData {
    std::string name;
    ncnn::Mat mat;
};

bool save_tensor_bin(const std::string& path, const ncnn::Mat& mat);
bool load_tensor_bin(const std::string& path, ncnn::Mat& mat);
bool save_tensor_json(const std::string& json_path, const std::string& tensor_name, const std::string& bin_path, const ncnn::Mat& mat);
bool load_tensor_from_image(const std::string& image_path, const std::vector<int>& shape, ncnn::Mat& mat);
bool create_tensor_from_shape(const std::vector<int>& shape, float fill_value, ncnn::Mat& mat);
bool create_tensor_from_file(const std::string& file_path, const std::vector<int>& shape, ncnn::Mat& mat);
bool save_run_stats_json(const std::string& json_path,
                          const std::string& model_name,
                          const std::vector<double>& latencies,
                          size_t peak_memory_bytes,
                          const std::vector<TensorData>& inputs,
                          const std::vector<TensorData>& outputs,
                          const std::string& backend_name);

#endif // DEMO_IO_UTILS_H
