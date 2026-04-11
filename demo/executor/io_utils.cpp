#include "io_utils.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <sys/stat.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <unistd.h>
#endif

#if NCNN_SIMPLEOCV
#include "simpleocv.h"
#endif

static void ensure_dir(const std::string& path)
{
#if defined(_WIN32) || defined(_WIN64)
    CreateDirectoryA(path.c_str(), NULL);
#else
    mkdir(path.c_str(), 0755);
#endif
}

bool save_tensor_bin(const std::string& path, const ncnn::Mat& mat)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open())
        return false;

    const float* data = mat;
    size_t total = mat.total();
    ofs.write(reinterpret_cast<const char*>(data), total * sizeof(float));
    return !ofs.fail();
}

bool load_tensor_bin(const std::string& path, ncnn::Mat& mat)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open())
        return false;

    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    size_t count = size / sizeof(float);
    mat.create(count);
    ifs.read(reinterpret_cast<char*>(mat.data), count * sizeof(float));
    return !ifs.fail();
}

bool save_tensor_json(const std::string& json_path, const std::string& tensor_name, const std::string& bin_path, const ncnn::Mat& mat)
{
    std::ofstream ofs(json_path);
    if (!ofs.is_open())
        return false;

    ofs << "{\n";
    ofs << "  \"bin_file\": \"" << bin_path << "\",\n";
    ofs << "  \"name\": \"" << tensor_name << "\",\n";
    ofs << "  \"w\": " << mat.w << ",\n";
    ofs << "  \"h\": " << mat.h << ",\n";
    ofs << "  \"c\": " << mat.c << ",\n";
    ofs << "  \"dims\": " << mat.dims << ",\n";
    ofs << "  \"elemsize\": " << mat.elemsize << ",\n";
    ofs << "  \"elempack\": " << mat.elempack << ",\n";
    ofs << "  \"total_elements\": " << mat.total() << ",\n";
    ofs << "  \"shape\": [";
    if (mat.dims == 1) {
        ofs << mat.w;
    } else if (mat.dims == 2) {
        ofs << mat.h << ", " << mat.w;
    } else if (mat.dims == 3) {
        ofs << mat.c << ", " << mat.h << ", " << mat.w;
    } else if (mat.dims == 4) {
        ofs << mat.c << ", " << mat.d << ", " << mat.h << ", " << mat.w;
    }
    ofs << "],\n";

    ofs << "  \"sample_values\": [";
    size_t n = mat.total();
    size_t preview = n > 8 ? 8 : n;
    for (size_t i = 0; i < preview; ++i) {
        ofs << std::fixed << std::setprecision(6) << mat[i];
        if (i + 1 < preview)
            ofs << ", ";
    }
    if (n > 8)
        ofs << ", ...";
    ofs << "]\n";
    ofs << "}\n";
    return !ofs.fail();
}

bool load_tensor_from_image(const std::string& image_path, const std::vector<int>& shape, ncnn::Mat& mat)
{
#if NCNN_SIMPLEOCV
    int target_w = 224;
    int target_h = 224;

    if (shape.size() >= 1)
        target_w = shape[0];
    if (shape.size() >= 2)
        target_h = shape[1];

    cv::Mat bgr = cv::imread(image_path, 1);
    if (bgr.empty())
        return false;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB,
                                                 bgr.cols, bgr.rows, target_w, target_h);

    in.substract_mean_normalize(0, 0);

    mat = in;
    return true;
#else
    (void)image_path;
    (void)shape;
    (void)mat;
    return false;
#endif
}

bool create_tensor_from_shape(const std::vector<int>& shape, float fill_value, ncnn::Mat& mat)
{
    if (shape.empty())
        return false;

    if (shape.size() == 1) {
        mat = ncnn::Mat(shape[0]);
    } else if (shape.size() == 2) {
        mat = ncnn::Mat(shape[0], shape[1]);
    } else if (shape.size() == 3) {
        mat = ncnn::Mat(shape[0], shape[1], shape[2]);
    } else if (shape.size() == 4) {
        mat = ncnn::Mat(shape[0], shape[1], shape[2], shape[3]);
    } else {
        return false;
    }

    if (mat.empty())
        return false;

    mat.fill(fill_value);
    return true;
}

bool create_tensor_from_file(const std::string& file_path, const std::vector<int>& shape, ncnn::Mat& mat)
{
    if (file_path.empty())
        return create_tensor_from_shape(shape, 0.01f, mat);

    std::string ext;
    size_t dot = file_path.find_last_of('.');
    if (dot != std::string::npos)
        ext = file_path.substr(dot + 1);

    if (ext == "bin" || ext == "raw") {
        if (!shape.empty())
        {
            if (!create_tensor_from_shape(shape, 0.f, mat))
                return false;

            std::ifstream ifs(file_path, std::ios::binary);
            if (!ifs.is_open())
                return false;

            ifs.read((char*)mat.data, (std::streamsize)(mat.total() * sizeof(float)));
            return ifs.good() || ifs.eof();
        }

        return load_tensor_bin(file_path, mat);
    }

    if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" || ext == "webp") {
        return load_tensor_from_image(file_path, shape, mat);
    }

    return false;
}

bool save_run_stats_json(const std::string& json_path,
                          const std::string& model_name,
                          const std::vector<double>& latencies,
                          size_t peak_memory_bytes,
                          const std::vector<TensorData>& inputs,
                          const std::vector<TensorData>& outputs,
                          const std::string& backend_name)
{
    if (latencies.empty())
        return false;

    ensure_dir(json_path.substr(0, json_path.find_last_of("/\\")));

    std::ofstream ofs(json_path);
    if (!ofs.is_open())
        return false;

    ofs << std::fixed << std::setprecision(4);
    ofs << "{\n";
    ofs << "  \"model\": \"" << model_name << "\",\n";
    ofs << "  \"backend\": \"" << backend_name << "\",\n";
    ofs << "  \"test_loops\": " << latencies.size() << ",\n";

    double sum = 0, min_v = latencies[0], max_v = latencies[0];
    for (size_t i = 0; i < latencies.size(); ++i) {
        sum += latencies[i];
        if (latencies[i] < min_v)
            min_v = latencies[i];
        if (latencies[i] > max_v)
            max_v = latencies[i];
    }
    double avg = sum / latencies.size();

    std::vector<double> sorted = latencies;
    std::sort(sorted.begin(), sorted.end());
    auto percentile = [&](double p) {
        double idx = p * (sorted.size() - 1);
        size_t lo = static_cast<size_t>(idx);
        size_t hi = lo + 1 < sorted.size() ? lo + 1 : lo;
        double frac = idx - lo;
        return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
    };

    ofs << "  \"latency_ms\": {\"min\": " << min_v << ", \"max\": " << max_v
        << ", \"avg\": " << avg << ", \"p50\": " << percentile(0.50)
        << ", \"p90\": " << percentile(0.90)
        << ", \"p99\": " << percentile(0.99) << "},\n";

    ofs << "  \"peak_memory_bytes\": " << peak_memory_bytes << ",\n";

    ofs << "  \"inputs\": [\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        const ncnn::Mat& m = inputs[i].mat;
        ofs << "    {\"name\": \"" << inputs[i].name << "\", \"shape\": [";
        if (m.dims == 1)
            ofs << m.w;
        else if (m.dims == 2)
            ofs << m.h << ", " << m.w;
        else if (m.dims == 3)
            ofs << m.c << ", " << m.h << ", " << m.w;
        else if (m.dims == 4)
            ofs << m.c << ", " << m.d << ", " << m.h << ", " << m.w;
        ofs << "], \"total_elements\": " << m.total() << "}";
        if (i + 1 < inputs.size())
            ofs << ",";
        ofs << "\n";
    }
    ofs << "  ],\n";

    ofs << "  \"outputs\": [\n";
    for (size_t i = 0; i < outputs.size(); ++i) {
        const ncnn::Mat& m = outputs[i].mat;
        ofs << "    {\"name\": \"" << outputs[i].name << "\", \"shape\": [";
        if (m.dims == 1)
            ofs << m.w;
        else if (m.dims == 2)
            ofs << m.h << ", " << m.w;
        else if (m.dims == 3)
            ofs << m.c << ", " << m.h << ", " << m.w;
        else if (m.dims == 4)
            ofs << m.c << ", " << m.d << ", " << m.h << ", " << m.w;
        ofs << "], \"total_elements\": " << m.total() << "}";
        if (i + 1 < outputs.size())
            ofs << ",";
        ofs << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";
    return !ofs.fail();
}
