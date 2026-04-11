#include "inference_test_case.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

static void show_usage()
{
    fprintf(stderr,
            "Usage: demo_inference_test [options]\n"
            "Options:\n"
            "  --param <path>          model param file (.param)\n"
            "  --bin <path>             model bin file (.bin)\n"
            "  --input <name:file:shape> input spec (repeatable)\n"
            "                           shape: W or W,H or W,H,C\n"
            "                           file: image path or .bin path or empty\n"
            "  --output <name>          output blob name (repeatable)\n"
            "  --loops <N>              test loops (default: 20)\n"
            "  --warmup <N>             warmup loops (default: 5)\n"
            "  --threads <N>            thread count (default: 1)\n"
            "  --backend <cpu|vulkan>   backend (default: cpu)\n"
            "  --save-dir <path>        output directory for results\n"
            "  --help                   show this message\n"
            "\n"
            "Example:\n"
            "  ./demo_inference_test --param model.param --bin model.bin \\\n"
            "    --input in:photo.jpg:224,224,3 --loops 20 --warmup 5 --save-dir ./output\n");
}

static std::vector<int> parse_shape(const char* s)
{
    std::vector<int> shape;
    char* copy = strdup(s);
    char* tok = strtok(copy, ",:");
    while (tok) {
        shape.push_back(atoi(tok));
        tok = strtok(nullptr, ",:");
    }
    free(copy);
    return shape;
}

int main(int argc, char** argv)
{
    RunConfig config;
    config.num_threads = 1;
    config.warmup_loops = 5;
    config.test_loops = 20;
    config.backend = Backend::CPU;
    bool have_param = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            show_usage();
            return 0;
        } else if (strcmp(argv[i], "--param") == 0 && i + 1 < argc) {
            config.param_path = argv[++i];
            have_param = true;
        } else if (strcmp(argv[i], "--bin") == 0 && i + 1 < argc) {
            config.bin_path = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            InputSpec spec;
            spec.name = "input";

            const char* colon1 = strchr(arg, ':');
            if (colon1) {
                std::string name_str(arg, colon1 - arg);
                if (!name_str.empty())
                    spec.name = name_str;

                const char* colon2 = strchr(colon1 + 1, ':');
                if (colon2) {
                    spec.file_path = std::string(colon1 + 1, colon2 - colon1 - 1);
                    spec.shape = parse_shape(colon2 + 1);
                } else {
                    spec.file_path = std::string(colon1 + 1);
                }
            } else {
                spec.file_path = arg;
            }

            config.inputs.push_back(spec);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            OutputSpec spec;
            spec.name = argv[++i];
            config.outputs.push_back(spec);
        } else if (strcmp(argv[i], "--loops") == 0 && i + 1 < argc) {
            config.test_loops = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            config.warmup_loops = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            config.num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            const char* be = argv[++i];
            if (strcmp(be, "vulkan") == 0 || strcmp(be, "gpu") == 0) {
                config.backend = Backend::Vulkan;
            } else {
                config.backend = Backend::CPU;
            }
        } else if (strcmp(argv[i], "--save-dir") == 0 && i + 1 < argc) {
            config.save_dir = argv[++i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            show_usage();
            return 1;
        }
    }

    if (!have_param) {
        fprintf(stderr, "Error: --param is required\n");
        show_usage();
        return 1;
    }

    InferenceTestCase test(config);
    if (test.setup() != 0) {
        fprintf(stderr, "Test setup failed\n");
        return 1;
    }

    if (test.run() != 0) {
        fprintf(stderr, "Test run failed\n");
        return 1;
    }

    fprintf(stderr, "Done.\n");
    return 0;
}
