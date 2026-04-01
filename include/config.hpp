/*
 * config.hpp
 * Purpose: Centralized configuration, constants, and runtime environment setup.
 */
#pragma once

#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>

namespace change {

struct AnchorWH {
    float w;
    float h;
};

struct Config {
    // SECTION 1 -- CONFIGURATION
    // Notebook-mirrored constants.
    static constexpr int64_t kSeed = 42;
    static constexpr int kCanvasSize = 128;
    static constexpr int kDigitSize = 28;
    static constexpr int kMinDigits = 2;
    static constexpr int kMaxDigits = 2;
    static constexpr float kPixelThresh = 0.3F;

    static constexpr int kNumTrain = 100000;
    static constexpr int kNumVal = 10000;
    static constexpr int kNumTest = 10000;

    static constexpr int kNumClassesDet = 10;
    static constexpr int kNumClassesSeg = 11;

    static constexpr int kUnetEpochs = 20;
    static constexpr double kUnetLr = 1.0e-3;
    static constexpr int kUnetPatience = 5;

    // Detector-side equivalent to the notebook YOLO section.
    static constexpr int kDetectorEpochs = 50;
    static constexpr int kDetectorPatience = 10;
    static constexpr int kDetectorGrid = 8;
    static constexpr int kDetectorNumAnchors = 2;
    static constexpr float kDetectorConfThresh = 0.25F;
    static constexpr float kDetectorIouThresh = 0.5F;
    static constexpr double kDetectorLr = 1.0e-3;

    // Augmentation and synthesis constants.
    static constexpr float kAugScaleMin = 0.8F;
    static constexpr float kAugScaleMax = 1.2F;
    static constexpr float kAugRotateDeg = 15.0F;
    static constexpr int kAugMinDigitSize = 20;
    static constexpr int kAugMaxDigitSize = 40;
    static constexpr float kBackgroundColorMax = 0.4F;
    static constexpr float kBackgroundNoiseStd = 0.05F;
    static constexpr float kDigitColorMin = 0.5F;
    static constexpr float kDigitColorMax = 1.0F;

    // Optimizer/scheduler constants.
    static constexpr double kWeightDecay = 1.0e-4;
    static constexpr double kReduceLrFactor = 0.5;
    static constexpr int kReduceLrPatience = 3;
    static constexpr double kEarlyStopMinDelta = 1.0e-4;

    // Loss balancing for detection.
    static constexpr double kLossBbox = 5.0;
    static constexpr double kLossObjectness = 1.0;
    static constexpr double kLossClass = 1.0;

    // Runtime values (initialized from constants).
    int64_t seed = kSeed;
    int canvas_size = kCanvasSize;
    int digit_size = kDigitSize;
    int min_digits = kMinDigits;
    int max_digits = kMaxDigits;
    float pixel_thresh = kPixelThresh;

    int num_train = kNumTrain;
    int num_val = kNumVal;
    int num_test = kNumTest;

    int num_classes_det = kNumClassesDet;
    int num_classes_seg = kNumClassesSeg;

    int unet_epochs = kUnetEpochs;
    double unet_lr = kUnetLr;
    int unet_patience = kUnetPatience;

    int detector_epochs = kDetectorEpochs;
    int detector_patience = kDetectorPatience;
    int detector_grid = kDetectorGrid;
    int detector_num_anchors = kDetectorNumAnchors;
    float detector_conf_thresh = kDetectorConfThresh;
    float detector_iou_thresh = kDetectorIouThresh;
    double detector_lr = kDetectorLr;

    float aug_scale_min = kAugScaleMin;
    float aug_scale_max = kAugScaleMax;
    float aug_rotate_deg = kAugRotateDeg;
    int aug_min_digit_size = kAugMinDigitSize;
    int aug_max_digit_size = kAugMaxDigitSize;

    float background_color_max = kBackgroundColorMax;
    float background_noise_std = kBackgroundNoiseStd;
    float digit_color_min = kDigitColorMin;
    float digit_color_max = kDigitColorMax;

    double weight_decay = kWeightDecay;
    double reduce_lr_factor = kReduceLrFactor;
    int reduce_lr_patience = kReduceLrPatience;
    double early_stop_min_delta = kEarlyStopMinDelta;

    double loss_bbox = kLossBbox;
    double loss_objectness = kLossObjectness;
    double loss_class = kLossClass;

    // Batch sizes are selected at runtime by device availability.
    int detector_batch = 16;
    int unet_batch = 16;

    int num_workers = 0;

    bool use_cuda = false;
    int gpu_count = 0;
    bool enable_amp = true;
    torch::Device device = torch::kCPU;
    torch::Device detector_device = torch::kCPU;
    torch::Device unet_device = torch::kCPU;
    bool train_models_in_parallel = false;

    std::array<AnchorWH, 2> detector_anchors{{AnchorWH{0.18F, 0.18F}, AnchorWH{0.32F, 0.32F}}};

    // Paths.
    std::filesystem::path data_dir;
    std::filesystem::path output_dir;
    std::filesystem::path dataset_dir;

    std::filesystem::path mnist_train_images;
    std::filesystem::path mnist_train_labels;
    std::filesystem::path mnist_test_images;
    std::filesystem::path mnist_test_labels;

    std::filesystem::path detector_checkpoint;
    std::filesystem::path unet_checkpoint;

    std::filesystem::path detector_history_csv;
    std::filesystem::path unet_history_csv;

    std::filesystem::path report_txt;

    std::filesystem::path samples_png;
    std::filesystem::path detector_vis_png;
    std::filesystem::path segmentation_vis_png;
    std::filesystem::path dashboard_png;
};

inline std::string device_to_string(const torch::Device& device) {
    return device.is_cuda() ? "cuda" : "cpu";
}

inline std::string device_with_index_to_string(const torch::Device& device) {
    if (!device.is_cuda()) {
        return "cpu";
    }
    return "cuda:" + std::to_string(device.index());
}

inline Config make_config(const std::filesystem::path& data_dir,
                          const std::filesystem::path& output_dir) {
    Config cfg;

    cfg.data_dir = data_dir;
    cfg.output_dir = output_dir;
    cfg.dataset_dir = cfg.output_dir / "digit_dataset";

    cfg.mnist_train_images = cfg.data_dir / "train-images-idx3-ubyte";
    cfg.mnist_train_labels = cfg.data_dir / "train-labels-idx1-ubyte";
    cfg.mnist_test_images = cfg.data_dir / "t10k-images-idx3-ubyte";
    cfg.mnist_test_labels = cfg.data_dir / "t10k-labels-idx1-ubyte";

    cfg.detector_checkpoint = cfg.output_dir / "detector_best.pt";
    cfg.unet_checkpoint = cfg.output_dir / "unet_best.pt";

    cfg.detector_history_csv = cfg.output_dir / "detector_history.csv";
    cfg.unet_history_csv = cfg.output_dir / "unet_history.csv";

    cfg.report_txt = cfg.output_dir / "report.txt";

    cfg.samples_png = cfg.output_dir / "mnistdd_rgb_samples.png";
    cfg.detector_vis_png = cfg.output_dir / "yolo_detection_results.png";
    cfg.segmentation_vis_png = cfg.output_dir / "unet_segmentation_results.png";
    cfg.dashboard_png = cfg.output_dir / "final_results_dashboard.png";

    cfg.use_cuda = torch::cuda::is_available();
    cfg.device = cfg.use_cuda ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
    cfg.detector_device = cfg.device;
    cfg.unet_device = cfg.device;
    cfg.gpu_count = 0;
    cfg.train_models_in_parallel = false;

    if (cfg.use_cuda) {
        const auto n_gpus = torch::cuda::device_count();
        cfg.gpu_count = static_cast<int>(n_gpus);
        if (n_gpus >= 2) {
            cfg.detector_batch = 256;
            cfg.unet_batch = 256;
            cfg.detector_device = torch::Device(torch::kCUDA, 0);
            cfg.unet_device = torch::Device(torch::kCUDA, 1);
            cfg.train_models_in_parallel = true;
        } else {
            cfg.detector_batch = 64;
            cfg.unet_batch = 128;
            cfg.detector_device = torch::Device(torch::kCUDA, 0);
            cfg.unet_device = torch::Device(torch::kCUDA, 0);
        }
    } else {
        cfg.detector_batch = 16;
        cfg.unet_batch = 16;
        cfg.detector_device = torch::Device(torch::kCPU);
        cfg.unet_device = torch::Device(torch::kCPU);
    }

    return cfg;
}

}  // namespace change
