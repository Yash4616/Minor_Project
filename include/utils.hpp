/*
 * utils.hpp
 * Purpose: Shared utilities, metric structures, and cross-module pipeline APIs.
 */
#pragma once

#include "config.hpp"
#include "dataset.hpp"
#include "detector.hpp"
#include "unet.hpp"

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace change {

struct ClassMetrics {
    int64_t tp = 0;
    int64_t fp = 0;
    int64_t fn = 0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
};

struct DetectionMetrics {
    std::vector<ClassMetrics> per_class;
    int64_t total_gt = 0;
    int64_t total_correct = 0;
    double accuracy = 0.0;
    double overall_precision = 0.0;
    double overall_recall = 0.0;
    double overall_f1 = 0.0;
};

struct SegmentationMetrics {
    std::vector<double> class_iou;
    std::vector<double> class_pixel_accuracy;
    double mean_iou = 0.0;
    double pixel_accuracy = 0.0;
    double foreground_accuracy = 0.0;
    cv::Mat confusion;  // CV_64F [num_classes, num_classes]
};

inline void ensure_directory(const std::filesystem::path& dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        throw std::runtime_error("Failed to create directory: " + dir.string() + " | " + ec.message());
    }
}

inline double compute_iou(const BoundingBox& a,
                          const BoundingBox& b) {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);

    const float iw = std::max(0.0F, ix2 - ix1);
    const float ih = std::max(0.0F, iy2 - iy1);
    const float inter = iw * ih;

    const float area_a = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
    const float area_b = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
    const float denom = area_a + area_b - inter;
    if (denom <= 0.0F) {
        return 0.0;
    }
    return static_cast<double>(inter / denom);
}

inline torch::Tensor load_rgb_image_tensor(const std::filesystem::path& path,
                                           int canvas_size) {
    cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("Failed to read image: " + path.string());
    }
    if (bgr.rows != canvas_size || bgr.cols != canvas_size) {
        cv::resize(bgr, bgr, cv::Size(canvas_size, canvas_size), 0.0, 0.0, cv::INTER_LINEAR);
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat rgb_float;
    rgb.convertTo(rgb_float, CV_32FC3, 1.0 / 255.0);

    auto tensor = torch::from_blob(
        rgb_float.data,
        {rgb_float.rows, rgb_float.cols, 3},
        torch::TensorOptions().dtype(torch::kFloat32));

    tensor = tensor.clone().permute({2, 0, 1});
    return tensor;
}

inline cv::Mat load_rgb_image_mat(const std::filesystem::path& path,
                                  int canvas_size) {
    cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("Failed to read image: " + path.string());
    }
    if (bgr.rows != canvas_size || bgr.cols != canvas_size) {
        cv::resize(bgr, bgr, cv::Size(canvas_size, canvas_size), 0.0, 0.0, cv::INTER_LINEAR);
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

inline cv::Mat load_mask_mat(const std::filesystem::path& path,
                             int canvas_size) {
    cv::Mat mask = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
    if (mask.empty()) {
        throw std::runtime_error("Failed to read mask: " + path.string());
    }
    if (mask.rows != canvas_size || mask.cols != canvas_size) {
        cv::resize(mask, mask, cv::Size(canvas_size, canvas_size), 0.0, 0.0, cv::INTER_NEAREST);
    }
    return mask;
}

inline cv::Mat tensor_mask_to_mat(const torch::Tensor& mask_tensor) {
    auto cpu_mask = mask_tensor.detach().to(torch::kCPU).to(torch::kLong).contiguous();
    const auto h = static_cast<int>(cpu_mask.size(0));
    const auto w = static_cast<int>(cpu_mask.size(1));

    cv::Mat mask(h, w, CV_8UC1);
    auto accessor = cpu_mask.accessor<int64_t, 2>();
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            mask.at<uint8_t>(y, x) = static_cast<uint8_t>(std::clamp<int64_t>(accessor[y][x], 0, 255));
        }
    }
    return mask;
}

inline cv::Mat colorize_mask_hsv(const cv::Mat& mask,
                                 int num_classes) {
    cv::Mat mask_scaled;
    const double scale = (num_classes > 1) ? (255.0 / static_cast<double>(num_classes - 1)) : 255.0;
    mask.convertTo(mask_scaled, CV_8UC1, scale);

    cv::Mat colored_bgr;
    cv::applyColorMap(mask_scaled, colored_bgr, cv::COLORMAP_HSV);

    cv::Mat colored_rgb;
    cv::cvtColor(colored_bgr, colored_rgb, cv::COLOR_BGR2RGB);
    return colored_rgb;
}

inline void save_history_csv(const std::filesystem::path& path,
                             const std::vector<double>& train_losses,
                             const std::vector<double>& val_losses) {
    std::ofstream os(path);
    if (!os) {
        throw std::runtime_error("Failed to write history CSV: " + path.string());
    }
    os << "epoch,train_loss,val_loss\n";
    const size_t n = std::min(train_losses.size(), val_losses.size());
    for (size_t i = 0; i < n; ++i) {
        os << (i + 1) << ',' << train_losses[i] << ',' << val_losses[i] << '\n';
    }
}

DetectionMetrics evaluate_detection(const Config& cfg,
                                    LightweightDetector& detector,
                                    const std::vector<SceneAnnotation>& test_annotations);

SegmentationMetrics evaluate_segmentation(const Config& cfg,
                                          UNet& model);

void print_detection_metrics(const DetectionMetrics& metrics,
                             int num_classes);

void print_segmentation_metrics(const SegmentationMetrics& metrics,
                                int num_classes);

void generate_visualizations(const Config& cfg,
                             LightweightDetector& detector,
                             UNet& unet,
                             const std::vector<SceneAnnotation>& test_annotations,
                             const DetectionMetrics& det_metrics,
                             const SegmentationMetrics& seg_metrics);

void write_final_report(const Config& cfg,
                        const DetectionMetrics& det_metrics,
                        const SegmentationMetrics& seg_metrics);

}  // namespace change
