/*
 * visualize.cpp
 * Purpose: SECTION 7 -- visualization utilities using OpenCV (boxes, masks, dashboard).
 */

#include "utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace change {
namespace {

std::vector<cv::Scalar> class_palette() {
    return {
        cv::Scalar(255, 99, 71),
        cv::Scalar(30, 144, 255),
        cv::Scalar(60, 179, 113),
        cv::Scalar(255, 215, 0),
        cv::Scalar(199, 21, 133),
        cv::Scalar(72, 209, 204),
        cv::Scalar(255, 140, 0),
        cv::Scalar(123, 104, 238),
        cv::Scalar(220, 20, 60),
        cv::Scalar(0, 191, 255)
    };
}

cv::Mat resize_to_height(const cv::Mat& image,
                         int target_height) {
    if (image.empty()) {
        return image;
    }
    if (image.rows == target_height) {
        return image.clone();
    }
    const double scale = static_cast<double>(target_height) / static_cast<double>(image.rows);
    const int width = std::max(1, static_cast<int>(std::round(image.cols * scale)));
    cv::Mat out;
    cv::resize(image, out, cv::Size(width, target_height), 0.0, 0.0, cv::INTER_LINEAR);
    return out;
}

cv::Mat build_tiled_grid(const std::vector<cv::Mat>& tiles,
                         int columns,
                         int tile_size) {
    if (tiles.empty()) {
        return cv::Mat();
    }

    std::vector<cv::Mat> resized;
    resized.reserve(tiles.size());
    for (const auto& t : tiles) {
        cv::Mat r;
        cv::resize(t, r, cv::Size(tile_size, tile_size), 0.0, 0.0, cv::INTER_LINEAR);
        resized.push_back(r);
    }

    std::vector<cv::Mat> rows;
    for (size_t i = 0; i < resized.size(); i += static_cast<size_t>(columns)) {
        std::vector<cv::Mat> row_tiles;
        for (size_t j = i; j < std::min(resized.size(), i + static_cast<size_t>(columns)); ++j) {
            row_tiles.push_back(resized[j]);
        }
        while (row_tiles.size() < static_cast<size_t>(columns)) {
            row_tiles.push_back(cv::Mat(tile_size, tile_size, CV_8UC3, cv::Scalar(25, 25, 25)));
        }
        cv::Mat row;
        cv::hconcat(row_tiles, row);
        rows.push_back(row);
    }

    cv::Mat grid;
    cv::vconcat(rows, grid);
    return grid;
}

void draw_box(cv::Mat& image,
              const BoundingBox& box,
              const cv::Scalar& color,
              const std::string& text) {
    const cv::Point p1(static_cast<int>(std::round(box.x1)), static_cast<int>(std::round(box.y1)));
    const cv::Point p2(static_cast<int>(std::round(box.x2)), static_cast<int>(std::round(box.y2)));

    cv::rectangle(image, p1, p2, color, 2, cv::LINE_AA);

    const int base_y = std::max(12, p1.y - 4);
    cv::putText(image,
                text,
                cv::Point(p1.x, base_y),
                cv::FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv::LINE_AA);
}

cv::Mat make_summary_panel(const Config& cfg,
                           const DetectionMetrics& det_metrics,
                           const SegmentationMetrics& seg_metrics,
                           int width) {
    cv::Mat panel(230, width, CV_8UC3, cv::Scalar(240, 244, 255));

    int y = 34;
    const int dy = 30;

    cv::putText(panel,
                "Final Results Summary",
                cv::Point(24, y),
                cv::FONT_HERSHEY_SIMPLEX,
                0.9,
                cv::Scalar(35, 55, 95),
                2,
                cv::LINE_AA);
    y += dy + 8;

    cv::putText(panel,
                "Detection (Custom Detector)",
                cv::Point(24, y),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(40, 75, 130),
                2,
                cv::LINE_AA);
    y += dy;

    cv::putText(panel,
                "Accuracy=" + std::to_string(det_metrics.accuracy).substr(0, 6) +
                    "  Precision=" + std::to_string(det_metrics.overall_precision).substr(0, 6) +
                    "  Recall=" + std::to_string(det_metrics.overall_recall).substr(0, 6) +
                    "  F1=" + std::to_string(det_metrics.overall_f1).substr(0, 6),
                cv::Point(24, y),
                cv::FONT_HERSHEY_SIMPLEX,
                0.58,
                cv::Scalar(10, 10, 10),
                1,
                cv::LINE_AA);
    y += dy + 6;

    cv::putText(panel,
                "Segmentation (U-Net)",
                cv::Point(24, y),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(40, 75, 130),
                2,
                cv::LINE_AA);
    y += dy;

    cv::putText(panel,
                "Foreground Acc=" + std::to_string(seg_metrics.foreground_accuracy).substr(0, 6) +
                    "  Pixel Acc=" + std::to_string(seg_metrics.pixel_accuracy).substr(0, 6) +
                    "  Mean IoU=" + std::to_string(seg_metrics.mean_iou).substr(0, 6),
                cv::Point(24, y),
                cv::FONT_HERSHEY_SIMPLEX,
                0.58,
                cv::Scalar(10, 10, 10),
                1,
                cv::LINE_AA);

    cv::putText(panel,
                "Canvas=" + std::to_string(cfg.canvas_size) + "x" + std::to_string(cfg.canvas_size) +
                    "  Classes(det/seg)=" + std::to_string(cfg.num_classes_det) + "/" + std::to_string(cfg.num_classes_seg),
                cv::Point(24, panel.rows - 16),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(60, 60, 60),
                1,
                cv::LINE_AA);

    return panel;
}

}  // namespace

void generate_visualizations(const Config& cfg,
                             LightweightDetector& detector,
                             UNet& unet,
                             const std::vector<SceneAnnotation>& test_annotations,
                             const DetectionMetrics& det_metrics,
                             const SegmentationMetrics& seg_metrics) {
    // SECTION 7 -- VISUALISATIONS
    const auto test_images = list_split_images(cfg, "test");
    if (test_images.size() != test_annotations.size()) {
        throw std::runtime_error("Test image/annotation mismatch in visualization");
    }

    detector->to(cfg.device);
    detector->eval();
    unet->to(cfg.device);
    unet->eval();

    const auto palette = class_palette();

    // 7.1 Detection visualizations.
    std::vector<size_t> all_ids(test_images.size());
    std::iota(all_ids.begin(), all_ids.end(), 0);

    std::mt19937 rng(static_cast<uint32_t>(cfg.seed + 999));
    std::shuffle(all_ids.begin(), all_ids.end(), rng);

    const size_t det_count = std::min<size_t>(8, all_ids.size());
    std::vector<size_t> det_ids(all_ids.begin(), all_ids.begin() + det_count);

    std::vector<cv::Mat> detection_tiles;
    detection_tiles.reserve(det_ids.size());

    torch::NoGradGuard no_grad;

    for (const auto idx : det_ids) {
        cv::Mat image = load_rgb_image_mat(test_images[idx], cfg.canvas_size);
        const auto input = load_rgb_image_tensor(test_images[idx], cfg.canvas_size).unsqueeze(0).to(cfg.device);

        const auto output = detector->forward(input);
        const auto preds = decode_detector_predictions(output, cfg, cfg.detector_conf_thresh);

        cv::Mat vis = image.clone();

        for (const auto& gt : test_annotations[idx].boxes) {
            draw_box(vis, gt, cv::Scalar(0, 255, 0), "GT:" + std::to_string(gt.label));
        }

        for (const auto& pred : preds) {
            const auto color = palette[static_cast<size_t>(pred.label) % palette.size()];
            const std::string text = "P:" + std::to_string(pred.label) + " " + std::to_string(pred.score).substr(0, 4);
            draw_box(vis, pred, color, text);
        }

        cv::putText(vis,
                    "Test #" + std::to_string(idx),
                    cv::Point(8, vis.rows - 8),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(255, 255, 255),
                    1,
                    cv::LINE_AA);

        detection_tiles.push_back(vis);
    }

    cv::Mat detection_grid = build_tiled_grid(detection_tiles, 4, 256);
    if (!detection_grid.empty()) {
        cv::Mat bgr;
        cv::cvtColor(detection_grid, bgr, cv::COLOR_RGB2BGR);
        cv::imwrite(cfg.detector_vis_png.string(), bgr);
    }

    // 7.2 Segmentation visualizations.
    const size_t seg_count = std::min<size_t>(4, det_ids.size());
    std::vector<cv::Mat> seg_rows;

    for (size_t i = 0; i < seg_count; ++i) {
        const size_t idx = det_ids[i];
        const auto image_path = test_images[idx];
        const auto mask_path = cfg.dataset_dir / "masks" / "test" / image_path.filename();

        cv::Mat image = load_rgb_image_mat(image_path, cfg.canvas_size);
        cv::Mat gt_mask = load_mask_mat(mask_path, cfg.canvas_size);

        const auto input = load_rgb_image_tensor(image_path, cfg.canvas_size).unsqueeze(0).to(cfg.device);
        auto pred_mask_t = unet->forward(input).argmax(1).squeeze(0);
        cv::Mat pred_mask = tensor_mask_to_mat(pred_mask_t);

        cv::Mat gt_color = colorize_mask_hsv(gt_mask, cfg.num_classes_seg);
        cv::Mat pred_color = colorize_mask_hsv(pred_mask, cfg.num_classes_seg);

        cv::putText(image,
                    "Input",
                    cv::Point(8, 20),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(255, 255, 255),
                    2,
                    cv::LINE_AA);

        cv::putText(gt_color,
                    "GT Mask",
                    cv::Point(8, 20),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(255, 255, 255),
                    2,
                    cv::LINE_AA);

        cv::putText(pred_color,
                    "Pred Mask",
                    cv::Point(8, 20),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(255, 255, 255),
                    2,
                    cv::LINE_AA);

        cv::Mat row;
        cv::hconcat(std::vector<cv::Mat>{image, gt_color, pred_color}, row);
        seg_rows.push_back(row);
    }

    cv::Mat seg_grid;
    if (!seg_rows.empty()) {
        cv::vconcat(seg_rows, seg_grid);
        cv::Mat bgr;
        cv::cvtColor(seg_grid, bgr, cv::COLOR_RGB2BGR);
        cv::imwrite(cfg.segmentation_vis_png.string(), bgr);
    }

    // 7.3 Dashboard composition via hconcat/vconcat.
    if (detection_grid.empty() || seg_grid.empty()) {
        std::cout << "Skipping dashboard generation: missing detection/segmentation visual tiles." << std::endl;
        return;
    }

    const int top_height = std::max(detection_grid.rows, seg_grid.rows);
    cv::Mat det_top = resize_to_height(detection_grid, top_height);
    cv::Mat seg_top = resize_to_height(seg_grid, top_height);

    cv::Mat top;
    cv::hconcat(std::vector<cv::Mat>{det_top, seg_top}, top);

    cv::Mat summary = make_summary_panel(cfg, det_metrics, seg_metrics, top.cols);

    cv::Mat dashboard;
    cv::vconcat(std::vector<cv::Mat>{top, summary}, dashboard);

    cv::Mat dashboard_bgr;
    cv::cvtColor(dashboard, dashboard_bgr, cv::COLOR_RGB2BGR);
    cv::imwrite(cfg.dashboard_png.string(), dashboard_bgr);
}

}  // namespace change
