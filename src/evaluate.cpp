/*
 * evaluate.cpp
 * Purpose: SECTION 6 -- detector and segmentation evaluation metrics.
 */

#include "utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace change {

DetectionMetrics evaluate_detection(const Config& cfg,
                                    LightweightDetector& detector,
                                    const std::vector<SceneAnnotation>& test_annotations) {
    // SECTION 6 -- DETECTION EVALUATION
    const auto test_images = list_split_images(cfg, "test");
    if (test_images.size() != test_annotations.size()) {
        throw std::runtime_error("Test image/annotation count mismatch in detection evaluation");
    }

    detector->to(cfg.device);
    detector->eval();

    DetectionMetrics metrics;
    metrics.per_class.resize(static_cast<size_t>(cfg.num_classes_det));

    torch::NoGradGuard no_grad;

    for (size_t i = 0; i < test_images.size(); ++i) {
        const auto image = load_rgb_image_tensor(test_images[i], cfg.canvas_size).unsqueeze(0).to(cfg.device);
        const auto output = detector->forward(image);
        auto predictions = decode_detector_predictions(output, cfg, cfg.detector_conf_thresh);
        auto gt_boxes = test_annotations[i].boxes;

        metrics.total_gt += static_cast<int64_t>(gt_boxes.size());

        std::sort(predictions.begin(), predictions.end(), [](const BoundingBox& lhs, const BoundingBox& rhs) {
            return lhs.score > rhs.score;
        });

        std::vector<bool> matched(gt_boxes.size(), false);

        for (const auto& pred : predictions) {
            if (pred.label < 0 || pred.label >= cfg.num_classes_det) {
                continue;
            }

            double best_iou = 0.0;
            int best_index = -1;

            for (size_t j = 0; j < gt_boxes.size(); ++j) {
                if (matched[j]) {
                    continue;
                }
                if (gt_boxes[j].label != pred.label) {
                    continue;
                }

                const double iou = compute_iou(pred, gt_boxes[j]);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_index = static_cast<int>(j);
                }
            }

            auto& cm = metrics.per_class[static_cast<size_t>(pred.label)];
            if (best_index >= 0 && best_iou >= cfg.detector_iou_thresh) {
                cm.tp += 1;
                metrics.total_correct += 1;
                matched[static_cast<size_t>(best_index)] = true;
            } else {
                cm.fp += 1;
            }
        }

        for (size_t j = 0; j < gt_boxes.size(); ++j) {
            if (!matched[j]) {
                const auto label = gt_boxes[j].label;
                if (label >= 0 && label < cfg.num_classes_det) {
                    metrics.per_class[static_cast<size_t>(label)].fn += 1;
                }
            }
        }
    }

    int64_t all_tp = 0;
    int64_t all_fp = 0;
    int64_t all_fn = 0;

    for (auto& cls : metrics.per_class) {
        cls.precision = static_cast<double>(cls.tp) /
                        static_cast<double>(cls.tp + cls.fp + 1.0e-8);
        cls.recall = static_cast<double>(cls.tp) /
                     static_cast<double>(cls.tp + cls.fn + 1.0e-8);
        cls.f1 = (2.0 * cls.precision * cls.recall) /
                 (cls.precision + cls.recall + 1.0e-8);

        all_tp += cls.tp;
        all_fp += cls.fp;
        all_fn += cls.fn;
    }

    metrics.overall_precision = static_cast<double>(all_tp) / static_cast<double>(all_tp + all_fp + 1.0e-8);
    metrics.overall_recall = static_cast<double>(all_tp) / static_cast<double>(all_tp + all_fn + 1.0e-8);
    metrics.overall_f1 = (2.0 * metrics.overall_precision * metrics.overall_recall) /
                         (metrics.overall_precision + metrics.overall_recall + 1.0e-8);
    metrics.accuracy = static_cast<double>(metrics.total_correct) / static_cast<double>(metrics.total_gt + 1.0e-8);

    return metrics;
}

SegmentationMetrics evaluate_segmentation(const Config& cfg,
                                          UNet& model) {
    // SECTION 6 -- SEGMENTATION EVALUATION
    MNISTDDDataset test_dataset(cfg.dataset_dir, "test", cfg.canvas_size);

    const auto size_opt = test_dataset.size();
    if (!size_opt.has_value()) {
        throw std::runtime_error("Test dataset size unavailable for segmentation evaluation");
    }

    const size_t test_size = size_opt.value();

    model->to(cfg.device);
    model->eval();

    auto confusion = torch::zeros(
        {cfg.num_classes_seg, cfg.num_classes_seg},
        torch::TensorOptions().dtype(torch::kFloat64));

    torch::NoGradGuard no_grad;

    for (size_t start = 0; start < test_size; start += static_cast<size_t>(cfg.unet_batch)) {
        const size_t end = std::min(test_size, start + static_cast<size_t>(cfg.unet_batch));

        std::vector<torch::Tensor> image_batch;
        std::vector<torch::Tensor> mask_batch;
        image_batch.reserve(end - start);
        mask_batch.reserve(end - start);

        for (size_t i = start; i < end; ++i) {
            const auto sample = test_dataset.get(i);
            image_batch.push_back(sample.data);
            mask_batch.push_back(sample.target);
        }

        auto images = torch::stack(image_batch).to(cfg.device);
        auto masks = torch::stack(mask_batch).to(torch::kLong);

        auto logits = model->forward(images);
        auto preds = logits.argmax(1).to(torch::kCPU).to(torch::kLong);
        masks = masks.to(torch::kCPU).to(torch::kLong);

        auto flat_preds = preds.reshape({-1});
        auto flat_masks = masks.reshape({-1});

        auto k = flat_preds * cfg.num_classes_seg + flat_masks;
        auto bc = torch::bincount(
            k,
            {},
            cfg.num_classes_seg * cfg.num_classes_seg).to(torch::kFloat64);

        confusion += bc.view({cfg.num_classes_seg, cfg.num_classes_seg});
    }

    SegmentationMetrics metrics;
    metrics.class_iou.resize(static_cast<size_t>(cfg.num_classes_seg), 0.0);
    metrics.class_pixel_accuracy.resize(static_cast<size_t>(cfg.num_classes_seg), 0.0);

    auto inter = confusion.diag();
    auto pred_total = confusion.sum(1);
    auto gt_total = confusion.sum(0);
    auto union_v = pred_total + gt_total - inter;

    double iou_sum = 0.0;
    int valid_iou = 0;

    for (int c = 0; c < cfg.num_classes_seg; ++c) {
        const double inter_c = inter[c].item<double>();
        const double union_c = union_v[c].item<double>();
        const double gt_c = gt_total[c].item<double>();

        if (union_c > 0.0) {
            metrics.class_iou[static_cast<size_t>(c)] = inter_c / union_c;
            iou_sum += metrics.class_iou[static_cast<size_t>(c)];
            ++valid_iou;
        }

        if (gt_c > 0.0) {
            metrics.class_pixel_accuracy[static_cast<size_t>(c)] = inter_c / gt_c;
        }
    }

    metrics.mean_iou = (valid_iou > 0) ? (iou_sum / static_cast<double>(valid_iou)) : 0.0;

    const double diag_sum = inter.sum().item<double>();
    const double total_sum = confusion.sum().item<double>();
    metrics.pixel_accuracy = (total_sum > 0.0) ? (diag_sum / total_sum) : 0.0;

    double fg_inter = 0.0;
    double fg_total = 0.0;
    for (int c = 1; c < cfg.num_classes_seg; ++c) {
        fg_inter += inter[c].item<double>();
        fg_total += gt_total[c].item<double>();
    }
    metrics.foreground_accuracy = (fg_total > 0.0) ? (fg_inter / fg_total) : 0.0;

    metrics.confusion = cv::Mat(cfg.num_classes_seg, cfg.num_classes_seg, CV_64F, cv::Scalar(0));
    auto conf_cpu = confusion.to(torch::kCPU);
    for (int r = 0; r < cfg.num_classes_seg; ++r) {
        for (int c = 0; c < cfg.num_classes_seg; ++c) {
            metrics.confusion.at<double>(r, c) = conf_cpu.index({r, c}).item<double>();
        }
    }

    return metrics;
}

void print_detection_metrics(const DetectionMetrics& metrics,
                             int num_classes) {
    std::cout << "Detection Accuracy: " << std::fixed << std::setprecision(4)
              << metrics.accuracy << " (" << metrics.total_correct << "/" << metrics.total_gt << ")"
              << std::endl;
    std::cout << std::endl;

    std::cout << std::right
              << std::setw(6) << "Class"
              << " | " << std::setw(8) << "Prec"
              << " | " << std::setw(8) << "Recall"
              << " | " << std::setw(8) << "F1"
              << " | " << std::setw(6) << "TP"
              << " " << std::setw(6) << "FP"
              << " " << std::setw(6) << "FN"
              << std::endl;

    std::cout << std::string(64, '-') << std::endl;

    for (int c = 0; c < num_classes; ++c) {
        const auto& cm = metrics.per_class[static_cast<size_t>(c)];
        std::cout << std::right
                  << std::setw(6) << c
                  << " | " << std::setw(8) << std::fixed << std::setprecision(4) << cm.precision
                  << " | " << std::setw(8) << cm.recall
                  << " | " << std::setw(8) << cm.f1
                  << " | " << std::setw(6) << cm.tp
                  << " " << std::setw(6) << cm.fp
                  << " " << std::setw(6) << cm.fn
                  << std::endl;
    }

    std::cout << std::string(64, '-') << std::endl;
    std::cout << std::right
              << std::setw(6) << "TOTAL"
              << " | " << std::setw(8) << std::fixed << std::setprecision(4) << metrics.overall_precision
              << " | " << std::setw(8) << metrics.overall_recall
              << " | " << std::setw(8) << metrics.overall_f1
              << " | " << std::setw(6) << "-"
              << " " << std::setw(6) << "-"
              << " " << std::setw(6) << "-"
              << std::endl;
}

void print_segmentation_metrics(const SegmentationMetrics& metrics,
                                int num_classes) {
    std::cout << std::endl;
    std::cout << "Segmentation Accuracy (foreground): " << std::fixed << std::setprecision(4)
              << metrics.foreground_accuracy << std::endl;
    std::cout << "Overall Pixel Accuracy: " << metrics.pixel_accuracy << std::endl;
    std::cout << "Mean IoU: " << metrics.mean_iou << std::endl;
    std::cout << std::endl;

    std::cout << std::right
              << std::setw(12) << "Class"
              << " | " << std::setw(8) << "IoU"
              << " | " << std::setw(8) << "PxAcc"
              << std::endl;
    std::cout << std::string(36, '-') << std::endl;

    for (int c = 0; c < num_classes; ++c) {
        const std::string name = (c == 0) ? "Background" : ("Digit " + std::to_string(c - 1));
        std::cout << std::right
                  << std::setw(12) << name
                  << " | " << std::setw(8) << std::fixed << std::setprecision(4) << metrics.class_iou[static_cast<size_t>(c)]
                  << " | " << std::setw(8) << metrics.class_pixel_accuracy[static_cast<size_t>(c)]
                  << std::endl;
    }
}

}  // namespace change
