/*
 * train_detector.cpp
 * Purpose: SECTION 4 -- detector training loop and lightweight detection decoding.
 */

#include "detector.hpp"

#include "dataset.hpp"
#include "utils.hpp"

#include <torch/nn/functional.h>
#include <torch/nn/utils/clip_grad.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace change {
namespace {

struct DetectorTargets {
    torch::Tensor bbox;  // [B, A, 4, H, W]
    torch::Tensor obj;   // [B, A, H, W]
    torch::Tensor cls;   // [B, A, H, W], -1 for background
};

struct DetectorLossParts {
    torch::Tensor total;
    torch::Tensor bbox;
    torch::Tensor obj;
    torch::Tensor cls;
};

float safe_log_ratio(float value,
                     float base,
                     float eps = 1.0e-6F) {
    return std::log(std::max(value, eps) / std::max(base, eps));
}

float width_height_iou(float w1,
                       float h1,
                       float w2,
                       float h2) {
    const float inter = std::min(w1, w2) * std::min(h1, h2);
    const float union_v = w1 * h1 + w2 * h2 - inter;
    if (union_v <= 0.0F) {
        return 0.0F;
    }
    return inter / union_v;
}

DetectorTargets build_detector_targets(const std::vector<SceneAnnotation>& annotations,
                                       const Config& cfg,
                                       const torch::Device& device,
                                       int64_t grid_h,
                                       int64_t grid_w) {
    const int64_t batch = static_cast<int64_t>(annotations.size());
    const int64_t anchors = cfg.detector_num_anchors;

    auto bbox = torch::zeros(
        {batch, anchors, 4, grid_h, grid_w},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));

    auto obj = torch::zeros(
        {batch, anchors, grid_h, grid_w},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));

    auto cls = torch::full(
        {batch, anchors, grid_h, grid_w},
        -1,
        torch::TensorOptions().dtype(torch::kLong).device(device));

    const float cell_w = static_cast<float>(cfg.canvas_size) / static_cast<float>(grid_w);
    const float cell_h = static_cast<float>(cfg.canvas_size) / static_cast<float>(grid_h);

    for (int64_t b = 0; b < batch; ++b) {
        for (const auto& box : annotations[static_cast<size_t>(b)].boxes) {
            if (box.label < 0 || box.label >= cfg.num_classes_det) {
                continue;
            }

            const float bw = std::max(1.0F, box.x2 - box.x1);
            const float bh = std::max(1.0F, box.y2 - box.y1);
            const float cx = 0.5F * (box.x1 + box.x2);
            const float cy = 0.5F * (box.y1 + box.y2);

            int64_t gx = static_cast<int64_t>(cx / cell_w);
            int64_t gy = static_cast<int64_t>(cy / cell_h);
            gx = std::max<int64_t>(0, std::min<int64_t>(grid_w - 1, gx));
            gy = std::max<int64_t>(0, std::min<int64_t>(grid_h - 1, gy));

            int64_t best_anchor = 0;
            float best_iou = -1.0F;
            for (int64_t a = 0; a < anchors; ++a) {
                const auto anchor = cfg.detector_anchors[static_cast<size_t>(a)];
                const float aw = anchor.w * static_cast<float>(cfg.canvas_size);
                const float ah = anchor.h * static_cast<float>(cfg.canvas_size);
                const float iou = width_height_iou(bw, bh, aw, ah);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_anchor = a;
                }
            }

            const float tx = (cx / cell_w) - static_cast<float>(gx);
            const float ty = (cy / cell_h) - static_cast<float>(gy);

            const auto anchor = cfg.detector_anchors[static_cast<size_t>(best_anchor)];
            const float tw = safe_log_ratio(
                bw / static_cast<float>(cfg.canvas_size),
                anchor.w);
            const float th = safe_log_ratio(
                bh / static_cast<float>(cfg.canvas_size),
                anchor.h);

            bbox.index_put_({b, best_anchor, 0, gy, gx}, tx);
            bbox.index_put_({b, best_anchor, 1, gy, gx}, ty);
            bbox.index_put_({b, best_anchor, 2, gy, gx}, tw);
            bbox.index_put_({b, best_anchor, 3, gy, gx}, th);
            obj.index_put_({b, best_anchor, gy, gx}, 1.0F);
            cls.index_put_({b, best_anchor, gy, gx}, static_cast<int64_t>(box.label));
        }
    }

    return DetectorTargets{bbox, obj, cls};
}

DetectorLossParts detector_loss(const torch::Tensor& raw_output,
                                const DetectorTargets& targets,
                                const Config& cfg) {
    const auto b = raw_output.size(0);
    const auto a = cfg.detector_num_anchors;
    const auto h = raw_output.size(2);
    const auto w = raw_output.size(3);

    auto pred = raw_output.contiguous().view({b, a, 5 + cfg.num_classes_det, h, w});

    auto pred_xy = torch::sigmoid(pred.slice(2, 0, 2));
    auto pred_wh = pred.slice(2, 2, 4);
    auto pred_obj = pred.select(2, 4);
    auto pred_cls = pred.slice(2, 5, 5 + cfg.num_classes_det);

    auto tgt_xy = targets.bbox.slice(2, 0, 2);
    auto tgt_wh = targets.bbox.slice(2, 2, 4);

    auto obj_mask = targets.obj.unsqueeze(2);
    auto positive_count = targets.obj.sum().clamp_min(1.0);

    auto bbox_xy_loss = torch::nn::functional::mse_loss(
        pred_xy * obj_mask,
        tgt_xy * obj_mask,
        torch::nn::functional::MSELossFuncOptions(torch::kSum));

    auto bbox_wh_loss = torch::nn::functional::mse_loss(
        pred_wh * obj_mask,
        tgt_wh * obj_mask,
        torch::nn::functional::MSELossFuncOptions(torch::kSum));

    auto bbox_loss = (bbox_xy_loss + bbox_wh_loss) / positive_count;

    auto obj_loss = torch::nn::functional::binary_cross_entropy_with_logits(
        pred_obj,
        targets.obj,
        torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions());

    auto cls_loss = torch::zeros({}, raw_output.options());
    auto positive_indices = (targets.obj > 0.5).nonzero();
    if (positive_indices.numel() > 0) {
        auto pred_cls_view = pred_cls.permute({0, 1, 3, 4, 2});  // [B, A, H, W, C]

        auto idx_b = positive_indices.select(1, 0);
        auto idx_a = positive_indices.select(1, 1);
        auto idx_h = positive_indices.select(1, 2);
        auto idx_w = positive_indices.select(1, 3);

        auto pred_pos = pred_cls_view.index({idx_b, idx_a, idx_h, idx_w});
        auto tgt_pos = targets.cls.index({idx_b, idx_a, idx_h, idx_w});

        cls_loss = torch::nn::functional::cross_entropy(
            pred_pos,
            tgt_pos,
            torch::nn::functional::CrossEntropyFuncOptions());
    }

    auto total = cfg.loss_bbox * bbox_loss + cfg.loss_objectness * obj_loss + cfg.loss_class * cls_loss;

    return DetectorLossParts{total, bbox_loss, obj_loss, cls_loss};
}

double get_lr(torch::optim::AdamW& optimizer) {
    return static_cast<torch::optim::AdamWOptions&>(optimizer.param_groups().front().options()).lr();
}

void set_lr(torch::optim::AdamW& optimizer,
            double new_lr) {
    for (auto& group : optimizer.param_groups()) {
        auto& options = static_cast<torch::optim::AdamWOptions&>(group.options());
        options.lr(new_lr);
    }
}

std::vector<BoundingBox> non_max_suppression(const std::vector<BoundingBox>& boxes,
                                             float iou_threshold) {
    std::vector<BoundingBox> sorted = boxes;
    std::sort(sorted.begin(), sorted.end(), [](const BoundingBox& lhs, const BoundingBox& rhs) {
        return lhs.score > rhs.score;
    });

    std::vector<BoundingBox> kept;
    std::vector<bool> removed(sorted.size(), false);

    for (size_t i = 0; i < sorted.size(); ++i) {
        if (removed[i]) {
            continue;
        }
        kept.push_back(sorted[i]);
        for (size_t j = i + 1; j < sorted.size(); ++j) {
            if (removed[j]) {
                continue;
            }
            if (sorted[i].label != sorted[j].label) {
                continue;
            }
            if (compute_iou(sorted[i], sorted[j]) >= iou_threshold) {
                removed[j] = true;
            }
        }
    }

    return kept;
}

}  // namespace

DetectorTrainingResult train_detector_model(const Config& cfg,
                                            LightweightDetector& model,
                                            const std::vector<SceneAnnotation>& train_annotations,
                                            const std::vector<SceneAnnotation>& val_annotations) {
    // SECTION 4 -- DETECTION MODEL: custom detector replacing YOLO train API.
    const torch::Device training_device = cfg.detector_device;
    model->to(training_device);

    const auto train_images = list_split_images(cfg, "train");
    const auto val_images = list_split_images(cfg, "val");

    if (train_images.size() != train_annotations.size()) {
        throw std::runtime_error("Train image/annotation count mismatch");
    }
    if (val_images.size() != val_annotations.size()) {
        throw std::runtime_error("Validation image/annotation count mismatch");
    }

    torch::optim::AdamW optimizer(
        model->parameters(),
        torch::optim::AdamWOptions(cfg.detector_lr).weight_decay(cfg.weight_decay));

    DetectorTrainingResult result;
    result.checkpoint_path = cfg.detector_checkpoint;

    double best_val_loss = std::numeric_limits<double>::infinity();
    int plateau_epochs = 0;

    std::vector<size_t> train_indices(train_images.size());
    std::iota(train_indices.begin(), train_indices.end(), 0);

    std::mt19937 rng(static_cast<uint32_t>(cfg.seed));

    for (int epoch = 1; epoch <= cfg.detector_epochs; ++epoch) {
        std::shuffle(train_indices.begin(), train_indices.end(), rng);

        model->train();
        double train_running = 0.0;
        size_t train_seen = 0;

        for (size_t start = 0; start < train_indices.size(); start += static_cast<size_t>(cfg.detector_batch)) {
            const size_t end = std::min(train_indices.size(), start + static_cast<size_t>(cfg.detector_batch));

            std::vector<torch::Tensor> batch_images;
            std::vector<SceneAnnotation> batch_annotations;
            batch_images.reserve(end - start);
            batch_annotations.reserve(end - start);

            for (size_t i = start; i < end; ++i) {
                const size_t idx = train_indices[i];
                batch_images.push_back(load_rgb_image_tensor(train_images[idx], cfg.canvas_size));
                batch_annotations.push_back(train_annotations[idx]);
            }

            auto images = torch::stack(batch_images).to(training_device);

            optimizer.zero_grad();
            auto output = model->forward(images);

            const auto targets = build_detector_targets(
                batch_annotations,
                cfg,
                images.device(),
                output.size(2),
                output.size(3));

            const auto parts = detector_loss(output, targets, cfg);
            parts.total.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            optimizer.step();

            train_running += parts.total.item<double>() * static_cast<double>(end - start);
            train_seen += (end - start);
        }

        const double train_loss = train_running / static_cast<double>(std::max<size_t>(1, train_seen));
        result.train_losses.push_back(train_loss);

        model->eval();
        torch::NoGradGuard no_grad;

        double val_running = 0.0;
        size_t val_seen = 0;

        for (size_t start = 0; start < val_images.size(); start += static_cast<size_t>(cfg.detector_batch)) {
            const size_t end = std::min(val_images.size(), start + static_cast<size_t>(cfg.detector_batch));

            std::vector<torch::Tensor> batch_images;
            std::vector<SceneAnnotation> batch_annotations;
            batch_images.reserve(end - start);
            batch_annotations.reserve(end - start);

            for (size_t i = start; i < end; ++i) {
                batch_images.push_back(load_rgb_image_tensor(val_images[i], cfg.canvas_size));
                batch_annotations.push_back(val_annotations[i]);
            }

            auto images = torch::stack(batch_images).to(training_device);
            auto output = model->forward(images);

            const auto targets = build_detector_targets(
                batch_annotations,
                cfg,
                images.device(),
                output.size(2),
                output.size(3));

            const auto parts = detector_loss(output, targets, cfg);
            val_running += parts.total.item<double>() * static_cast<double>(end - start);
            val_seen += (end - start);
        }

        const double val_loss = val_running / static_cast<double>(std::max<size_t>(1, val_seen));
        result.val_losses.push_back(val_loss);

        bool improved = false;
        if (val_loss < best_val_loss - cfg.early_stop_min_delta) {
            best_val_loss = val_loss;
            plateau_epochs = 0;
            torch::save(model, cfg.detector_checkpoint.string());
            improved = true;
        } else {
            ++plateau_epochs;
            if (plateau_epochs % cfg.reduce_lr_patience == 0) {
                const double next_lr = std::max(1.0e-6, get_lr(optimizer) * cfg.reduce_lr_factor);
                set_lr(optimizer, next_lr);
            }
        }

        std::cout << "[Detector] Epoch " << epoch << '/' << cfg.detector_epochs
                  << " | Train " << train_loss
                  << " | Val " << val_loss
                  << " | LR " << get_lr(optimizer)
                  << " | P " << plateau_epochs << '/' << cfg.detector_patience
                  << (improved ? " | saved" : "")
                  << std::endl;

        if (plateau_epochs >= cfg.detector_patience) {
            std::cout << "[Detector] Early stopping triggered at epoch " << epoch << std::endl;
            break;
        }
    }

    save_history_csv(cfg.detector_history_csv, result.train_losses, result.val_losses);

    if (std::filesystem::exists(cfg.detector_checkpoint)) {
        torch::load(model, cfg.detector_checkpoint.string());
    }
    model->eval();

    return result;
}

std::vector<BoundingBox> decode_detector_predictions(const torch::Tensor& raw_output,
                                                     const Config& cfg,
                                                     float conf_threshold) {
    torch::Tensor output = raw_output;
    if (output.dim() == 4) {
        output = output.squeeze(0);
    }
    output = output.detach().to(torch::kCPU).contiguous();

    if (output.dim() != 3) {
        throw std::runtime_error("Detector output must be [C,H,W] or [1,C,H,W]");
    }

    const int64_t channels = output.size(0);
    const int64_t h = output.size(1);
    const int64_t w = output.size(2);

    const int64_t expected_channels = cfg.detector_num_anchors * (5 + cfg.num_classes_det);
    if (channels != expected_channels) {
        throw std::runtime_error("Unexpected detector channel count during decode");
    }

    auto pred = output.view({cfg.detector_num_anchors, 5 + cfg.num_classes_det, h, w}).permute({0, 2, 3, 1});

    std::vector<BoundingBox> boxes;

    for (int64_t a = 0; a < cfg.detector_num_anchors; ++a) {
        const auto anchor = cfg.detector_anchors[static_cast<size_t>(a)];
        const float anchor_w = anchor.w * static_cast<float>(cfg.canvas_size);
        const float anchor_h = anchor.h * static_cast<float>(cfg.canvas_size);

        for (int64_t gy = 0; gy < h; ++gy) {
            for (int64_t gx = 0; gx < w; ++gx) {
                auto cell = pred.index({a, gy, gx});

                const float tx = torch::sigmoid(cell.index({0})).item<float>();
                const float ty = torch::sigmoid(cell.index({1})).item<float>();
                const float tw = cell.index({2}).item<float>();
                const float th = cell.index({3}).item<float>();
                const float obj = torch::sigmoid(cell.index({4})).item<float>();

                auto cls_logits = cell.slice(0, 5, 5 + cfg.num_classes_det);
                auto cls_probs = torch::softmax(cls_logits, 0);
                auto max_pair = cls_probs.max(0);

                const float cls_prob = std::get<0>(max_pair).item<float>();
                const int64_t cls_id = std::get<1>(max_pair).item<int64_t>();

                const float score = obj * cls_prob;
                if (score < conf_threshold) {
                    continue;
                }

                const float cx = ((static_cast<float>(gx) + tx) / static_cast<float>(w)) * static_cast<float>(cfg.canvas_size);
                const float cy = ((static_cast<float>(gy) + ty) / static_cast<float>(h)) * static_cast<float>(cfg.canvas_size);
                const float bw = std::exp(tw) * anchor_w;
                const float bh = std::exp(th) * anchor_h;

                const float x1 = std::max(0.0F, cx - 0.5F * bw);
                const float y1 = std::max(0.0F, cy - 0.5F * bh);
                const float x2 = std::min(static_cast<float>(cfg.canvas_size - 1), cx + 0.5F * bw);
                const float y2 = std::min(static_cast<float>(cfg.canvas_size - 1), cy + 0.5F * bh);

                boxes.push_back(BoundingBox{x1, y1, x2, y2, cls_id, score});
            }
        }
    }

    return non_max_suppression(boxes, cfg.detector_iou_thresh);
}

}  // namespace change
