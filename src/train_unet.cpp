/*
 * train_unet.cpp
 * Purpose: SECTION 5 -- U-Net training loop (AdamW, scheduler, AMP-aware path, early stopping).
 */

#include "unet.hpp"

#include "dataset.hpp"
#include "utils.hpp"

#include <torch/nn/utils/clip_grad.h>

#if __has_include(<ATen/autocast_mode.h>)
#include <ATen/autocast_mode.h>
#define CHANGE_HAS_AUTOCAST 1
#else
#define CHANGE_HAS_AUTOCAST 0
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace change {
namespace {

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

torch::Tensor forward_unet_with_optional_autocast(UNet& model,
                                                   const torch::Tensor& images,
                                                   bool use_amp) {
#if CHANGE_HAS_AUTOCAST
    if (use_amp) {
        const bool previous_state = at::autocast::is_autocast_enabled(at::kCUDA);
        at::autocast::set_autocast_enabled(at::kCUDA, true);
        auto out = model->forward(images);
        at::autocast::set_autocast_enabled(at::kCUDA, previous_state);
        return out;
    }
#endif
    return model->forward(images);
}

}  // namespace

UNetTrainingResult train_unet_model(const Config& cfg,
                                    UNet& model) {
    // SECTION 5 -- SEGMENTATION MODEL: U-NET
    const torch::Device training_device = cfg.unet_device;
    model->to(training_device);

    MNISTDDDataset train_dataset(cfg.dataset_dir, "train", cfg.canvas_size);
    MNISTDDDataset val_dataset(cfg.dataset_dir, "val", cfg.canvas_size);

    const auto train_size_opt = train_dataset.size();
    const auto val_size_opt = val_dataset.size();

    if (!train_size_opt.has_value() || !val_size_opt.has_value()) {
        throw std::runtime_error("Dataset size is not available for U-Net training");
    }

    const size_t train_size = train_size_opt.value();
    const size_t val_size = val_size_opt.value();

    std::vector<size_t> train_indices(train_size);
    std::iota(train_indices.begin(), train_indices.end(), 0);

    std::mt19937 rng(static_cast<uint32_t>(cfg.seed + 101));

    torch::optim::AdamW optimizer(
        model->parameters(),
        torch::optim::AdamWOptions(cfg.unet_lr).weight_decay(cfg.weight_decay));

    torch::nn::CrossEntropyLoss criterion;

    UNetTrainingResult result;
    result.checkpoint_path = cfg.unet_checkpoint;

    double best_val_loss = std::numeric_limits<double>::infinity();
    int early_stop_counter = 0;
    int lr_plateau_counter = 0;

    const bool use_amp = training_device.is_cuda() && cfg.enable_amp;

    for (int epoch = 1; epoch <= cfg.unet_epochs; ++epoch) {
        std::shuffle(train_indices.begin(), train_indices.end(), rng);

        model->train();
        double train_running = 0.0;
        size_t train_seen = 0;

        for (size_t start = 0; start < train_size; start += static_cast<size_t>(cfg.unet_batch)) {
            const size_t end = std::min(train_size, start + static_cast<size_t>(cfg.unet_batch));

            std::vector<torch::Tensor> image_batch;
            std::vector<torch::Tensor> mask_batch;
            image_batch.reserve(end - start);
            mask_batch.reserve(end - start);

            for (size_t i = start; i < end; ++i) {
                const auto sample = train_dataset.get(train_indices[i]);
                image_batch.push_back(sample.data);
                mask_batch.push_back(sample.target);
            }

            auto images = torch::stack(image_batch).to(training_device);
            auto masks = torch::stack(mask_batch).to(training_device).to(torch::kLong);

            optimizer.zero_grad();

            const auto logits = forward_unet_with_optional_autocast(model, images, use_amp);
            const auto loss = criterion(logits, masks);

            loss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            optimizer.step();

            train_running += loss.item<double>() * static_cast<double>(end - start);
            train_seen += (end - start);
        }

        const double train_loss = train_running / static_cast<double>(std::max<size_t>(1, train_seen));
        result.train_losses.push_back(train_loss);

        model->eval();
        torch::NoGradGuard no_grad;

        double val_running = 0.0;
        size_t val_seen = 0;

        for (size_t start = 0; start < val_size; start += static_cast<size_t>(cfg.unet_batch)) {
            const size_t end = std::min(val_size, start + static_cast<size_t>(cfg.unet_batch));

            std::vector<torch::Tensor> image_batch;
            std::vector<torch::Tensor> mask_batch;
            image_batch.reserve(end - start);
            mask_batch.reserve(end - start);

            for (size_t i = start; i < end; ++i) {
                const auto sample = val_dataset.get(i);
                image_batch.push_back(sample.data);
                mask_batch.push_back(sample.target);
            }

            auto images = torch::stack(image_batch).to(training_device);
            auto masks = torch::stack(mask_batch).to(training_device).to(torch::kLong);

            const auto logits = forward_unet_with_optional_autocast(model, images, use_amp);
            const auto loss = criterion(logits, masks);

            val_running += loss.item<double>() * static_cast<double>(end - start);
            val_seen += (end - start);
        }

        const double val_loss = val_running / static_cast<double>(std::max<size_t>(1, val_seen));
        result.val_losses.push_back(val_loss);

        bool improved = false;
        if (val_loss < best_val_loss - cfg.early_stop_min_delta) {
            best_val_loss = val_loss;
            early_stop_counter = 0;
            lr_plateau_counter = 0;
            torch::save(model, cfg.unet_checkpoint.string());
            improved = true;
        } else {
            ++early_stop_counter;
            ++lr_plateau_counter;

            if (lr_plateau_counter >= cfg.reduce_lr_patience) {
                const double next_lr = std::max(1.0e-6, get_lr(optimizer) * cfg.reduce_lr_factor);
                set_lr(optimizer, next_lr);
                lr_plateau_counter = 0;
            }
        }

        std::cout << "[UNet] Epoch " << epoch << '/' << cfg.unet_epochs
                  << " | Train " << train_loss
                  << " | Val " << val_loss
                  << " | LR " << get_lr(optimizer)
                  << " | P " << early_stop_counter << '/' << cfg.unet_patience
                  << (improved ? " | saved" : "")
                  << std::endl;

        if (early_stop_counter >= cfg.unet_patience) {
            std::cout << "[UNet] Early stopping triggered at epoch " << epoch << std::endl;
            break;
        }
    }

    save_history_csv(cfg.unet_history_csv, result.train_losses, result.val_losses);

    if (std::filesystem::exists(cfg.unet_checkpoint)) {
        torch::load(model, cfg.unet_checkpoint.string());
    }
    model->eval();

    return result;
}

}  // namespace change
