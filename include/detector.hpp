/*
 * detector.hpp
 * Purpose: Lightweight detector module and training/inference APIs replacing YOLO training.
 */
#pragma once

#include "config.hpp"
#include "dataset.hpp"

#include <torch/torch.h>

#include <filesystem>
#include <vector>

namespace change {

class ConvBlockImpl : public torch::nn::Module {
public:
    ConvBlockImpl(int64_t in_channels, int64_t out_channels)
        : conv_(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1).bias(false)),
          bn_(out_channels),
          relu_(torch::nn::ReLUOptions(true)),
          pool_(torch::nn::MaxPool2dOptions(2)) {
        register_module("conv", conv_);
        register_module("bn", bn_);
        register_module("relu", relu_);
        register_module("pool", pool_);
    }

    torch::Tensor forward(const torch::Tensor& x) {
        auto y = conv_->forward(x);
        y = bn_->forward(y);
        y = relu_->forward(y);
        y = pool_->forward(y);
        return y;
    }

private:
    torch::nn::Conv2d conv_{nullptr};
    torch::nn::BatchNorm2d bn_{nullptr};
    torch::nn::ReLU relu_{nullptr};
    torch::nn::MaxPool2d pool_{nullptr};
};
TORCH_MODULE(ConvBlock);

class LightweightDetectorImpl : public torch::nn::Module {
public:
    LightweightDetectorImpl(int64_t num_classes = Config::kNumClassesDet,
                            int64_t num_anchors = Config::kDetectorNumAnchors)
        : num_classes_(num_classes),
          num_anchors_(num_anchors),
          block1_(ConvBlock(3, 32)),
          block2_(ConvBlock(32, 64)),
          block3_(ConvBlock(64, 128)),
          block4_(ConvBlock(128, 256)),
          head_(torch::nn::Conv2dOptions(256, num_anchors_ * (5 + num_classes_), 1)) {
        register_module("block1", block1_);
        register_module("block2", block2_);
        register_module("block3", block3_);
        register_module("block4", block4_);
        register_module("head", head_);
    }

    torch::Tensor forward(const torch::Tensor& x) {
        auto y = block1_->forward(x);
        y = block2_->forward(y);
        y = block3_->forward(y);
        y = block4_->forward(y);
        return head_->forward(y);
    }

    int64_t num_classes() const {
        return num_classes_;
    }

    int64_t num_anchors() const {
        return num_anchors_;
    }

private:
    int64_t num_classes_;
    int64_t num_anchors_;

    ConvBlock block1_{nullptr};
    ConvBlock block2_{nullptr};
    ConvBlock block3_{nullptr};
    ConvBlock block4_{nullptr};
    torch::nn::Conv2d head_{nullptr};
};
TORCH_MODULE(LightweightDetector);

struct DetectorTrainingResult {
    std::filesystem::path checkpoint_path;
    std::vector<double> train_losses;
    std::vector<double> val_losses;
};

DetectorTrainingResult train_detector_model(const Config& cfg,
                                            LightweightDetector& model,
                                            const std::vector<SceneAnnotation>& train_annotations,
                                            const std::vector<SceneAnnotation>& val_annotations);

std::vector<BoundingBox> decode_detector_predictions(const torch::Tensor& raw_output,
                                                     const Config& cfg,
                                                     float conf_threshold);

}  // namespace change
