/*
 * unet.hpp
 * Purpose: U-Net module definitions and training API for semantic segmentation.
 */
#pragma once

#include "config.hpp"

#include <torch/torch.h>

#include <filesystem>
#include <vector>

namespace change {

class DoubleConvImpl : public torch::nn::Module {
public:
    DoubleConvImpl(int64_t in_channels, int64_t out_channels)
        : layers_(torch::nn::Sequential(
              torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1).bias(false)),
              torch::nn::BatchNorm2d(out_channels),
              torch::nn::ReLU(torch::nn::ReLUOptions(true)),
              torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1).bias(false)),
              torch::nn::BatchNorm2d(out_channels),
              torch::nn::ReLU(torch::nn::ReLUOptions(true)))) {
        register_module("layers", layers_);
    }

    torch::Tensor forward(const torch::Tensor& x) {
        return layers_->forward(x);
    }

private:
    torch::nn::Sequential layers_{nullptr};
};
TORCH_MODULE(DoubleConv);

class DownImpl : public torch::nn::Module {
public:
    DownImpl(int64_t in_channels, int64_t out_channels)
        : layers_(torch::nn::Sequential(
              torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)),
              DoubleConv(in_channels, out_channels))) {
        register_module("layers", layers_);
    }

    torch::Tensor forward(const torch::Tensor& x) {
        return layers_->forward(x);
    }

private:
    torch::nn::Sequential layers_{nullptr};
};
TORCH_MODULE(Down);

class UpImpl : public torch::nn::Module {
public:
    UpImpl(int64_t in_channels, int64_t skip_channels, int64_t out_channels)
        : up_(torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 2).stride(2)),
          conv_(DoubleConv(out_channels + skip_channels, out_channels)) {
        register_module("up", up_);
        register_module("conv", conv_);
    }

    torch::Tensor forward(const torch::Tensor& x,
                          const torch::Tensor& skip) {
        auto upsampled = up_->forward(x);

        const auto diff_y = skip.size(2) - upsampled.size(2);
        const auto diff_x = skip.size(3) - upsampled.size(3);

        if (diff_x != 0 || diff_y != 0) {
            const int64_t pad_left = diff_x / 2;
            const int64_t pad_right = diff_x - pad_left;
            const int64_t pad_top = diff_y / 2;
            const int64_t pad_bottom = diff_y - pad_top;
            upsampled = torch::nn::functional::pad(
                upsampled,
                torch::nn::functional::PadFuncOptions({pad_left, pad_right, pad_top, pad_bottom}));
        }

        const auto merged = torch::cat({skip, upsampled}, 1);
        return conv_->forward(merged);
    }

private:
    torch::nn::ConvTranspose2d up_{nullptr};
    DoubleConv conv_{nullptr};
};
TORCH_MODULE(Up);

class UNetImpl : public torch::nn::Module {
public:
    UNetImpl(int64_t in_channels = 3, int64_t num_classes = Config::kNumClassesSeg)
        : inc_(DoubleConv(in_channels, 64)),
          down1_(Down(64, 128)),
          down2_(Down(128, 256)),
          down3_(Down(256, 512)),
          down4_(Down(512, 1024)),
          up1_(Up(1024, 512, 512)),
          up2_(Up(512, 256, 256)),
          up3_(Up(256, 128, 128)),
          up4_(Up(128, 64, 64)),
          out_(torch::nn::Conv2dOptions(64, num_classes, 1)) {
        register_module("inc", inc_);
        register_module("down1", down1_);
        register_module("down2", down2_);
        register_module("down3", down3_);
        register_module("down4", down4_);
        register_module("up1", up1_);
        register_module("up2", up2_);
        register_module("up3", up3_);
        register_module("up4", up4_);
        register_module("out", out_);
    }

    torch::Tensor forward(const torch::Tensor& x) {
        const auto x1 = inc_->forward(x);
        const auto x2 = down1_->forward(x1);
        const auto x3 = down2_->forward(x2);
        const auto x4 = down3_->forward(x3);
        const auto x5 = down4_->forward(x4);

        auto y = up1_->forward(x5, x4);
        y = up2_->forward(y, x3);
        y = up3_->forward(y, x2);
        y = up4_->forward(y, x1);
        return out_->forward(y);
    }

private:
    DoubleConv inc_{nullptr};
    Down down1_{nullptr};
    Down down2_{nullptr};
    Down down3_{nullptr};
    Down down4_{nullptr};

    Up up1_{nullptr};
    Up up2_{nullptr};
    Up up3_{nullptr};
    Up up4_{nullptr};

    torch::nn::Conv2d out_{nullptr};
};
TORCH_MODULE(UNet);

struct UNetTrainingResult {
    std::filesystem::path checkpoint_path;
    std::vector<double> train_losses;
    std::vector<double> val_losses;
};

UNetTrainingResult train_unet_model(const Config& cfg,
                                    UNet& model);

}  // namespace change
