/*
 * dataset.hpp
 * Purpose: Dataset/data-generation interfaces for MNISTDD-RGB scenes and LibTorch datasets.
 */
#pragma once

#include "config.hpp"

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace change {

struct BoundingBox {
    float x1 = 0.0F;
    float y1 = 0.0F;
    float x2 = 0.0F;
    float y2 = 0.0F;
    int64_t label = 0;
    float score = 1.0F;
};

struct SceneAnnotation {
    std::vector<BoundingBox> boxes;
};

struct RawMnistData {
    std::vector<cv::Mat> images;   // CV_32FC1, values in [0, 1]
    std::vector<uint8_t> labels;   // 0..9
};

class MNISTDDDataset : public torch::data::datasets::Dataset<MNISTDDDataset> {
public:
    MNISTDDDataset(std::filesystem::path dataset_root,
                   std::string split,
                   int canvas_size = Config::kCanvasSize);

    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;

    const std::vector<std::filesystem::path>& image_paths() const;

private:
    std::filesystem::path dataset_root_;
    std::string split_;
    int canvas_size_;

    std::vector<std::filesystem::path> image_paths_;
    std::vector<std::filesystem::path> mask_paths_;
};

RawMnistData load_mnist_idx(const std::filesystem::path& images_idx,
                            const std::filesystem::path& labels_idx);

void prepare_mnistdd_dataset(const Config& cfg);

std::vector<std::filesystem::path> list_split_images(const Config& cfg,
                                                     const std::string& split);

std::vector<SceneAnnotation> load_split_annotations(const Config& cfg,
                                                    const std::string& split,
                                                    size_t expected_count);

}  // namespace change
