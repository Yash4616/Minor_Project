/*
 * data_gen.cpp
 * Purpose: SECTION 2 -- data generation, raw MNIST IDX loading, and dataset IO.
 */

#include "dataset.hpp"

#include "utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace change {
namespace {

uint32_t read_be_u32(std::ifstream& stream) {
    std::array<uint8_t, 4> bytes{};
    stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream) {
        throw std::runtime_error("Failed to read 4 bytes from IDX stream");
    }
    return (static_cast<uint32_t>(bytes[0]) << 24U) |
           (static_cast<uint32_t>(bytes[1]) << 16U) |
           (static_cast<uint32_t>(bytes[2]) << 8U) |
           static_cast<uint32_t>(bytes[3]);
}

std::string scene_name(size_t index) {
    std::ostringstream oss;
    oss << "scene_" << std::setw(5) << std::setfill('0') << index;
    return oss.str();
}

std::string to_posix_path(std::string path) {
    std::replace(path.begin(), path.end(), '\\', '/');
    return path;
}

bool split_complete(const Config& cfg,
                    const std::string& split,
                    int expected_count) {
    if (expected_count <= 0) {
        return false;
    }

    const auto last_name = scene_name(static_cast<size_t>(expected_count - 1));
    const auto image_path = cfg.dataset_dir / "images" / split / (last_name + ".png");
    const auto mask_path = cfg.dataset_dir / "masks" / split / (last_name + ".png");
    const auto label_path = cfg.dataset_dir / "labels" / split / (last_name + ".txt");

    return std::filesystem::exists(image_path) &&
           std::filesystem::exists(mask_path) &&
           std::filesystem::exists(label_path);
}

cv::Mat augment_digit(const cv::Mat& digit,
                      const Config& cfg,
                      std::mt19937& rng,
                      int& out_size) {
    std::uniform_real_distribution<float> scale_dist(cfg.aug_scale_min, cfg.aug_scale_max);
    std::uniform_real_distribution<float> angle_dist(-cfg.aug_rotate_deg, cfg.aug_rotate_deg);

    const float scale = scale_dist(rng);
    out_size = static_cast<int>(std::round(static_cast<float>(cfg.digit_size) * scale));
    out_size = std::max(cfg.aug_min_digit_size, std::min(cfg.aug_max_digit_size, out_size));

    cv::Mat resized;
    cv::resize(digit, resized, cv::Size(out_size, out_size), 0.0, 0.0, cv::INTER_LINEAR);

    const float angle = angle_dist(rng);
    const cv::Point2f center(static_cast<float>(out_size - 1) / 2.0F,
                             static_cast<float>(out_size - 1) / 2.0F);
    const cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);

    cv::Mat rotated;
    cv::warpAffine(
        resized,
        rotated,
        rot,
        cv::Size(out_size, out_size),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0.0));

    return rotated;
}

SceneAnnotation generate_and_save_scene(const RawMnistData& mnist,
                                        size_t scene_index,
                                        const Config& cfg,
                                        const std::filesystem::path& image_dir,
                                        const std::filesystem::path& mask_dir,
                                        const std::filesystem::path& label_dir,
                                        bool augment,
                                        std::mt19937& rng) {
    if (mnist.images.empty() || mnist.labels.empty() || mnist.images.size() != mnist.labels.size()) {
        throw std::runtime_error("Invalid MNIST source data passed to scene generator");
    }

    std::uniform_int_distribution<int> digit_count_dist(cfg.min_digits, cfg.max_digits);
    std::uniform_int_distribution<size_t> sample_dist(0, mnist.images.size() - 1);
    std::uniform_real_distribution<float> bg_dist(0.0F, cfg.background_color_max);
    std::uniform_real_distribution<float> color_dist(cfg.digit_color_min, cfg.digit_color_max);

    const int n_digits = digit_count_dist(rng);

    const cv::Vec3f bg_color(bg_dist(rng), bg_dist(rng), bg_dist(rng));
    cv::Mat canvas(cfg.canvas_size, cfg.canvas_size, CV_32FC3, bg_color);

    cv::Mat noise(cfg.canvas_size, cfg.canvas_size, CV_32FC3);
    cv::randn(noise, cv::Scalar::all(0.0), cv::Scalar::all(cfg.background_noise_std));
    canvas += noise;
    cv::min(canvas, 1.0, canvas);
    cv::max(canvas, 0.0, canvas);

    cv::Mat mask(cfg.canvas_size, cfg.canvas_size, CV_8UC1, cv::Scalar(0));

    SceneAnnotation annotation;
    annotation.boxes.reserve(static_cast<size_t>(n_digits));

    for (int i = 0; i < n_digits; ++i) {
        const size_t sample_idx = sample_dist(rng);
        const int64_t label = static_cast<int64_t>(mnist.labels[sample_idx]);

        cv::Mat digit = mnist.images[sample_idx];
        int dz = cfg.digit_size;

        if (augment) {
            digit = augment_digit(digit, cfg, rng, dz);
        } else {
            if (digit.rows != cfg.digit_size || digit.cols != cfg.digit_size) {
                cv::resize(digit, digit, cv::Size(cfg.digit_size, cfg.digit_size), 0.0, 0.0, cv::INTER_LINEAR);
            }
            dz = cfg.digit_size;
        }

        if (dz >= cfg.canvas_size) {
            continue;
        }

        std::uniform_int_distribution<int> x_dist(0, cfg.canvas_size - dz);
        std::uniform_int_distribution<int> y_dist(0, cfg.canvas_size - dz);

        const int x = x_dist(rng);
        const int y = y_dist(rng);

        const cv::Vec3f digit_color(color_dist(rng), color_dist(rng), color_dist(rng));

        for (int yy = 0; yy < dz; ++yy) {
            const float* row_ptr = digit.ptr<float>(yy);
            for (int xx = 0; xx < dz; ++xx) {
                const float v = row_ptr[xx];
                if (v <= cfg.pixel_thresh) {
                    continue;
                }
                cv::Vec3f& pix = canvas.at<cv::Vec3f>(y + yy, x + xx);
                pix[0] = digit_color[0] * v;
                pix[1] = digit_color[1] * v;
                pix[2] = digit_color[2] * v;
                mask.at<uint8_t>(y + yy, x + xx) = static_cast<uint8_t>(label + 1);
            }
        }

        annotation.boxes.push_back(BoundingBox{
            static_cast<float>(x),
            static_cast<float>(y),
            static_cast<float>(x + dz),
            static_cast<float>(y + dz),
            label,
            1.0F});
    }

    cv::min(canvas, 1.0, canvas);
    cv::max(canvas, 0.0, canvas);

    cv::Mat canvas_u8;
    canvas.convertTo(canvas_u8, CV_8UC3, 255.0);

    cv::Mat canvas_bgr;
    cv::cvtColor(canvas_u8, canvas_bgr, cv::COLOR_RGB2BGR);

    const std::string name = scene_name(scene_index);
    const auto image_path = image_dir / (name + ".png");
    const auto mask_path = mask_dir / (name + ".png");
    const auto label_path = label_dir / (name + ".txt");

    if (!cv::imwrite(image_path.string(), canvas_bgr)) {
        throw std::runtime_error("Failed to save generated image: " + image_path.string());
    }
    if (!cv::imwrite(mask_path.string(), mask)) {
        throw std::runtime_error("Failed to save generated mask: " + mask_path.string());
    }

    std::ofstream lf(label_path);
    if (!lf) {
        throw std::runtime_error("Failed to write label file: " + label_path.string());
    }

    for (const auto& box : annotation.boxes) {
        const float cx = ((box.x1 + box.x2) * 0.5F) / static_cast<float>(cfg.canvas_size);
        const float cy = ((box.y1 + box.y2) * 0.5F) / static_cast<float>(cfg.canvas_size);
        const float w = (box.x2 - box.x1) / static_cast<float>(cfg.canvas_size);
        const float h = (box.y2 - box.y1) / static_cast<float>(cfg.canvas_size);
        lf << box.label << ' ' << cx << ' ' << cy << ' ' << w << ' ' << h << '\n';
    }

    return annotation;
}

void generate_split(const RawMnistData& source,
                    const Config& cfg,
                    int n_scenes,
                    const std::string& split,
                    bool augment) {
    const auto image_dir = cfg.dataset_dir / "images" / split;
    const auto mask_dir = cfg.dataset_dir / "masks" / split;
    const auto label_dir = cfg.dataset_dir / "labels" / split;

    ensure_directory(image_dir);
    ensure_directory(mask_dir);
    ensure_directory(label_dir);

    std::mt19937 rng(static_cast<uint32_t>(cfg.seed + std::hash<std::string>{}(split)));

    std::cout << "Generating split '" << split << "' with " << n_scenes << " scenes" << std::endl;
    for (int i = 0; i < n_scenes; ++i) {
        generate_and_save_scene(
            source,
            static_cast<size_t>(i),
            cfg,
            image_dir,
            mask_dir,
            label_dir,
            augment,
            rng);

        if ((i + 1) % 2000 == 0 || (i + 1) == n_scenes) {
            std::cout << "  " << split << ": " << (i + 1) << "/" << n_scenes << std::endl;
        }
    }
}

void write_data_yaml(const Config& cfg) {
    const auto yaml_path = cfg.dataset_dir / "data.yaml";
    std::ofstream os(yaml_path);
    if (!os) {
        throw std::runtime_error("Failed to write data.yaml: " + yaml_path.string());
    }

    const std::string dataset_abs = to_posix_path(std::filesystem::absolute(cfg.dataset_dir).string());
    os << "path: " << dataset_abs << '\n';
    os << "train: images/train\n";
    os << "val: images/val\n";
    os << "test: images/test\n";
    os << "nc: " << cfg.num_classes_det << '\n';
    os << "names: ['0','1','2','3','4','5','6','7','8','9']\n";
}

void save_sample_grid(const Config& cfg) {
    std::vector<cv::Mat> top_row;
    std::vector<cv::Mat> bottom_row;

    for (int i = 0; i < 5; ++i) {
        const auto name = scene_name(static_cast<size_t>(i));
        const auto img_path = cfg.dataset_dir / "images" / "train" / (name + ".png");
        const auto mask_path = cfg.dataset_dir / "masks" / "train" / (name + ".png");

        if (!std::filesystem::exists(img_path) || !std::filesystem::exists(mask_path)) {
            continue;
        }

        cv::Mat image = load_rgb_image_mat(img_path, cfg.canvas_size);
        cv::Mat mask = load_mask_mat(mask_path, cfg.canvas_size);
        cv::Mat mask_color = colorize_mask_hsv(mask, cfg.num_classes_seg);

        cv::putText(image,
                    "Scene " + std::to_string(i),
                    cv::Point(6, 18),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(255, 255, 255),
                    1,
                    cv::LINE_AA);

        cv::putText(mask_color,
                    "Mask " + std::to_string(i),
                    cv::Point(6, 18),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(255, 255, 255),
                    1,
                    cv::LINE_AA);

        top_row.push_back(image);
        bottom_row.push_back(mask_color);
    }

    if (top_row.empty() || bottom_row.empty()) {
        return;
    }

    cv::Mat top, bottom, grid;
    cv::hconcat(top_row, top);
    cv::hconcat(bottom_row, bottom);
    cv::vconcat(top, bottom, grid);

    cv::Mat grid_bgr;
    cv::cvtColor(grid, grid_bgr, cv::COLOR_RGB2BGR);
    cv::imwrite(cfg.samples_png.string(), grid_bgr);
}

}  // namespace

MNISTDDDataset::MNISTDDDataset(std::filesystem::path dataset_root,
                               std::string split,
                               int canvas_size)
    : dataset_root_(std::move(dataset_root)),
      split_(std::move(split)),
      canvas_size_(canvas_size) {
    const auto image_dir = dataset_root_ / "images" / split_;
    const auto mask_dir = dataset_root_ / "masks" / split_;

    if (!std::filesystem::exists(image_dir)) {
        throw std::runtime_error("Image split directory missing: " + image_dir.string());
    }
    if (!std::filesystem::exists(mask_dir)) {
        throw std::runtime_error("Mask split directory missing: " + mask_dir.string());
    }

    for (const auto& entry : std::filesystem::directory_iterator(image_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto ext = entry.path().extension().string();
        if (ext != ".png") {
            continue;
        }
        image_paths_.push_back(entry.path());
        mask_paths_.push_back(mask_dir / entry.path().filename());
    }

    std::sort(image_paths_.begin(), image_paths_.end());
    std::sort(mask_paths_.begin(), mask_paths_.end());

    if (image_paths_.size() != mask_paths_.size()) {
        throw std::runtime_error("Image and mask count mismatch for split: " + split_);
    }
}

torch::data::Example<> MNISTDDDataset::get(size_t index) {
    if (index >= image_paths_.size()) {
        throw std::out_of_range("MNISTDDDataset index out of range");
    }

    cv::Mat image = load_rgb_image_mat(image_paths_[index], canvas_size_);
    cv::Mat image_f32;
    image.convertTo(image_f32, CV_32FC3, 1.0 / 255.0);

    auto image_tensor = torch::from_blob(
        image_f32.data,
        {image_f32.rows, image_f32.cols, 3},
        torch::TensorOptions().dtype(torch::kFloat32));
    image_tensor = image_tensor.clone().permute({2, 0, 1});

    cv::Mat mask = load_mask_mat(mask_paths_[index], canvas_size_);
    auto mask_tensor = torch::from_blob(
        mask.data,
        {mask.rows, mask.cols},
        torch::TensorOptions().dtype(torch::kUInt8));
    mask_tensor = mask_tensor.clone().to(torch::kLong);

    return {image_tensor, mask_tensor};
}

torch::optional<size_t> MNISTDDDataset::size() const {
    return image_paths_.size();
}

const std::vector<std::filesystem::path>& MNISTDDDataset::image_paths() const {
    return image_paths_;
}

RawMnistData load_mnist_idx(const std::filesystem::path& images_idx,
                            const std::filesystem::path& labels_idx) {
    if (!std::filesystem::exists(images_idx)) {
        throw std::runtime_error("MNIST image IDX file not found: " + images_idx.string());
    }
    if (!std::filesystem::exists(labels_idx)) {
        throw std::runtime_error("MNIST label IDX file not found: " + labels_idx.string());
    }

    std::ifstream img_stream(images_idx, std::ios::binary);
    std::ifstream lbl_stream(labels_idx, std::ios::binary);
    if (!img_stream) {
        throw std::runtime_error("Failed to open IDX image file: " + images_idx.string());
    }
    if (!lbl_stream) {
        throw std::runtime_error("Failed to open IDX label file: " + labels_idx.string());
    }

    const uint32_t img_magic = read_be_u32(img_stream);
    const uint32_t n_images = read_be_u32(img_stream);
    const uint32_t rows = read_be_u32(img_stream);
    const uint32_t cols = read_be_u32(img_stream);

    const uint32_t lbl_magic = read_be_u32(lbl_stream);
    const uint32_t n_labels = read_be_u32(lbl_stream);

    if (img_magic != 2051U) {
        throw std::runtime_error("Unexpected IDX image magic number: " + std::to_string(img_magic));
    }
    if (lbl_magic != 2049U) {
        throw std::runtime_error("Unexpected IDX label magic number: " + std::to_string(lbl_magic));
    }
    if (n_images != n_labels) {
        throw std::runtime_error("MNIST image/label count mismatch");
    }

    RawMnistData data;
    data.images.reserve(n_images);
    data.labels.resize(n_labels);

    std::vector<uint8_t> image_buf(rows * cols);
    for (uint32_t i = 0; i < n_images; ++i) {
        img_stream.read(reinterpret_cast<char*>(image_buf.data()), static_cast<std::streamsize>(image_buf.size()));
        if (!img_stream) {
            throw std::runtime_error("Failed while reading MNIST image payload");
        }

        cv::Mat image_u8(static_cast<int>(rows), static_cast<int>(cols), CV_8UC1, image_buf.data());
        cv::Mat image_f32;
        image_u8.convertTo(image_f32, CV_32FC1, 1.0 / 255.0);
        data.images.push_back(image_f32.clone());
    }

    lbl_stream.read(reinterpret_cast<char*>(data.labels.data()), static_cast<std::streamsize>(data.labels.size()));
    if (!lbl_stream) {
        throw std::runtime_error("Failed while reading MNIST label payload");
    }

    return data;
}

void prepare_mnistdd_dataset(const Config& cfg) {
    // SECTION 2 -- DATA GENERATION
    ensure_directory(cfg.output_dir);
    ensure_directory(cfg.dataset_dir);

    ensure_directory(cfg.dataset_dir / "images" / "train");
    ensure_directory(cfg.dataset_dir / "images" / "val");
    ensure_directory(cfg.dataset_dir / "images" / "test");

    ensure_directory(cfg.dataset_dir / "masks" / "train");
    ensure_directory(cfg.dataset_dir / "masks" / "val");
    ensure_directory(cfg.dataset_dir / "masks" / "test");

    ensure_directory(cfg.dataset_dir / "labels" / "train");
    ensure_directory(cfg.dataset_dir / "labels" / "val");
    ensure_directory(cfg.dataset_dir / "labels" / "test");

    const bool train_ready = split_complete(cfg, "train", cfg.num_train);
    const bool val_ready = split_complete(cfg, "val", cfg.num_val);
    const bool test_ready = split_complete(cfg, "test", cfg.num_test);

    if (train_ready && val_ready && test_ready) {
        std::cout << "Dataset already exists. Skipping regeneration." << std::endl;
        write_data_yaml(cfg);
        save_sample_grid(cfg);
        return;
    }

    if (!std::filesystem::exists(cfg.mnist_train_images) ||
        !std::filesystem::exists(cfg.mnist_train_labels) ||
        !std::filesystem::exists(cfg.mnist_test_images) ||
        !std::filesystem::exists(cfg.mnist_test_labels)) {
        throw std::runtime_error(
            "Raw MNIST IDX files are missing. Run scripts/download_mnist.py first.");
    }

    std::cout << "Loading raw MNIST IDX files..." << std::endl;
    const RawMnistData mnist_train = load_mnist_idx(cfg.mnist_train_images, cfg.mnist_train_labels);
    const RawMnistData mnist_test = load_mnist_idx(cfg.mnist_test_images, cfg.mnist_test_labels);
    std::cout << "MNIST loaded. Train=" << mnist_train.images.size()
              << " Test=" << mnist_test.images.size() << std::endl;

    generate_split(mnist_train, cfg, cfg.num_train, "train", true);
    generate_split(mnist_train, cfg, cfg.num_val, "val", false);
    generate_split(mnist_test, cfg, cfg.num_test, "test", false);

    write_data_yaml(cfg);
    save_sample_grid(cfg);

    std::cout << "Dataset generation complete: " << cfg.dataset_dir.string() << std::endl;
}

std::vector<std::filesystem::path> list_split_images(const Config& cfg,
                                                     const std::string& split) {
    const auto image_dir = cfg.dataset_dir / "images" / split;
    if (!std::filesystem::exists(image_dir)) {
        throw std::runtime_error("Image split not found: " + image_dir.string());
    }

    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(image_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() == ".png") {
            paths.push_back(entry.path());
        }
    }

    std::sort(paths.begin(), paths.end());
    return paths;
}

std::vector<SceneAnnotation> load_split_annotations(const Config& cfg,
                                                    const std::string& split,
                                                    size_t expected_count) {
    const auto label_dir = cfg.dataset_dir / "labels" / split;
    if (!std::filesystem::exists(label_dir)) {
        throw std::runtime_error("Label split not found: " + label_dir.string());
    }

    size_t count = expected_count;
    if (count == 0) {
        count = list_split_images(cfg, split).size();
    }

    std::vector<SceneAnnotation> annotations;
    annotations.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        SceneAnnotation scene;
        const auto path = label_dir / (scene_name(i) + ".txt");
        if (!std::filesystem::exists(path)) {
            annotations.push_back(scene);
            continue;
        }

        std::ifstream is(path);
        if (!is) {
            throw std::runtime_error("Failed to open label file: " + path.string());
        }

        while (is) {
            int cls = -1;
            float cx = 0.0F;
            float cy = 0.0F;
            float w = 0.0F;
            float h = 0.0F;
            is >> cls >> cx >> cy >> w >> h;
            if (!is) {
                break;
            }
            const float x1 = (cx - 0.5F * w) * static_cast<float>(cfg.canvas_size);
            const float y1 = (cy - 0.5F * h) * static_cast<float>(cfg.canvas_size);
            const float x2 = (cx + 0.5F * w) * static_cast<float>(cfg.canvas_size);
            const float y2 = (cy + 0.5F * h) * static_cast<float>(cfg.canvas_size);
            scene.boxes.push_back(BoundingBox{x1, y1, x2, y2, cls, 1.0F});
        }

        annotations.push_back(scene);
    }

    return annotations;
}

}  // namespace change
