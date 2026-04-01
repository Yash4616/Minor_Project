/*
 * main.cpp
 * Purpose: Entry point orchestrating all notebook-equivalent pipeline sections.
 */

#include "config.hpp"
#include "dataset.hpp"
#include "detector.hpp"
#include "unet.hpp"
#include "utils.hpp"

#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_usage(const char* exe_name) {
    std::cout << "Usage: " << exe_name
              << " --data_dir ./data --output_dir ./output [--generate_only]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path data_dir = "./data";
        std::filesystem::path output_dir = "./output";
        bool generate_only = false;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--data_dir" && i + 1 < argc) {
                data_dir = argv[++i];
            } else if (arg == "--output_dir" && i + 1 < argc) {
                output_dir = argv[++i];
            } else if (arg == "--generate_only") {
                generate_only = true;
            } else if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else {
                std::cerr << "Unknown argument: " << arg << std::endl;
                print_usage(argv[0]);
                return 1;
            }
        }

        const auto cfg = change::make_config(data_dir, output_dir);

        change::ensure_directory(cfg.data_dir);
        change::ensure_directory(cfg.output_dir);

        // SECTION 0 -- DEPENDENCIES
        std::cout << "SECTION 0 -- DEPENDENCIES" << std::endl;
        std::cout << "Using LibTorch + OpenCV pipeline" << std::endl;

        // SECTION 1 -- CONFIGURATION
        std::cout << "SECTION 1 -- CONFIGURATION" << std::endl;
        std::cout << "Device=" << change::device_to_string(cfg.device)
                  << " | gpus=" << cfg.gpu_count
                  << " | detector_device=" << change::device_with_index_to_string(cfg.detector_device)
                  << " | unet_device=" << change::device_with_index_to_string(cfg.unet_device)
                  << " | parallel_training=" << (cfg.train_models_in_parallel ? "on" : "off")
                  << " | detector_batch=" << cfg.detector_batch
                  << " | unet_batch=" << cfg.unet_batch
                  << std::endl;

        // SECTION 2 -- DATA GENERATION
        std::cout << "SECTION 2 -- DATA GENERATION" << std::endl;
        change::prepare_mnistdd_dataset(cfg);

        if (generate_only) {
            std::cout << "Dataset generation completed (generate-only mode)." << std::endl;
            return 0;
        }

        const auto train_annotations = change::load_split_annotations(cfg, "train", static_cast<size_t>(cfg.num_train));
        const auto val_annotations = change::load_split_annotations(cfg, "val", static_cast<size_t>(cfg.num_val));
        const auto test_annotations = change::load_split_annotations(cfg, "test", static_cast<size_t>(cfg.num_test));

        change::LightweightDetector detector(cfg.num_classes_det, cfg.detector_num_anchors);
        change::UNet unet(3, cfg.num_classes_seg);

        change::DetectorTrainingResult detector_train;
        change::UNetTrainingResult unet_train;

        if (cfg.train_models_in_parallel) {
            std::cout << "SECTION 4/5 -- PARALLEL TRAINING" << std::endl;
            std::cout << "Detector on " << change::device_with_index_to_string(cfg.detector_device)
                      << " | UNet on " << change::device_with_index_to_string(cfg.unet_device)
                      << std::endl;

            auto detector_future = std::async(std::launch::async, [&cfg, &detector, &train_annotations, &val_annotations]() {
                return change::train_detector_model(cfg, detector, train_annotations, val_annotations);
            });

            auto unet_future = std::async(std::launch::async, [&cfg, &unet]() {
                return change::train_unet_model(cfg, unet);
            });

            detector_train = detector_future.get();
            unet_train = unet_future.get();
        } else {
            // SECTION 4 -- DETECTION MODEL: CUSTOM DETECTOR
            std::cout << "SECTION 4 -- DETECTION MODEL" << std::endl;
            detector_train = change::train_detector_model(cfg, detector, train_annotations, val_annotations);

            // SECTION 5 -- SEGMENTATION MODEL: U-NET
            std::cout << "SECTION 5 -- SEGMENTATION MODEL" << std::endl;
            unet_train = change::train_unet_model(cfg, unet);
        }

        std::cout << "Detector checkpoint: " << detector_train.checkpoint_path.string() << std::endl;
        std::cout << "UNet checkpoint: " << unet_train.checkpoint_path.string() << std::endl;

        // SECTION 6 -- EVALUATION
        std::cout << "SECTION 6 -- EVALUATION" << std::endl;
        const auto det_metrics = change::evaluate_detection(cfg, detector, test_annotations);
        change::print_detection_metrics(det_metrics, cfg.num_classes_det);

        const auto seg_metrics = change::evaluate_segmentation(cfg, unet);
        change::print_segmentation_metrics(seg_metrics, cfg.num_classes_seg);

        // SECTION 7 -- VISUALISATIONS
        std::cout << "SECTION 7 -- VISUALISATIONS" << std::endl;
        change::generate_visualizations(cfg, detector, unet, test_annotations, det_metrics, seg_metrics);

        // SECTION 9 -- FINAL REPORT
        std::cout << "SECTION 9 -- FINAL REPORT" << std::endl;
        change::write_final_report(cfg, det_metrics, seg_metrics);

        std::cout << "Pipeline complete." << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        return 1;
    }
}
