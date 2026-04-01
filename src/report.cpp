/*
 * report.cpp
 * Purpose: SECTION 9 -- final text report generation to stdout and report.txt.
 */

#include "utils.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace change {

void write_final_report(const Config& cfg,
                        const DetectionMetrics& det_metrics,
                        const SegmentationMetrics& seg_metrics) {
    std::ostringstream report;

    report << "=================================================================\n";
    report << "  BCO074C MINOR PROJECT - FINAL RESULTS REPORT\n";
    report << "=================================================================\n\n";

    report << "--- OBJECT DETECTION (Custom Detector) ---\n";
    report << "  Detection Accuracy : " << std::fixed << std::setprecision(4)
           << det_metrics.accuracy << "  ("
           << det_metrics.total_correct << "/" << det_metrics.total_gt << " objects)\n";
    report << "  Overall Precision  : " << det_metrics.overall_precision << "\n";
    report << "  Overall Recall     : " << det_metrics.overall_recall << "\n";
    report << "  Overall F1 Score   : " << det_metrics.overall_f1 << "\n\n";

    report << "   Class | Precision |   Recall |       F1\n";
    report << "  -----------------------------------------\n";
    for (size_t c = 0; c < det_metrics.per_class.size(); ++c) {
        const auto& cls = det_metrics.per_class[c];
        report << std::setw(7) << c << " | "
               << std::setw(9) << cls.precision << " | "
               << std::setw(8) << cls.recall << " | "
               << std::setw(8) << cls.f1 << "\n";
    }
    report << "\n";

    report << "--- SEMANTIC SEGMENTATION (U-Net) ---\n";
    report << "  Seg Accuracy (fg)  : " << seg_metrics.foreground_accuracy << "\n";
    report << "  Mean IoU           : " << seg_metrics.mean_iou << "\n";
    report << "  Pixel Accuracy     : " << seg_metrics.pixel_accuracy << "\n\n";

    report << "        Class |      IoU |  Pixel Acc\n";
    report << "  ------------------------------------\n";
    for (size_t c = 0; c < seg_metrics.class_iou.size(); ++c) {
        const std::string name = (c == 0) ? "Background" : ("Digit " + std::to_string(c - 1));
        report << std::setw(12) << name << " | "
               << std::setw(8) << seg_metrics.class_iou[c] << " | "
               << std::setw(9) << seg_metrics.class_pixel_accuracy[c] << "\n";
    }

    report << "\n";
    report << "=================================================================\n";
    report << "  END OF REPORT\n";
    report << "=================================================================\n";

    const std::string report_text = report.str();
    std::cout << report_text;

    std::ofstream os(cfg.report_txt);
    if (!os) {
        throw std::runtime_error("Failed to write report file: " + cfg.report_txt.string());
    }
    os << report_text;

    std::cout << "Report saved to: " << cfg.report_txt.string() << std::endl;
}

}  // namespace change
