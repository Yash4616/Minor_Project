# Change (C++17 LibTorch + OpenCV)

This project is a full C++17 port of the MNISTDD-RGB notebook pipeline for:
- Object detection (custom lightweight detector replacing YOLO training in the legacy C++ path)
- Semantic segmentation (U-Net in LibTorch)
- Dataset generation, evaluation, visualization, and final reporting

This repo also includes a Kaggle-focused Python pipeline that trains a real YOLO detector first,
then trains U-Net segmentation second (sequentially).

## 1. Prerequisites

1. CMake >= 3.18
2. C++17 compiler (GCC/Clang/MSVC)
3. LibTorch (matching your CUDA or CPU runtime):
   https://pytorch.org/get-started/locally/
4. OpenCV 4.x

OpenCV install example (Ubuntu):

```bash
sudo apt-get update
sudo apt-get install -y libopencv-dev
```

## 2. Download Raw MNIST IDX Files

Run the helper script (downloads and decompresses raw IDX files):

```bash
python scripts/download_mnist.py --out_dir ./data
```

Expected files inside `./data`:
- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

## 3. Build

Use the exact commands below (replace LibTorch path):

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
make -j$(nproc)
```

If your LibTorch build requires old ABI, configure with:

```bash
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch -DCHANGE_FORCE_GLIBCXX_ABI0=ON
```

## 4. Run

```bash
./Change --data_dir ./data --output_dir ./output
```

Generate dataset only (skip C++ training/evaluation):

```bash
./Change --data_dir ./data --output_dir ./output --generate_only
```

## 5. Expected Output Files

Inside `./output`:
- `digit_dataset/` (generated MNISTDD-RGB images, masks, and labels)
- `detector_best.pt`
- `unet_best.pt`
- `detector_history.csv`
- `unet_history.csv`
- `report.txt`
- `mnistdd_rgb_samples.png`
- `yolo_detection_results.png`
- `unet_segmentation_results.png`
- `final_results_dashboard.png`

## 6. CUDA vs CPU Notes

- If CUDA is available, the app uses GPU automatically.
- If CUDA is not available, it falls back to CPU automatically.
- Batch sizes are reduced on CPU mode to avoid OOM.

## 7. Project Layout

- `include/config.hpp`: global constants and runtime setup
- `include/dataset.hpp`: MNISTDD dataset + data generation interfaces
- `include/unet.hpp`: U-Net module definitions
- `include/detector.hpp`: lightweight detector module and APIs
- `include/utils.hpp`: shared metrics, IO, and helper utilities
- `src/data_gen.cpp`: IDX loading + scene synthesis + dataset writing
- `src/train_detector.cpp`: detector training and decode logic
- `src/train_unet.cpp`: U-Net training loop with AMP-aware path
- `src/evaluate.cpp`: detection/segmentation metrics
- `src/visualize.cpp`: OpenCV visualization outputs
- `src/report.cpp`: final text report output
- `src/main.cpp`: top-level orchestration

## 8. Kaggle Recommended Path (YOLO -> U-Net, Sequential)

Use this path if you want real YOLO for detection and U-Net for segmentation.

1. Generate dataset (C++ binary):

```bash
./Change --data_dir ./data --output_dir ./output --generate_only
```

2. Train YOLO first, then U-Net second (sequential):

```bash
python3 -m pip install ultralytics
python3 scripts/train_yolo_then_unet.py \
   --dataset_dir ./output/digit_dataset \
   --output_dir ./output
```

This script:
- trains YOLO detection first,
- trains U-Net segmentation after YOLO,
- is hardcoded for Kaggle 2x T4 (16GB) with fixed device IDs, batches, AMP, and data loader settings,
- uses YOLOv8n with DDP (via Ultralytics) and U-Net DataParallel,
- writes final report to `./output/report.txt`.

To change batches/workers/devices, edit the constants at the top of
`scripts/train_yolo_then_unet.py`.

## 9. Final Results Report

=================================================================
  BCO074C MINOR PROJECT - FINAL RESULTS REPORT
=================================================================

--- OBJECT DETECTION (YOLO) ---
  Weights Path        : /kaggle/working/output/yolo_train/weights/best.pt
  Precision           : 0.9819
  Recall              : 0.9704
  mAP@0.50            : 0.9930
  mAP@0.50:0.95       : 0.8626

--- SEMANTIC SEGMENTATION (U-Net) ---
  Seg Accuracy (fg)   : 0.9856
  Mean IoU            : 0.9605
  Pixel Accuracy      : 0.9996

        Class |      IoU |  Pixel Acc
  ------------------------------------
  Background |   0.9998 |    0.9998
     Digit 0 |   0.9674 |    0.9919
     Digit 1 |   0.9497 |    0.9744
     Digit 2 |   0.9612 |    0.9841
     Digit 3 |   0.9612 |    0.9917
     Digit 4 |   0.9500 |    0.9858
     Digit 5 |   0.9542 |    0.9815
     Digit 6 |   0.9625 |    0.9870
     Digit 7 |   0.9528 |    0.9842
     Digit 8 |   0.9605 |    0.9890
     Digit 9 |   0.9462 |    0.9797

=================================================================
  END OF REPORT
=================================================================
