#!/usr/bin/env python3
"""
Train a real YOLO detector first, then a U-Net segmenter, sequentially.

Hardcoded for Kaggle 2x T4 (16GB) with fixed devices, batches, AMP, and data
loader settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO

KAGGLE_DEVICE_IDS = [0, 1]
YOLO_DEVICE = "0,1"
YOLO_MODEL = "yolov8n.pt"
YOLO_BATCH = 256
UNET_BATCH = 512
YOLO_WORKERS = 2
UNET_WORKERS = 2
PREFETCH_FACTOR = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True
AMP_ENABLED = True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_posix(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def write_yolo_data_yaml(dataset_dir: Path, output_dir: Path) -> Path:
    yaml_path = output_dir / "data_yolo.yaml"
    text = "\n".join(
        [
            f"path: {to_posix(dataset_dir)}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "nc: 10",
            "names: ['0','1','2','3','4','5','6','7','8','9']",
            "",
        ]
    )
    yaml_path.write_text(text, encoding="utf-8")
    return yaml_path


def extract_yolo_metrics(val_result) -> Dict[str, float]:
    stats: Dict[str, float] = {
        "precision": float("nan"),
        "recall": float("nan"),
        "map50": float("nan"),
        "map50_95": float("nan"),
    }

    box = getattr(val_result, "box", None)
    if box is None:
        return stats

    mapping = {
        "precision": "mp",
        "recall": "mr",
        "map50": "map50",
        "map50_95": "map",
    }

    for out_key, attr in mapping.items():
        value = getattr(box, attr, None)
        if value is not None:
            stats[out_key] = float(value)

    return stats


class SegmentationDataset(Dataset):
    def __init__(self, dataset_dir: Path, split: str, num_classes: int) -> None:
        self.dataset_dir = dataset_dir
        self.split = split
        self.num_classes = num_classes

        image_dir = dataset_dir / "images" / split
        mask_dir = dataset_dir / "masks" / split

        self.image_paths = sorted(image_dir.glob("*.png"))
        self.mask_paths = [mask_dir / p.name for p in self.image_paths]

        if not self.image_paths:
            raise RuntimeError(f"No images found for split '{split}' in {image_dir}")

        missing = [str(p) for p in self.mask_paths if not p.exists()]
        if missing:
            raise RuntimeError(
                f"Missing {len(missing)} mask files in split '{split}'. Example: {missing[0]}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.int64)
        mask = np.clip(mask, 0, self.num_classes - 1)

        image_t = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        mask_t = torch.from_numpy(mask).long()
        return image_t, mask_t


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.seq = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        if diff_x != 0 or diff_y != 0:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 11, base_channels: int = 32) -> None:
        super().__init__()

        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2

        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)

        self.up1 = Up(c5, c4, c4)
        self.up2 = Up(c4, c3, c3)
        self.up3 = Up(c3, c2, c2)
        self.up4 = Up(c2, c1, c1)

        self.out = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        y = self.up1(x5, x4)
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4(y, x1)
        return self.out(y)


def estimate_class_weights(mask_paths: List[Path], num_classes: int, max_samples: int = 1200) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)

    if not mask_paths:
        return torch.ones(num_classes, dtype=torch.float32)

    stride = max(1, len(mask_paths) // max_samples)
    sampled = mask_paths[::stride][:max_samples]

    for mask_path in sampled:
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.int64)
        mask = np.clip(mask, 0, num_classes - 1)
        hist = np.bincount(mask.reshape(-1), minlength=num_classes)
        counts += hist

    total = counts.sum()
    if total <= 0:
        return torch.ones(num_classes, dtype=torch.float32)

    freq = counts / total
    weights = 1.0 / np.sqrt(freq + 1.0e-8)

    # Reduce dominance of background while still keeping it in loss.
    weights[0] *= 0.35
    weights /= np.mean(weights)

    return torch.tensor(weights, dtype=torch.float32)


def dice_loss_foreground(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    target_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = (probs * target_one_hot).sum(dims)
    denominator = probs.sum(dims) + target_one_hot.sum(dims)

    dice = 1.0 - (2.0 * intersection + 1.0e-6) / (denominator + 1.0e-6)
    if num_classes > 1:
        return dice[1:].mean()
    return dice.mean()


def combined_segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    num_classes: int,
    dice_weight: float,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, weight=class_weights)
    dice = dice_loss_foreground(logits, targets, num_classes)
    return ce + dice_weight * dice


@dataclass
class SegmentationMetrics:
    class_iou: List[float]
    class_pixel_accuracy: List[float]
    mean_iou: float
    pixel_accuracy: float
    foreground_accuracy: float


def evaluate_segmentation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> SegmentationMetrics:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    model.eval()
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            preds = logits.argmax(dim=1)

            k = preds.reshape(-1) * num_classes + masks.reshape(-1)
            bincount = torch.bincount(k.to(torch.int64).cpu(), minlength=num_classes * num_classes)
            confusion += bincount.view(num_classes, num_classes).to(torch.float64)

    inter = confusion.diag()
    pred_total = confusion.sum(1)
    gt_total = confusion.sum(0)
    union = pred_total + gt_total - inter

    class_iou: List[float] = []
    class_acc: List[float] = []
    valid = 0
    iou_sum = 0.0

    for c in range(num_classes):
        inter_c = float(inter[c].item())
        union_c = float(union[c].item())
        gt_c = float(gt_total[c].item())

        if union_c > 0.0:
            iou_c = inter_c / union_c
            iou_sum += iou_c
            valid += 1
        else:
            iou_c = 0.0

        if gt_c > 0.0:
            acc_c = inter_c / gt_c
        else:
            acc_c = 0.0

        class_iou.append(iou_c)
        class_acc.append(acc_c)

    mean_iou = iou_sum / valid if valid > 0 else 0.0

    diag_sum = float(inter.sum().item())
    total_sum = float(confusion.sum().item())
    pixel_accuracy = diag_sum / total_sum if total_sum > 0 else 0.0

    fg_inter = float(inter[1:].sum().item())
    fg_total = float(gt_total[1:].sum().item())
    foreground_accuracy = fg_inter / fg_total if fg_total > 0 else 0.0

    return SegmentationMetrics(
        class_iou=class_iou,
        class_pixel_accuracy=class_acc,
        mean_iou=mean_iou,
        pixel_accuracy=pixel_accuracy,
        foreground_accuracy=foreground_accuracy,
    )


def write_unet_history_csv(path: Path, rows: List[Tuple[int, float, float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr"])
        for epoch, train_loss, val_loss, lr in rows:
            writer.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}", f"{lr:.8f}"])


def train_unet_stage(args, dataset_dir: Path, output_dir: Path) -> Tuple[Path, SegmentationMetrics]:
    num_classes = args.num_classes_seg

    train_ds = SegmentationDataset(dataset_dir, "train", num_classes)
    val_ds = SegmentationDataset(dataset_dir, "val", num_classes)
    test_ds = SegmentationDataset(dataset_dir, "test", num_classes)

    pin_memory = PIN_MEMORY
    persistent_workers = PERSISTENT_WORKERS

    train_loader = DataLoader(
        train_ds,
        batch_size=UNET_BATCH,
        shuffle=True,
        num_workers=UNET_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=PREFETCH_FACTOR,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=UNET_BATCH,
        shuffle=False,
        num_workers=UNET_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=PREFETCH_FACTOR,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=UNET_BATCH,
        shuffle=False,
        num_workers=UNET_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=PREFETCH_FACTOR,
    )

    device_ids = KAGGLE_DEVICE_IDS
    primary_device = torch.device("cuda:0")

    model: nn.Module = UNet(
        in_channels=3,
        num_classes=num_classes,
        base_channels=args.unet_base_channels,
    )

    model = nn.DataParallel(model, device_ids=device_ids)

    model = model.to(primary_device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.unet_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.unet_lr_patience,
    )

    class_weights = estimate_class_weights(train_ds.mask_paths, num_classes).to(primary_device)

    amp_enabled = AMP_ENABLED
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val_loss = math.inf
    no_improve_epochs = 0

    ckpt_path = output_dir / "unet_best.pth"
    history_csv = output_dir / "unet_history.csv"
    history_rows: List[Tuple[int, float, float, float]] = []

    print("\n[UNet] Training starts")
    print(f"[UNet] Train/Val/Test sizes = {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"[UNet] Device = {primary_device} | DataParallel = on")
    print(f"[UNet] Batch = {UNET_BATCH} | AMP = {'on' if amp_enabled else 'off'}")

    for epoch in range(1, args.unet_epochs + 1):
        model.train()
        train_running = 0.0
        train_seen = 0

        for images, masks in train_loader:
            images = images.to(primary_device, non_blocking=True)
            masks = masks.to(primary_device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.float16):
                logits = model(images)
                loss = combined_segmentation_loss(
                    logits,
                    masks,
                    class_weights,
                    num_classes,
                    args.dice_weight,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = images.size(0)
            train_running += float(loss.detach().item()) * bs
            train_seen += bs

        train_loss = train_running / max(1, train_seen)

        model.eval()
        val_running = 0.0
        val_seen = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(primary_device, non_blocking=True)
                masks = masks.to(primary_device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.float16):
                    logits = model(images)
                    loss = combined_segmentation_loss(
                        logits,
                        masks,
                        class_weights,
                        num_classes,
                        args.dice_weight,
                    )

                bs = images.size(0)
                val_running += float(loss.detach().item()) * bs
                val_seen += bs

        val_loss = val_running / max(1, val_seen)
        scheduler.step(val_loss)

        current_lr = float(optimizer.param_groups[0]["lr"])
        history_rows.append((epoch, train_loss, val_loss, current_lr))

        improved = val_loss < (best_val_loss - args.min_delta)
        if improved:
            best_val_loss = val_loss
            no_improve_epochs = 0
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "num_classes": num_classes,
                    "base_channels": args.unet_base_channels,
                    "best_val_loss": best_val_loss,
                },
                ckpt_path,
            )
        else:
            no_improve_epochs += 1

        print(
            f"[UNet] Epoch {epoch}/{args.unet_epochs} | "
            f"Train {train_loss:.5f} | Val {val_loss:.5f} | "
            f"LR {current_lr:.6g} | P {no_improve_epochs}/{args.unet_patience}"
            + (" | saved" if improved else "")
        )

        if no_improve_epochs >= args.unet_patience:
            print(f"[UNet] Early stopping at epoch {epoch}")
            break

    write_unet_history_csv(history_csv, history_rows)

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=primary_device)
        model_to_load = model.module if isinstance(model, nn.DataParallel) else model
        model_to_load.load_state_dict(ckpt["model_state_dict"])

    metrics = evaluate_segmentation(model, test_loader, primary_device, num_classes)
    return ckpt_path, metrics


def train_yolo_stage(args, dataset_dir: Path, output_dir: Path) -> Tuple[Path, Dict[str, float]]:
    yolo_yaml = write_yolo_data_yaml(dataset_dir, output_dir)

    yolo_device = YOLO_DEVICE
    val_device = 0

    print("\n[YOLO] Training starts")
    print(f"[YOLO] Data YAML = {yolo_yaml}")
    print(f"[YOLO] Model = {YOLO_MODEL} | Batch = {YOLO_BATCH}")
    print(f"[YOLO] Device = {yolo_device}")

    model = YOLO(YOLO_MODEL)
    model.train(
        data=str(yolo_yaml),
        epochs=args.yolo_epochs,
        imgsz=args.yolo_imgsz,
        batch=YOLO_BATCH,
        workers=YOLO_WORKERS,
        project=str(output_dir),
        name="yolo_train",
        exist_ok=True,
        device=yolo_device,
        seed=args.seed,
        amp=True,
    )

    best_path = output_dir / "yolo_train" / "weights" / "best.pt"
    if not best_path.exists():
        fallback = output_dir / "yolo_train" / "weights" / "last.pt"
        if not fallback.exists():
            raise RuntimeError("YOLO training finished but no best.pt/last.pt found")
        best_path = fallback

    best_model = YOLO(str(best_path))
    val_result = best_model.val(
        data=str(yolo_yaml),
        split="test",
        imgsz=args.yolo_imgsz,
        batch=YOLO_BATCH,
        workers=YOLO_WORKERS,
        device=val_device,
    )

    stats = extract_yolo_metrics(val_result)

    metrics_json_path = output_dir / "yolo_test_metrics.json"
    metrics_json_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(
        "[YOLO] Test | "
        f"P {stats['precision']:.4f} | R {stats['recall']:.4f} | "
        f"mAP50 {stats['map50']:.4f} | mAP50-95 {stats['map50_95']:.4f}"
    )

    return best_path, stats


def write_final_report(
    report_path: Path,
    yolo_weights_path: Path,
    yolo_stats: Dict[str, float],
    seg_metrics: SegmentationMetrics,
) -> None:
    lines: List[str] = []
    lines.append("=================================================================")
    lines.append("  BCO074C MINOR PROJECT - FINAL RESULTS REPORT")
    lines.append("=================================================================")
    lines.append("")

    lines.append("--- OBJECT DETECTION (YOLO) ---")
    lines.append(f"  Weights Path        : {yolo_weights_path}")
    lines.append(f"  Precision           : {yolo_stats['precision']:.4f}")
    lines.append(f"  Recall              : {yolo_stats['recall']:.4f}")
    lines.append(f"  mAP@0.50            : {yolo_stats['map50']:.4f}")
    lines.append(f"  mAP@0.50:0.95       : {yolo_stats['map50_95']:.4f}")
    lines.append("")

    lines.append("--- SEMANTIC SEGMENTATION (U-Net) ---")
    lines.append(f"  Seg Accuracy (fg)   : {seg_metrics.foreground_accuracy:.4f}")
    lines.append(f"  Mean IoU            : {seg_metrics.mean_iou:.4f}")
    lines.append(f"  Pixel Accuracy      : {seg_metrics.pixel_accuracy:.4f}")
    lines.append("")
    lines.append("        Class |      IoU |  Pixel Acc")
    lines.append("  ------------------------------------")

    for class_id, (iou, acc) in enumerate(zip(seg_metrics.class_iou, seg_metrics.class_pixel_accuracy)):
        name = "Background" if class_id == 0 else f"Digit {class_id - 1}"
        lines.append(f"  {name:>10} | {iou:8.4f} | {acc:9.4f}")

    lines.append("")
    lines.append("=================================================================")
    lines.append("  END OF REPORT")
    lines.append("=================================================================")

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    print("\n" + report_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO first, then U-Net segmentation")

    parser.add_argument("--dataset_dir", type=str, default="/kaggle/working/output/digit_dataset")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/output")

    parser.add_argument("--yolo_epochs", type=int, default=40)
    parser.add_argument("--yolo_imgsz", type=int, default=128)

    parser.add_argument("--num_classes_seg", type=int, default=11)
    parser.add_argument("--unet_epochs", type=int, default=30)
    parser.add_argument("--unet_base_channels", type=int, default=32)
    parser.add_argument("--unet_lr", type=float, default=1.0e-3)
    parser.add_argument("--unet_patience", type=int, default=8)
    parser.add_argument("--unet_lr_patience", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--dice_weight", type=float, default=0.7)
    parser.add_argument("--min_delta", type=float, default=1.0e-4)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def validate_dataset_layout(dataset_dir: Path) -> None:
    required_dirs = [
        dataset_dir / "images" / "train",
        dataset_dir / "images" / "val",
        dataset_dir / "images" / "test",
        dataset_dir / "labels" / "train",
        dataset_dir / "labels" / "val",
        dataset_dir / "labels" / "test",
        dataset_dir / "masks" / "train",
        dataset_dir / "masks" / "val",
        dataset_dir / "masks" / "test",
    ]
    missing = [str(p) for p in required_dirs if not p.exists()]
    if missing:
        raise RuntimeError(
            "Dataset layout is incomplete. Missing directories:\n- " + "\n- ".join(missing)
        )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cudnn.benchmark = True

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_dataset_layout(dataset_dir)

    print("Hardcoded Kaggle 2x T4 setup:")
    print(
        f"  devices={YOLO_DEVICE} | YOLO_BATCH={YOLO_BATCH} | UNET_BATCH={UNET_BATCH} | "
        f"workers={UNET_WORKERS} | AMP={'on' if AMP_ENABLED else 'off'}"
    )

    yolo_weights_path, yolo_stats = train_yolo_stage(args, dataset_dir, output_dir)
    unet_ckpt, seg_metrics = train_unet_stage(args, dataset_dir, output_dir)

    report_path = output_dir / "report.txt"
    write_final_report(report_path, yolo_weights_path, yolo_stats, seg_metrics)

    summary = {
        "yolo_weights": str(yolo_weights_path),
        "unet_checkpoint": str(unet_ckpt),
        "report": str(report_path),
        "yolo_metrics": yolo_stats,
        "segmentation": {
            "mean_iou": seg_metrics.mean_iou,
            "pixel_accuracy": seg_metrics.pixel_accuracy,
            "foreground_accuracy": seg_metrics.foreground_accuracy,
        },
    }

    (output_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSaved summary:", output_dir / "pipeline_summary.json")


if __name__ == "__main__":
    main()
