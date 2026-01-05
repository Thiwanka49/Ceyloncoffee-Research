import os
import time
import csv
import argparse
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Checkpoint helpers (TRUE resume)
# -----------------------------
def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc):
    ckpt = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "sched_state": scheduler.state_dict() if scheduler else None,
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)

    # full checkpoint format
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_acc = float(ckpt.get("best_acc", 0.0))

        if optimizer is not None and ckpt.get("optim_state") is not None:
            optimizer.load_state_dict(ckpt["optim_state"])

        if scheduler is not None and ckpt.get("sched_state") is not None:
            scheduler.load_state_dict(ckpt["sched_state"])

        print(f"✅ Resumed from: {path} | start_epoch={start_epoch} | best_acc={best_acc:.4f}")
        return start_epoch, best_acc

    # weights-only format
    model.load_state_dict(ckpt)
    print(f"⚠️ Loaded weights-only from: {path} (optimizer not resumed). Starting from epoch 0.")
    return 0, 0.0


# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


# -----------------------------
# CSV logging
# -----------------------------
def ensure_csv_header(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])


def append_csv(csv_path, row):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(row)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="datasets/coffee_defect_dataset_v2", type=str)
    ap.add_argument("--model", default="mobilenetv3_large_100", type=str)
    ap.add_argument("--img", default=512, type=int)
    ap.add_argument("--batch", default=16, type=int)
    ap.add_argument("--epochs", default=30, type=int)
    ap.add_argument("--lr", default=1e-4, type=float)
    ap.add_argument("--num_workers", default=2, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--resume", action="store_true", help="resume from models/defect_last.pt if exists")
    args = ap.parse_args()

    set_seed(args.seed)

    # Paths
    TRAIN_DIR = os.path.join(args.data_dir, "train")
    VAL_DIR = os.path.join(args.data_dir, "val")

    os.makedirs("models", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    BEST_WEIGHTS = "models/defect_best.pt"   # weights-only (best for inference)
    LAST_CKPT = "models/defect_last.pt"      # full checkpoint (for resume)
    CSV_PATH = "runs/defect_history.csv"

    # Sanity checks
    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(f"Missing train dir: {TRAIN_DIR}")
    if not os.path.exists(VAL_DIR):
        raise FileNotFoundError(f"Missing val dir: {VAL_DIR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("✅ Device:", device)

    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((args.img, args.img)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((args.img, args.img)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Dataset
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=val_tf)

    # IMPORTANT: verify class names
    print("✅ Classes found:", train_ds.classes)
    if len(train_ds.classes) != 3:
        raise ValueError("Expected 3 classes (good/broken/severe_defect). Check your folders.")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Model
    model = timm.create_model(args.model, pretrained=True, num_classes=3)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    # Resume
    start_epoch = 0
    best_acc = 0.0
    if args.resume and os.path.exists(LAST_CKPT):
        start_epoch, best_acc = load_checkpoint(LAST_CKPT, model, optimizer, scheduler, device=device)

    # CSV logging
    ensure_csv_header(CSV_PATH)

    print("\n--- Training config ---")
    print("Data:", args.data_dir)
    print("Model:", args.model)
    print("Image:", args.img)
    print("Batch:", args.batch)
    print("Epochs:", args.epochs)
    print("LR:", args.lr)
    print("Resume:", args.resume)
    print("-----------------------\n")

    # Train loop
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        # Save full checkpoint every epoch
        save_checkpoint(LAST_CKPT, model, optimizer, scheduler, epoch, best_acc)

        # Save best weights
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), BEST_WEIGHTS)
            print(f"🏆 Best updated -> {BEST_WEIGHTS} | val_acc={best_acc:.4f}")

        # Log CSV
        append_csv(CSV_PATH, [epoch, train_loss, train_acc, val_loss, val_acc, lr_now])

        dt = time.time() - t0
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train: loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val: loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"lr={lr_now:.6f} | {dt:.1f}s"
        )

    print("\n✅ DONE")
    print("Best weights:", BEST_WEIGHTS)
    print("Resume checkpoint:", LAST_CKPT)
    print("History CSV:", CSV_PATH)


if __name__ == "__main__":
    main()
