# model_defect/train_defect.py
import os
import time
import math
import copy
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm


def count_per_class(folder, class_names):
    counts = []
    for c in class_names:
        p = os.path.join(folder, c)
        counts.append(len(os.listdir(p)) if os.path.exists(p) else 0)
    return counts


def build_transforms(img_size: int):
    # Stronger augmentation to generalize better (mobile photos, lighting changes)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf


def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        if device == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/coffee_defect_dataset_v2", help="dataset root")
    parser.add_argument("--model", default="mobilenetv3_large_100", help="timm model name")
    parser.add_argument("--imgsz", type=int, default=512, help="image size")
    parser.add_argument("--epochs", type=int, default=25, help="epochs")
    parser.add_argument("--batch", type=int, default=16, help="batch size (adjust if VRAM low)")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--patience", type=int, default=6, help="early stopping patience")
    parser.add_argument("--out", default="models/defect_best.pt", help="output checkpoint path")
    args = parser.parse_args()

    train_dir = os.path.join(args.data, "train")
    val_dir = os.path.join(args.data, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Dataset folders not found: {train_dir} / {val_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("✅ Device:", device)

    train_tf, val_tf = build_transforms(args.imgsz)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    class_names = train_ds.classes
    print("✅ Classes:", class_names)

    train_counts = count_per_class(train_dir, class_names)
    val_counts = count_per_class(val_dir, class_names)
    print("✅ Train counts:", dict(zip(class_names, train_counts)))
    print("✅ Val counts:", dict(zip(class_names, val_counts)))

    # Class-weighted loss to reduce bias toward "good"
    # weight_i = total / (num_classes * count_i)
    total = sum(train_counts)
    weights = [total / (len(class_names) * max(1, c)) for c in train_counts]
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print("✅ Class weights:", {class_names[i]: float(weights[i]) for i in range(len(class_names))})

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=(device == "cuda"))

    model = timm.create_model(args.model, pretrained=True, num_classes=len(class_names))
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val_acc = 0.0
    best_state = None
    best_epoch = -1
    patience_left = args.patience

    print("\n🚀 Training started...\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"time={dt:.1f}s")

        # Early stopping + save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = args.patience

            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({
                "model_state": best_state,
                "model_name": args.model,
                "img_size": args.imgsz,
                "class_names": class_names,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
            }, args.out)
            print(f"💾 Saved BEST -> {args.out} (val_acc={best_val_acc:.4f})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"\n🛑 Early stopping at epoch {epoch}. Best epoch was {best_epoch} (val_acc={best_val_acc:.4f})")
                break

    print(f"\n✅ Training finished. Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"✅ Best model saved at: {args.out}")


if __name__ == "__main__":
    main()
