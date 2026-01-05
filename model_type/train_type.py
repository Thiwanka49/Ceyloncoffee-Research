import os, time, csv, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

CLASS_NAMES = ["arabica", "robusta"]

def build_loaders(data_dir, img_size, batch, workers):
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_dir, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers)
    val_loader   = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers)

    return train_loader, val_loader, train_ds.class_to_idx

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total

def train(args):
    os.makedirs("models", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("✅ Device:", device)

    train_loader, val_loader, class_to_idx = build_loaders(
        args.data, args.imgsz, args.batch, args.workers
    )
    print("✅ Class index:", class_to_idx)

    model = timm.create_model(args.model, pretrained=True, num_classes=len(CLASS_NAMES))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # CSV history
    history_path = "runs/type_history.csv"
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val = 0.0
    patience_count = 0
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # write to csv
        with open(history_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc])

        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"time={time.time()-t0:.1f}s")

        # save best
        if val_acc > best_val:
            best_val = val_acc
            patience_count = 0
            ckpt = {
                "model_name": args.model,
                "img_size": args.imgsz,
                "class_names": CLASS_NAMES,
                "model_state": model.state_dict(),
            }
            torch.save(ckpt, "models/bean_type_best.pt")
            print(f"💾 Saved BEST -> models/bean_type_best.pt (val_acc={best_val:.4f})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\n🛑 Early stopping at epoch {epoch+1}. Best val_acc={best_val:.4f}")
                break

    print("\n✅ Training finished.")
    print("✅ Best Val Acc:", best_val)
    print("✅ History saved at:", history_path)
    print("⏱️ Total time:", round(time.time()-start, 1), "s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="datasets/bean_type_dataset")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--model", type=str, default="mobilenetv3_large_100")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()
    train(args)
