import os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DATA_DIR = "datasets/bean_type_dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")

CLASS_NAMES = ["arabica", "robusta"]   # folder names must match
IMG_SIZE = 512
BATCH = 1
EPOCHS = 30
NUM_WORKERS = 0
LR = 3e-4

SAVE_PATH = "models/bean_type_best.pt"
os.makedirs("models", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# TRANSFORMS
# =========================
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(8),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406),
                         std=(0.229,0.224,0.225)),
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406),
                         std=(0.229,0.224,0.225)),
])

# =========================
# DATASETS
# =========================
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
val_ds   = datasets.ImageFolder(VAL_DIR, transform=val_tfms)

print("Detected classes:", train_ds.classes)
assert train_ds.classes == CLASS_NAMES, "❌ Class folder names mismatch!"

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# =========================
# MODEL
# =========================
model = timm.create_model(
    "mobilenetv3_large_100",
    pretrained=True,
    num_classes=len(CLASS_NAMES)
)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

# =========================
# EVALUATION
# =========================
def evaluate():
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return loss_sum / total, correct / total

# =========================
# TRAINING LOOP
# =========================
best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    start = time.time()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        pbar.set_postfix(loss=loss.item())

    train_loss = running_loss / total
    train_acc = correct / total

    val_loss, val_acc = evaluate()

    print(f"\nEpoch {epoch} finished in {time.time()-start:.1f}s")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "model_name": "mobilenetv3_large_100",
            "img_size": IMG_SIZE,
            "class_names": CLASS_NAMES,
            "model_state": model.state_dict(),
        }, SAVE_PATH)
        print(f"✅ Saved BEST model to {SAVE_PATH}")

print("\n🎉 Training completed. Best Val Acc:", best_acc)
