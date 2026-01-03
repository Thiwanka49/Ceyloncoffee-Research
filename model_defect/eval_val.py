# model_defect/eval_val.py
import os
import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import defaultdict

CKPT = "models/defect_best.pt"
VAL_DIR = "datasets/coffee_defect_dataset_v2/val"
BATCH = 16

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(CKPT, map_location=device)
model_name = ckpt.get("model_name", "mobilenetv3_large_100")
img_size = ckpt.get("img_size", 512)
class_names = ckpt.get("class_names", ["broken", "good", "severe_defect"])

model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
model.load_state_dict(ckpt["model_state"])
model.to(device).eval()

tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

ds = datasets.ImageFolder(VAL_DIR, transform=tf)
loader = DataLoader(ds, batch_size=BATCH, shuffle=False)

# confusion[true][pred]
confusion = torch.zeros(len(class_names), len(class_names), dtype=torch.int64)
per_class_correct = defaultdict(int)
per_class_total = defaultdict(int)

@torch.no_grad()
def run():
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        probs = torch.softmax(model(x), dim=1)
        pred = probs.argmax(dim=1)

        for t, p in zip(y.cpu().tolist(), pred.cpu().tolist()):
            confusion[t, p] += 1
            per_class_total[t] += 1
            if t == p:
                per_class_correct[t] += 1

    total = confusion.sum().item()
    correct = confusion.diag().sum().item()
    print("\n✅ Overall Acc:", correct / total)

    print("\n📌 Per-class accuracy:")
    for i, name in enumerate(class_names):
        acc = per_class_correct[i] / max(1, per_class_total[i])
        print(f"  {name}: {acc:.4f}  ({per_class_correct[i]}/{per_class_total[i]})")

    print("\n🧾 Confusion Matrix (rows=true, cols=pred):")
    header = " " * 14 + " ".join([f"{n:>14}" for n in class_names])
    print(header)
    for i, name in enumerate(class_names):
        row = " ".join([f"{confusion[i,j].item():>14}" for j in range(len(class_names))])
        print(f"{name:>14} {row}")
    print("")

run()
