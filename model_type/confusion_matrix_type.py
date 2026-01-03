# model_type/confusion_matrix_type.py
import os
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

CKPT = "models/bean_type_best.pt"
VAL_DIR = "datasets/bean_type_dataset/val"
SAVE_PATH = "confusion_matrices/bean_type_confusion_matrix.png"

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(CKPT, map_location=device)
model_name = ckpt["model_name"]
img_size = ckpt["img_size"]
class_names = ckpt["class_names"]

model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
model.load_state_dict(ckpt["model_state"])
model.to(device).eval()

tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

ds = datasets.ImageFolder(VAL_DIR, transform=tf)
loader = DataLoader(ds, batch_size=16, shuffle=False)

cm = np.zeros((len(class_names), len(class_names)), dtype=int)

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        preds = torch.argmax(model(x), dim=1).cpu().numpy()
        y = y.numpy()
        for t, p in zip(y, preds):
            cm[t, p] += 1

plt.figure()
plt.imshow(cm)
plt.title("Bean Type Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(range(len(class_names)), class_names)
plt.yticks(range(len(class_names)), class_names)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300)
plt.close()

print("✅ Saved:", SAVE_PATH)
