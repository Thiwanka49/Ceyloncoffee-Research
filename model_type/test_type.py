# model_type/test_type.py

import os
import argparse
import torch
import timm
from PIL import Image
from torchvision import transforms


def is_image_file(p: str) -> bool:
    return p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def load_checkpoint(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    # Defaults (in case metadata is not saved)
    model_name = "efficientnet_b0"
    img_size = 320
    class_names = ["arabica", "robusta"]

    # If checkpoint is a dict, read metadata
    if isinstance(ckpt, dict):
        model_name = ckpt.get("model_name", model_name)
        img_size = ckpt.get("img_size", img_size)
        class_names = ckpt.get("class_names", class_names)

    # Build model
    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))

    # Load weights
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        # If it's directly a state_dict
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()

    return model, img_size, class_names


def get_transform(img_size: int):
    # Note: normalize is ImageNet standard; should match your training pipeline
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


@torch.no_grad()
def predict_one(model, transform, device, img_path: str, class_names):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu()

    pred_idx = int(torch.argmax(probs).item())
    pred_name = class_names[pred_idx]

    return pred_name, probs.tolist()


def collect_images(path: str):
    if os.path.isfile(path):
        if not is_image_file(path):
            return []
        return [path]

    if os.path.isdir(path):
        files = []
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp) and is_image_file(fp):
                files.append(fp)
        files.sort()
        return files

    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="models/bean_type_best.pt",
        help="Path to checkpoint (default: models/bean_type_best.pt)"
    )
    parser.add_argument(
        "--path",
        default="datasets/bean_type_dataset/val/arabica",
        help="Image file OR folder to test (default: val/arabica folder)"
    )
    parser.add_argument(
        "--random",
        type=int,
        default=0,
        help="If >0 and --path is a folder, randomly test N images"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.ckpt):
        print(f"❌ Checkpoint not found: {args.ckpt}")
        print("👉 Make sure this file exists: models/bean_type_best.pt")
        return

    model, img_size, class_names = load_checkpoint(args.ckpt, device)
    transform = get_transform(img_size)

    images = collect_images(args.path)

    if not images:
        print(f"❌ No images found in: {args.path}")
        print("👉 Provide a valid image file or a folder containing images.")
        return

    # Random selection
    if args.random > 0 and os.path.isdir(args.path):
        import random
        random.shuffle(images)
        images = images[:args.random]
        print(f"\n🎲 Random testing: {len(images)} images\n")
    else:
        print(f"\n🧪 Testing: {len(images)} image(s)\n")

    print(f"✅ Device: {device}")
    print(f"✅ Model: {type(model).__name__}")
    print(f"✅ Input size: {img_size}")
    print(f"✅ Classes: {class_names}\n")

    for img_path in images:
        pred_name, probs = predict_one(model, transform, device, img_path, class_names)

        print("-" * 50)
        print("Image:", img_path)
        print("Predicted:", pred_name)

        for i, cls in enumerate(class_names):
            print(f"{cls}: {probs[i]:.4f}")

    print("\n✅ Done.\n")


if __name__ == "__main__":
    main()
