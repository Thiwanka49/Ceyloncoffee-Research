# model_defect/test_defect.py
import os
import argparse
import random
import torch
import timm
from PIL import Image
from torchvision import transforms


def is_image_file(p: str) -> bool:
    return p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def collect_images(path: str):
    if os.path.isfile(path):
        return [path] if is_image_file(path) else []
    if os.path.isdir(path):
        imgs = []
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp) and is_image_file(fp):
                imgs.append(fp)
        imgs.sort()
        return imgs
    return []


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    model_name = ckpt.get("model_name", "mobilenetv3_large_100")
    img_size = int(ckpt.get("img_size", 512))
    class_names = ckpt.get("class_names", ["broken", "good", "severe_defect"])

    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return model, img_size, class_names


def build_transform(img_size: int, center_crop: float):
    # center_crop: 0.0 disables it; otherwise use (0.5–0.95) like 0.85
    tf_list = []

    if center_crop and center_crop > 0:
        # CenterCrop expects pixels; we compute crop size later per-image
        # We'll handle crop in code to avoid needing the original size here.
        pass

    tf_list.extend([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transforms.Compose(tf_list)


def apply_center_crop(img: Image.Image, crop_ratio: float) -> Image.Image:
    if not crop_ratio or crop_ratio <= 0:
        return img

    w, h = img.size
    crop_ratio = max(0.1, min(0.95, float(crop_ratio)))  # safe clamp

    new_w = int(w * crop_ratio)
    new_h = int(h * crop_ratio)

    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h

    return img.crop((left, top, right, bottom))


@torch.no_grad()
def predict_one(model, tf, device, img_path, class_names, center_crop: float):
    img = Image.open(img_path).convert("RGB")
    img = apply_center_crop(img, center_crop)

    x = tf(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

    pred_idx = int(torch.tensor(probs).argmax().item())
    pred_name = class_names[pred_idx]
    return pred_name, probs


def smart_decision(pred_name, probs, class_names, gate, good_gate, close_margin):
    """
    gate: general "max probability" gate
    good_gate: stricter requirement to accept class "good"
    close_margin: if good is predicted but broken is close, mark UNKNOWN
    """

    max_conf = max(probs)
    final_pred = pred_name if max_conf >= gate else "UNKNOWN"

    # If class names exist, apply special rules
    if "good" in class_names:
        good_i = class_names.index("good")
        good_conf = probs[good_i]

        # Don't accept "good" unless it's confident
        if final_pred == "good" and good_conf < good_gate:
            final_pred = "UNKNOWN"

        # If broken exists and it's close to good, safer to say UNKNOWN
        if final_pred == "good" and "broken" in class_names:
            broken_i = class_names.index("broken")
            broken_conf = probs[broken_i]

            # If good and broken are too close, reject
            if (good_conf - broken_conf) < close_margin:
                final_pred = "UNKNOWN"

    return final_pred, max_conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="models/defect_best.pt", help="checkpoint path")
    parser.add_argument("--path", default="datasets/coffee_defect_dataset_v2/val", help="image file or folder")
    parser.add_argument("--random", type=int, default=0, help="random N images from folder")
    parser.add_argument("--gate", type=float, default=0.70, help="general confidence gate (max prob)")
    parser.add_argument("--good_gate", type=float, default=0.80, help="minimum confidence to accept 'good'")
    parser.add_argument("--close_margin", type=float, default=0.10, help="if good-broken < margin => UNKNOWN")
    parser.add_argument("--center_crop", type=float, default=0.85,
                        help="center crop ratio to reduce background (0 disables, try 0.80–0.90)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.ckpt):
        print(f"❌ Checkpoint not found: {args.ckpt}")
        return

    model, img_size, class_names = load_model(args.ckpt, device)
    tf = build_transform(img_size, args.center_crop)

    images = collect_images(args.path)
    if not images:
        print(f"❌ No images found: {args.path}")
        return

    if args.random > 0 and os.path.isdir(args.path):
        random.shuffle(images)
        images = images[:args.random]
        print(f"\n🎲 Random testing: {len(images)} images\n")
    else:
        print(f"\n🧪 Testing: {len(images)} image(s)\n")

    print(f"✅ Device: {device}")
    print(f"✅ Model imgsz: {img_size}")
    print(f"✅ Classes: {class_names}")
    print(f"✅ gate={args.gate} | good_gate={args.good_gate} | close_margin={args.close_margin} | center_crop={args.center_crop}\n")

    for img_path in images:
        pred, probs = predict_one(model, tf, device, img_path, class_names, args.center_crop)
        final_pred, max_conf = smart_decision(pred, probs, class_names, args.gate, args.good_gate, args.close_margin)

        print("-" * 60)
        print("Image:", img_path)
        print("Predicted:", final_pred, f"(raw={pred}, max_conf={max_conf:.3f})")
        for i, c in enumerate(class_names):
            print(f"{c}: {probs[i]:.4f}")

    print("\n✅ Done.\n")


if __name__ == "__main__":
    main()
