# predict_all.py
import os
import argparse
import torch
import timm
from PIL import Image
from torchvision import transforms


def load_ckpt(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(f"Checkpoint format not supported: {ckpt_path} (expected dict with 'model_state')")
    return ckpt


def build_model_from_ckpt(ckpt, device: str):
    model_name = ckpt.get("model_name", "mobilenetv3_large_100")
    img_size = int(ckpt.get("img_size", 512))
    class_names = ckpt.get("class_names", [])
    if not class_names:
        raise ValueError("class_names not found in checkpoint")

    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, img_size, class_names


def get_transform(img_size: int, center_crop: float):
    # center_crop: 0 disables. e.g., 0.85 crops 85% center area to reduce background.
    def apply_center_crop(img: Image.Image) -> Image.Image:
        if not center_crop or center_crop <= 0:
            return img
        w, h = img.size
        r = max(0.1, min(0.95, float(center_crop)))
        nw, nh = int(w * r), int(h * r)
        left = (w - nw) // 2
        top = (h - nh) // 2
        return img.crop((left, top, left + nw, top + nh))

    tf = transforms.Compose([
        transforms.Lambda(apply_center_crop),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tf


@torch.no_grad()
def predict(model, tf, device: str, img: Image.Image, class_names):
    x = tf(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu()
    pred_idx = int(torch.argmax(probs).item())
    pred_name = class_names[pred_idx]
    conf = float(probs[pred_idx].item())
    return pred_name, conf, probs.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="path to a single image")
    parser.add_argument("--type_ckpt", default="models/bean_type_best.pt", help="bean type checkpoint")
    parser.add_argument("--defect_ckpt", default="models/defect_best.pt", help="defect checkpoint")

    # gates
    parser.add_argument("--type_gate", type=float, default=0.70, help="min confidence to accept type")
    parser.add_argument("--defect_gate", type=float, default=0.70, help="min confidence to accept defect")
    parser.add_argument("--good_gate", type=float, default=0.80, help="min confidence to accept 'good' defect class")
    parser.add_argument("--close_margin", type=float, default=0.10, help="if good-broken < margin => UNKNOWN")

    parser.add_argument("--center_crop", type=float, default=0.85, help="center crop ratio (0 disables)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return
    if not os.path.exists(args.type_ckpt):
        print(f"❌ Type checkpoint not found: {args.type_ckpt}")
        return
    if not os.path.exists(args.defect_ckpt):
        print(f"❌ Defect checkpoint not found: {args.defect_ckpt}")
        return

    # Load image once
    img = Image.open(args.image).convert("RGB")

    # Load models
    type_ckpt = load_ckpt(args.type_ckpt, device)
    defect_ckpt = load_ckpt(args.defect_ckpt, device)

    type_model, type_size, type_classes = build_model_from_ckpt(type_ckpt, device)
    defect_model, defect_size, defect_classes = build_model_from_ckpt(defect_ckpt, device)

    type_tf = get_transform(type_size, args.center_crop)
    defect_tf = get_transform(defect_size, args.center_crop)

    # Predict type
    type_pred, type_conf, type_probs = predict(type_model, type_tf, device, img, type_classes)
    final_type = type_pred if type_conf >= args.type_gate else "UNKNOWN"

    # Predict defect
    defect_pred, defect_conf, defect_probs = predict(defect_model, defect_tf, device, img, defect_classes)
    final_defect = defect_pred if defect_conf >= args.defect_gate else "UNKNOWN"

    # Extra safety: don't accept "good" unless confident
    if final_defect != "UNKNOWN" and "good" in defect_classes:
        good_i = defect_classes.index("good")
        good_conf = defect_probs[good_i]

        if final_defect == "good" and good_conf < args.good_gate:
            final_defect = "UNKNOWN"

        # If good and broken are too close, safer UNKNOWN
        if final_defect == "good" and "broken" in defect_classes:
            broken_i = defect_classes.index("broken")
            broken_conf = defect_probs[broken_i]
            if (good_conf - broken_conf) < args.close_margin:
                final_defect = "UNKNOWN"

    # Print combined result
    print("\n================= ✅ FINAL RESULT =================")
    print("Image:", args.image)
    print(f"Bean Type   : {final_type} (conf={type_conf:.3f}, raw={type_pred})")
    print(f"Bean Quality: {final_defect} (conf={defect_conf:.3f}, raw={defect_pred})")

    print("\n--- Type probabilities ---")
    for i, c in enumerate(type_classes):
        print(f"{c}: {type_probs[i]:.4f}")

    print("\n--- Defect probabilities ---")
    for i, c in enumerate(defect_classes):
        print(f"{c}: {defect_probs[i]:.4f}")

    print("==================================================\n")


if __name__ == "__main__":
    main()
