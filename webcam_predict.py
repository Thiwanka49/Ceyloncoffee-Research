import os
import cv2
import torch
import timm
from torchvision import transforms
from PIL import Image


# ============================================================
# MODEL PATHS
# ============================================================
TYPE_CKPT   = "models/bean_type_best.pt"
DEFECT_CKPT = "models/defect_best.pt"

# MUST match ImageFolder class order used in training
TYPE_CLASSES   = ["arabica", "robusta"]
DEFECT_CLASSES = ["broken", "good", "severe_defect"]

IMG_SIZE = 512


# ============================================================
# ROI + GATING SETTINGS
# ============================================================
ROI_SCALE = 0.40                 # smaller square
NO_BEAN_VAR_THRESH = 35.0        # lower = stricter "no bean"

TYPE_CONF_THRESH   = 0.75
DEFECT_CONF_THRESH = 0.70
MARGIN_THRESH      = 0.15


# ============================================================
# UTILS
# ============================================================
def ensure_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")

def center_square_roi(frame, scale=0.40):
    h, w = frame.shape[:2]
    side = int(min(h, w) * scale)
    cx, cy = w // 2, h // 2
    x1 = max(cx - side // 2, 0)
    y1 = max(cy - side // 2, 0)
    x2 = min(x1 + side, w)
    y2 = min(y1 + side, h)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

def no_bean_filter(roi_bgr, thresh):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var < thresh, var

def load_model(ckpt_path, num_classes, device):
    ensure_exists(ckpt_path)
    model = timm.create_model(
        "mobilenetv3_large_100",
        pretrained=False,
        num_classes=num_classes
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


TFM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@torch.no_grad()
def predict_probs(model, roi_bgr, device):
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = TFM(pil).unsqueeze(0).to(device)
    logits = model(x)
    return torch.softmax(logits, dim=1).squeeze(0)

def gate_unknown(probs, conf_thresh, margin_thresh):
    top2 = torch.topk(probs, 2)
    p1 = float(top2.values[0])
    p2 = float(top2.values[1])
    idx = int(top2.indices[0])
    margin = p1 - p2
    unknown = (p1 < conf_thresh) or (margin < margin_thresh)
    return unknown, idx, p1, margin


# ============================================================
# MAIN
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("✅ Device:", device)

    type_model   = load_model(TYPE_CKPT, len(TYPE_CLASSES), device)
    defect_model = load_model(DEFECT_CKPT, len(DEFECT_CLASSES), device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open webcam")

    print("🎥 Webcam running (Type + Defect). Press Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        roi, (x1, y1, x2, y2) = center_square_roi(frame, ROI_SCALE)
        is_no_bean, var = no_bean_filter(roi, NO_BEAN_VAR_THRESH)

        if is_no_bean:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
            cv2.putText(frame, f"NO BEAN DETECTED (var={var:.1f})",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0,0,255), 2)
        else:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"ROI var={var:.1f}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255,255,255), 2)

            # ---------- Bean Type ----------
            t_probs = predict_probs(type_model, roi, device)
            t_unk, t_idx, t_conf, t_margin = gate_unknown(
                t_probs, TYPE_CONF_THRESH, MARGIN_THRESH
            )
            if t_unk:
                t_text = f"Type: UNKNOWN (p={t_conf:.2f})"
                t_col = (0,255,255)
            else:
                t_text = f"Type: {TYPE_CLASSES[t_idx]} (p={t_conf:.2f})"
                t_col = (0,255,0)
            cv2.putText(frame, t_text, (10,65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, t_col, 2)

            # ---------- Defect ----------
            d_probs = predict_probs(defect_model, roi, device)
            d_unk, d_idx, d_conf, d_margin = gate_unknown(
                d_probs, DEFECT_CONF_THRESH, MARGIN_THRESH
            )
            if d_unk:
                d_text = f"Defect: UNKNOWN (p={d_conf:.2f})"
                d_col = (0,255,255)
            else:
                d_text = f"Defect: {DEFECT_CLASSES[d_idx]} (p={d_conf:.2f})"
                d_col = (0,255,0)
            cv2.putText(frame, d_text, (10,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, d_col, 2)

        cv2.imshow("CeylonCoffee – Bean Type + Defect", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
