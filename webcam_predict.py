import cv2
import torch
import timm
from PIL import Image
from torchvision import transforms
from collections import deque

# =====================
# CONFIG
# =====================
TYPE_CKPT   = "models/bean_type_best.pt"
DEFECT_CKPT = "models/defect_best.pt"

TYPE_GATE   = 0.80     # stricter gate for type
DEFECT_GATE = 0.70
GOOD_GATE   = 0.80
CLOSE_MARGIN = 0.10

CENTER_ROI_RATIO = 0.6     # 60% center crop (IMPORTANT)
RUN_EVERY_N_FRAMES = 8     # increase if laggy
CAM_INDEX = 0              # laptop webcam

device = "cuda" if torch.cuda.is_available() else "cpu"

# buffers for smoothing
type_buffer = deque(maxlen=10)
defect_buffer = deque(maxlen=10)


# =====================
# LOAD MODEL
# =====================
def load_ckpt(path):
    ckpt = torch.load(path, map_location=device)
    model = timm.create_model(
        ckpt["model_name"],
        pretrained=False,
        num_classes=len(ckpt["class_names"])
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["img_size"], ckpt["class_names"]


def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])


@torch.no_grad()
def predict(model, tf, pil_img, class_names):
    x = tf(pil_img).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0].cpu()
    idx = int(torch.argmax(probs))
    return class_names[idx], float(probs[idx]), probs.tolist()


def smart_defect_decision(pred, conf, probs, class_names):
    if conf < DEFECT_GATE:
        return "UNKNOWN"

    if pred == "good":
        good_i = class_names.index("good")
        if probs[good_i] < GOOD_GATE:
            return "UNKNOWN"

        if "broken" in class_names:
            broken_i = class_names.index("broken")
            if (probs[good_i] - probs[broken_i]) < CLOSE_MARGIN:
                return "UNKNOWN"

    return pred


# =====================
# MAIN
# =====================
def main():
    print("🔄 Loading models...")
    type_model, type_size, type_classes = load_ckpt(TYPE_CKPT)
    defect_model, defect_size, defect_classes = load_ckpt(DEFECT_CKPT)

    type_tf = get_transform(type_size)
    defect_tf = get_transform(defect_size)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    frame_count = 0
    last_type = "UNKNOWN"
    last_defect = "UNKNOWN"

    print("✅ Webcam running — place bean inside GREEN BOX")
    print("👉 Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # ===== ROI (CRITICAL FIX) =====
        roi_size = int(min(h, w) * CENTER_ROI_RATIO)
        x1 = (w - roi_size) // 2
        y1 = (h - roi_size) // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size

        roi = frame[y1:y2, x1:x2]

        frame_count += 1

        if frame_count % RUN_EVERY_N_FRAMES == 0:
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # ---- TYPE ----
            t_pred, t_conf, t_probs = predict(
                type_model, type_tf, pil_img, type_classes
            )

            final_type = t_pred if t_conf >= TYPE_GATE else "UNKNOWN"
            type_buffer.append(final_type)
            last_type = max(set(type_buffer), key=type_buffer.count)

            # ---- DEFECT ----
            d_pred, d_conf, d_probs = predict(
                defect_model, defect_tf, pil_img, defect_classes
            )

            final_defect = smart_defect_decision(
                d_pred, d_conf, d_probs, defect_classes
            )
            defect_buffer.append(final_defect)
            last_defect = max(set(defect_buffer), key=defect_buffer.count)

        # ===== DRAW UI =====
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame,
                    f"Type: {last_type}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.putText(frame,
                    f"Quality: {last_defect}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.imshow("Ceyloncoffee | Live IoT Camera Test", frame)

        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
