import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from facenet_pytorch import MTCNN
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]
THRESHOLD  = 0.50
CLASS_TO_IDX = {"Fake": 0, "Real": 1}

print(f"✅ Device : {DEVICE}")

# ── Charger modèle ────────────────────────────────────────────────────
def load_model():
    model = efficientnet_b4(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()
mtcnn = MTCNN(keep_all=False, device=DEVICE)

# ── Transform ─────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# ── Détection visage ──────────────────────────────────────────────────
def detect_face(img: Image.Image) -> Image.Image:
    boxes, _ = mtcnn.detect(img)
    if boxes is not None:
        x1, y1, x2, y2 = [int(v) for v in boxes[0]]
        margin = int(0.1 * max(x2-x1, y2-y1))
        x1 = max(0, x1-margin); y1 = max(0, y1-margin)
        x2 = min(img.width, x2+margin); y2 = min(img.height, y2+margin)
        return img.crop((x1, y1, x2, y2)), True
    return img, False

# ── Grad-CAM ──────────────────────────────────────────────────────────
def get_gradcam(tensor, face_img):
    target_layers  = [model.features[-1]]
    cam            = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam  = cam(
        input_tensor=tensor,
        targets=[ClassifierOutputTarget(CLASS_TO_IDX["Real"])]
    )[0]
    img_np = face_img.resize((224, 224))
    img_np = np.array(img_np).astype(np.float32) / 255.0
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return Image.fromarray(visualization)

# ── PIL → base64 ──────────────────────────────────────────────────────
def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

# ── Routes ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    file     = request.files["image"]
    img      = Image.open(file.stream).convert("RGB")
    face_img, face_detected = detect_face(img)

    # Prédiction TTA
    real_idx = CLASS_TO_IDX["Real"]
    probs    = []
    for _ in range(10):
        tensor = transform(face_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(tensor)
            p   = torch.softmax(out, dim=1)
        probs.append(p[0][real_idx].item())

    real_prob  = float(np.mean(probs))
    fake_prob  = 1 - real_prob
    confidence = abs(real_prob - 0.5) * 2
    prediction = "REAL" if real_prob >= THRESHOLD else "FAKE"

    # Grad-CAM
    tensor_cam = transform(face_img).unsqueeze(0).to(DEVICE)
    gradcam_img = get_gradcam(tensor_cam, face_img)

    return jsonify({
        "prediction"    : prediction,
        "real_prob"     : round(real_prob * 100, 1),
        "fake_prob"     : round(fake_prob * 100, 1),
        "confidence"    : round(confidence * 100, 1),
        "face_detected" : face_detected,
        "face_image"    : pil_to_base64(face_img.resize((300, 300))),
        "gradcam_image" : pil_to_base64(gradcam_img)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)