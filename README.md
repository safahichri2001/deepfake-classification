# Deepfake Detection — EfficientNetB4

AI system for detecting deepfake facial images using fine-tuned EfficientNetB4 with Grad-CAM explainability and a Flask web interface.

---

## Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **91.08%** |
| **ROC-AUC** | **~0.97** |
| Best Val Accuracy | 92.52% |
| Dataset | 14,823 images (Fake / Real) |
| Training images | 10,662 (after cleaning) |

---

## Pipeline

```
Raw Dataset
    └── Stage 1 : Audit          — class distribution, visual inspection
    └── Stage 2 : Cleaning       — duplicate detection (phash), corruption check
    └── Stage 3 : Augmentation   — flip, rotation, JPEG compression, Gaussian blur
    └── Stage 4 : DataLoaders    — WeightedRandomSampler for class balance
    └── Stage 5 : Model          — EfficientNetB4 + deep classifier head
    └── Stage 6 : Training       — FocalLoss, AdamW, OneCycleLR, early stopping
    └── Stage 7 : Evaluation     — confusion matrix, ROC curve, AUC
    └── Stage 8 : Grad-CAM       — visual explanation of predictions
    └── Stage 9 : Inference TTA  — 15-pass test-time augmentation
    └── Stage 10: Deployment     — TorchScript export (.pt)
```

---

## Model Architecture

- **Backbone**: EfficientNetB4 (ImageNet pretrained)
- **Fine-tuned**: last 4 feature blocks + classifier
- **Head**: `Linear(1792→512) → BN → SiLU → Linear(512→2)`
- **Loss**: Focal Loss (γ=2) + label smoothing (0.05)
- **Optimizer**: AdamW + OneCycleLR scheduler
- **Regularization**: gradient clipping, dropout (0.5 / 0.3), early stopping

---

## Web Interface

Flask app with MTCNN face detection and Grad-CAM visualization.

```bash
pip install -r requirements.txt
python app.py
# open http://localhost:5000
```

---

## Project Structure

```
deepfake-classification/
├── app.py                          # Flask web app
├── notebooks/
│   └── deepfake_detection.ipynb   # Full training pipeline
├── templates/
│   └── index.html                 # Web interface
├── requirements.txt
└── README.md
```

---

## Technologies

Python 3.11 · PyTorch 2.5 · EfficientNetB4 · facenet-pytorch (MTCNN) · pytorch-grad-cam · Flask · scikit-learn · imagehash

---

## Author

Safa Hichri — AI & Cybersecurity Student
