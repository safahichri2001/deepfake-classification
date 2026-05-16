<div align="center">

# Deepfake Detection System

**Deep learning pipeline for detecting AI-generated facial images**

[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-91.08%25-brightgreen?style=for-the-badge)](.)
[![AUC](https://img.shields.io/badge/ROC--AUC-0.97-blue?style=for-the-badge)](.)
[![Python](https://img.shields.io/badge/Python-3.11-yellow?style=for-the-badge&logo=python)](.)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?style=for-the-badge&logo=pytorch)](.)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask)](.)

</div>

---

## Results

<div align="center">

| Metric | Value |
|:---|:---:|
| Test Accuracy | **91.08 %** |
| ROC-AUC | **≈ 0.97** |
| Best Validation Accuracy | **92.52 %** |
| Dataset Size | **14 823 images** |
| Training Set (after cleaning) | **10 662 images** |
| Model | **EfficientNetB4** |

</div>

---

## Demo

The web interface accepts any image and returns:
- **REAL / FAKE** prediction with confidence score
- **Grad-CAM heatmap** highlighting the manipulated regions
- Automatic **face detection** and cropping via MTCNN

<div align="center">

```
 Upload Image  →  Face Detection (MTCNN)  →  EfficientNetB4  →  REAL / FAKE
                                                                      ↓
                                                              Grad-CAM Heatmap
```

</div>

---

## Pipeline

```
Raw Dataset  (14 823 images — Fake / Real)
│
├── Stage 1  ── Audit            class distribution, visual inspection
├── Stage 2  ── Cleaning         perceptual hash duplicates, corruption check
│                                → 99.1 % images conserved, 133 duplicates removed
├── Stage 3  ── Augmentation     flip, rotation, JPEG compression, Gaussian blur
├── Stage 4  ── DataLoaders      WeightedRandomSampler for class balance
├── Stage 5  ── Model            EfficientNetB4 + deep classifier head
├── Stage 6  ── Training         FocalLoss · AdamW · OneCycleLR · early stopping
├── Stage 7  ── Evaluation       confusion matrix · ROC curve · AUC
├── Stage 8  ── Grad-CAM         visual explanation of model decisions
├── Stage 9  ── Inference TTA    15-pass test-time augmentation
└── Stage 10 ── Deployment       TorchScript export (.pt) + Flask web app
```

---

## Model Architecture

| Component | Details |
|:---|:---|
| Backbone | EfficientNetB4 (ImageNet pretrained) |
| Fine-tuned blocks | Last 4 feature blocks + classifier |
| Classifier head | `Linear(1792→512) → BN → SiLU → Dropout → Linear(512→2)` |
| Loss | Focal Loss (γ=2) + label smoothing (0.05) |
| Optimizer | AdamW + OneCycleLR |
| Regularization | Gradient clipping · Dropout (0.5/0.3) · Early stopping (patience=8) |
| Inference | 15-pass Test-Time Augmentation (TTA) |

---

## Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/safahichri2001/deepfake-classification.git
cd deepfake-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the web app
python app.py
# → open http://localhost:5000
```

---

## Project Structure

```
deepfake-classification/
├── app.py                            Flask web application
├── notebooks/
│   └── deepfake_detection.ipynb     Full training pipeline (10 stages)
├── templates/
│   └── index.html                   Web interface
├── requirements.txt
└── README.md
```

---

## Technologies

`Python 3.11` · `PyTorch 2.5` · `EfficientNetB4` · `facenet-pytorch` · `pytorch-grad-cam` · `Flask` · `scikit-learn` · `imagehash` · `NumPy` · `Matplotlib`

---

<div align="center">

## Authors

**Safa Hichri** · **Aya Saied**

*Artificial Intelligence & Cybersecurity*

---

*If you find this project useful, leave a star!*

</div>
