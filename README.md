# DSA4266 — AI Image Detection

Deep learning project for detecting AI-generated (fake) images using modern vision architectures.

This repository trains and evaluates neural network models to classify images as:

- Real (0)
- Fake (1)

Dataset: DeepDetect-2025  
Task: Binary image classification (AI-generated vs real)

---

# Project Overview

AI-generated images from models like StyleGAN3, DALL·E 3, Midjourney, and Stable Diffusion 3 are becoming increasingly photorealistic.

This project builds deep learning classifiers capable of distinguishing:

- Real photographs  
- AI-generated synthetic images  

We focus on:

- Strong generalization across diverse domains
- Proper validation-based threshold selection
- Balanced evaluation metrics (not just raw accuracy)

# Setup Guide (For New Users)

## 1. Clone Repository

git clone <repo-url>  
cd DSA4266

---

## 2. Create Virtual Environment

Windows:

python -m venv .venv  
.\.venv\Scripts\activate  

Mac / Linux:

python3 -m venv .venv  
source .venv/bin/activate  

---

## 3. Install Dependencies

pip install --upgrade pip  
pip install -r requirements.txt  

---

## 4. Download & Setup Dataset

Run:

python scripts/data_shortcut.py  

This script will:

- Create required folders:
  - data/
  - outputs/
  - models/swin/v1/checkpoints/
- Download DeepDetect-2025 from KaggleHub
- Copy dataset into:

data/deepdetect-2025/ddata/



# Models Implemented

## 1. Swin Transformer (Primary Model)

Architecture:
- swin_tiny_patch4_window7_224
- Pretrained on ImageNet
- Fine-tuned for binary classification
- Single-logit output using BCEWithLogitsLoss

Why Swin?

Swin Transformer:
- Captures local texture artifacts
- Models global contextual inconsistencies
- Uses hierarchical window-based attention
- Strong for subtle AI artifact detection

---

### Training Pipeline

1. Load dataset from data/train  
2. Split into:
   - 90% training
   - 10% validation  
3. Train using:
   - BCEWithLogitsLoss  
   - AdamW optimizer  
4. Select best threshold using validation balanced accuracy  
5. Save best checkpoint  

---

### Train Swin

python -m models.swin.v1.train  

---

### Test Swin

python -m models.swin.v1.test  

Outputs include:

- Accuracy  
- Balanced Accuracy  
- ROC-AUC  
- Confusion Matrix  
- Precision / Recall / F1-score  

---

# Example Performance (Swin v1)

Example results:

- Accuracy: ~0.87  
- Balanced Accuracy: ~0.86  
- ROC-AUC: ~0.97  

Example confusion matrix:

[[11319    58]  
 [ 2775  7624]]

Interpretation:

- Very low false positives (real misclassified as fake)  
- Good fake detection with room for threshold tuning  

---

## 2. EfficientNet Model (TO UPDATE)

⚠ EfficientNet implementation is pending integration.

Planned configuration:

- Architecture: efficientnet_b0  
- Pretrained ImageNet backbone  
- Replace final classifier with single-logit output  
- Same train/validation/test pipeline as Swin  

Expected strengths:

- Efficient CNN baseline  
- Strong texture modeling  
- Lightweight and fast  

TODO:

- Add EfficientNet model implementation  
- Add training script  
- Benchmark against Swin  

---

# Evaluation Metrics

We report:

- Accuracy  
- Balanced Accuracy  
- ROC-AUC  
- Confusion Matrix  
- Precision  
- Recall  
- F1-score  

Balanced Accuracy is emphasized because:

- Prevents majority-class collapse  
- Measures performance on both real and fake classes equally  

---

# Experimental Design Decisions

- Single-logit output instead of 2-class softmax  
- Validation-based threshold tuning  
- No aggressive class weighting (dataset near-balanced)  
- Pretrained ImageNet initialization  

---

# What Is NOT Committed

To keep the repository clean, we do NOT commit:

- .venv/  
- data/  
- outputs/  
- checkpoints/  

To reproduce results:

1. Clone repo  
2. Create venv  
3. Install requirements  
4. Run dataset setup script  
5. Train model  

---

# Dataset Description

DeepDetect-2025 includes:

- 100,000+ images  
- Real: ~60,000  
- Fake: ~55,000  
- Multiple generators:
  - StyleGAN3  
  - DALL·E 3  
  - Midjourney  
  - Stable Diffusion 3  
- Diverse domains:
  - Portraits  
  - Nature  
  - Urban environments  
  - Artworks  
  - Synthetic objects  

This ensures strong generalization testing.

---

# Future Improvements

- Add EfficientNet baseline  
- Cross-generator evaluation  
- Per-category analysis  
- Domain generalization experiments  
- Model ensembling (Swin + EfficientNet)  
- Robustness testing against adversarial edits  

---

# Contributors

- Jing Zhi Ng
- Sze Yui xxx
- Alexendra
- Dillon?
- Joshua

---

Course: DSA4266  
Project: AI Image Detection  
Objective: Build robust models to detect AI-generated visual content.