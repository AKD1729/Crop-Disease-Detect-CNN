# Crop_Disease_Detection
# 🌾 Crop Disease Detection Using CNN (From Scratch)

An end-to-end deep learning system for automated crop disease classification across two major grain staples—**Corn (Maize)** and **Rice**—using a custom Convolutional Neural Network built from first principles (no transfer learning).

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview

This project implements a unified 10-class CNN classifier trained on **~7,700 images** to identify healthy plants and critical foliar diseases. The CNN architecture is deliberately built and trained from scratch rather than leveraging pre-trained weights (Transfer Learning) to explore neural representation learning and inductive biases directly.

### Targeted Classes (10 Total)

| Crop | Disease / Condition | Source Dataset | Image Count |
|---|---|---|---|
| **Corn (Maize)** | Cercospora / Gray Leaf Spot | Raw PlantVillage | 513 |
| **Corn (Maize)** | Common Rust | Raw PlantVillage | 1,192 |
| **Corn (Maize)** | Northern Leaf Blight | Raw PlantVillage | 985 |
| **Corn (Maize)** | Healthy | Raw PlantVillage | 1,162 |
| **Rice** | Bacterial Leaf Blight | Kaggle Rice Leaf AUG | 636 |
| **Rice** | Brown Spot | Kaggle Rice Leaf AUG | 646 |
| **Rice** | Leaf Blast | Kaggle Rice Leaf AUG | 634 |
| **Rice** | Leaf Scald | Kaggle Rice Leaf AUG | 628 |
| **Rice** | Sheath Blight | Kaggle Rice Leaf AUG | 632 |
| **Rice** | Healthy Rice Leaf | Kaggle Rice Leaf AUG | 653 |

---

## 🏗️ Architecture & Methodology

### 1. CNN Model Design (`ScratchCropCNN`)
- **Input:** `224 × 224 × 3` RGB
- **Conv Blocks (4):** Filter progression `32 → 64 → 128 → 256`
  - Each block: `Conv2d (3×3, stride 1, pad 1)` → `BatchNorm2d` → `ReLU` → `MaxPool2d (2×2, stride 2)`
- **Spatial Reduction:** `224 × 224` → `112 × 112` → `56 × 56` → `28 × 28` → `14 × 14`
- **Classifier Head:** `AdaptiveAvgPool2d (GAP)` → `Dropout (p=0.5)` → `Linear (256, 10)`

### 2. Leakage-Proof Preprocessing & Split
- **Perceptual Deduplication:** Image hashing (`imagehash.phash`, distance threshold = 5) clusters near-duplicate images within each class.
- **Group-Aware Stratified Split:** 70% Train, 15% Validation, 15% Test. All members of near-duplicate clusters are kept strictly in the same split to avoid data leakage.
- **Dataset-Specific Normalization:** RGB channel mean and standard deviation are calculated strictly from the training partition:
  - `Mean: [0.4093, 0.5035, 0.3134]`
  - `Std:  [0.2068, 0.2053, 0.2021]`
- **Augmentation (Train split only):** Random Horizontal Flip ($p=0.5$), Random Rotation ($\pm 20^\circ$), Random Resized Crop (scale $0.8 - 1.0$). Photometric distortion is excluded to preserve disease color signals.

### 3. Optimization & Training Routine
- **Loss:** Class-weighted `CrossEntropyLoss` (mitigates mild 2.4:1 class imbalance).
- **Optimizer:** `Adam` (Initial Learning Rate = `1e-3`, $\beta_1=0.9, \beta_2=0.999$).
- **LR Scheduler:** `ReduceLROnPlateau` (factor = 0.5, patience = 3).
- **Early Stopping:** Monitored on validation loss (patience = 8).

---

## 📊 Experimental Results

On the held-out test partition (**1,162 unseen images**):

- **Overall Test Accuracy:** `94.0%`
- **Macro-F1 Score:** `0.9321`

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Corn Cercospora Gray Leaf Spot | 0.88 | 0.90 | 0.89 | 78 |
| Corn Common Rust | 1.00 | 0.99 | 1.00 | 180 |
| Corn Northern Leaf Blight | 0.94 | 0.93 | 0.94 | 149 |
| Corn Healthy | 1.00 | 1.00 | 1.00 | 175 |
| Rice Bacterial Leaf Blight | 0.91 | 0.93 | 0.92 | 96 |
| Rice Brown Spot | 0.93 | 0.81 | 0.86 | 98 |
| Rice Leaf Blast | 0.84 | 0.92 | 0.88 | 96 |
| Rice Leaf Scald | 0.94 | 0.94 | 0.94 | 95 |
| Rice Sheath Blight | 0.99 | 0.92 | 0.95 | 96 |
| Rice Healthy | 0.92 | 1.00 | 0.96 | 99 |

---

## 🚀 Getting Started

### 1. Installation

Clone this repository and install dependencies:

```bash
git clone --recurse-submodules https://github.com/AKD1729/Crop-Disease-Detection-CNN.git
cd Crop-Disease-Detection-CNN

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application

To run the interactive diagnosis interface:

```bash
streamlit run Project/app.py
```

Upload any leaf image from `sample/` to test live inference.

### 3. Re-running the Full Training Pipeline

To execute the data preparation, deduplication, training, and evaluation pipeline from scratch:

```bash
python crop_disease_pipeline.py
```
*(Or run the Jupyter Notebook: `hcl-project.ipynb`)*

---

## 📂 Repository Structure

```text
├── .gitignore                      # Git ignore configuration
├── .gitmodules                     # Submodule definition for Project/
├── requirements.txt                # Python environment dependencies
├── README.md                       # Comprehensive documentation
├── Module_9_Report.md              # Official 7-section coursework technical report
├── plan (1).md                     # Project specification & design document
├── crop_disease_pipeline.py        # Complete modular training pipeline (Modules 1-7)
├── hcl-project.ipynb               # Full experiment notebook with executed outputs
├── artifacts/
│   ├── figures/                    # Exported training curves, confusion matrix, and EDA plots
│   └── diagrams/                   # Architecture and pipeline workflow diagrams (Mermaid)
├── Project/                        # Submodule: Streamlit deployment application
│   ├── app.py                      # Interactive Web UI
│   ├── inference.py                # Multi-tier OOD and uncertainty rejection engine
│   ├── crop_disease_scratch_cnn.pth# Trained CNN checkpoint (~1.58 MB)
│   └── normalization_stats.txt     # RGB dataset normalization values
└── sample/                         # Test images for quick validation
```

---

## 📚 Coursework Report & Visual Artifacts

- **Full Coursework Report:** Read [`Module_9_Report.md`](./Module_9_Report.md) for the complete 7-section technical analysis, methodology, and limitations.
- **Figures:** View exported [training curves, confusion matrix, and EDA distributions](./artifacts/figures/).
- **Architecture & Workflow Diagrams:** View high-resolution [Mermaid diagrams and specifications](./artifacts/diagrams/README.md).

---

## ⚠️ Known Limitations

1. **Condition Mismatch:** Corn images were captured under controlled lab backdrops (PlantVillage), whereas rice images have semi-field backgrounds.
2. **Out-of-Distribution (OOD) Fallback:** Softmax probability thresholds detect inter-class confusion but do not naturally reject arbitrary non-plant imagery. An input validation filter is recommended for field use.
3. **Symptom Severity:** The model classifies categorical presence/absence, not disease progression or percentage severity.
