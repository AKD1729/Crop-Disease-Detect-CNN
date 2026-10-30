# Module 9: Technical Project Report
## Crop Disease Detection Using a Custom CNN Architecture

**Author:** Machine Learning Engineering Team  
**Date:** October 2026  
**Repository:** [AKD1729/Crop-Disease-Detection-CNN](https://github.com/AKD1729/Crop-Disease-Detection-CNN)  
**Primary Deliverable:** Coursework & Technical Implementation  

---

## 1. Problem Statement

### 1.1 Objective & Context
Agricultural crop diseases pose substantial threats to global food security, rural livelihoods, and supply chain resilience. Early, accurate foliar lesion detection enables timely, targeted intervention—minimizing yield losses and preventing the overapplication of chemical agrochemicals. 

This project develops an end-to-end deep learning system for automated disease diagnosis focusing on two of the world's primary staple cereal crops:
1. **Corn (Maize, *Zea mays*)**
2. **Rice (*Oryza sativa*)**

### 1.2 Task Formulation & Scope
- **Machine Learning Formulation:** Single-label, multi-class image classification.
- **Target Space ($K=10$ classes):**
  - **Corn (4 classes):** Cercospora Leaf Spot / Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy.
  - **Rice (6 classes):** Bacterial Leaf Blight, Brown Spot, Leaf Blast, Leaf Scald, Sheath Blight, Healthy.
- **Architectural Scope:** A single unified model (`ScratchCropCNN`) covering all 10 classes across both crops simultaneously. The model is built entirely from first principles (scratch CNN) without pre-trained transfer learning backbones (e.g., ResNet, EfficientNet), providing an uncompromised evaluation of CNN inductive biases, feature hierarchies, and optimization mechanics on agricultural patterns.
- **Explicitly Out of Scope:** Multi-label classification (co-occurring infections), disease severity estimation (% lesion area), and cross-domain zero-shot field domain adaptation.
- **Success Criteria:**
  - Macro-averaged F1 score ($\text{Macro-F1} \ge 0.90$) across all 10 classes on an isolated, leakage-proof test split.
  - Robust confusion matrix explainability across intra-crop and inter-crop prediction boundaries.
  - Interactive deployment via a responsive web application featuring real-time out-of-distribution (OOD) and uncertainty rejection.

---

## 2. Data Acquisition & Distribution

### 2.1 Data Sources & Provenance
To compile a diverse, representative dataset while avoiding pre-augmented near-duplicate leakage, data was acquired from two primary repositories:

1. **Corn (Maize):** Sourced from the raw PlantVillage dataset ([spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)). The pre-augmented versions commonly hosted on public competition platforms were deliberately avoided to prevent identical leaf lesions from appearing across training, validation, and test partitions. The corn dataset contains images captured under controlled laboratory lighting with uniform backgrounds.
2. **Rice:** Sourced from the Kaggle AgML rice disease dataset ([anshulm257/rice-disease-dataset](https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset)). Images reflect semi-field settings featuring variable ambient sunlight, natural field canopies, and diverse crop angles.

### 2.2 Class Inventory & Distribution
The consolidated dataset contains **7,681 total images** distributed as follows:

| Class Name | Botanical Scope | Environmental Condition | Image Count | % of Dataset |
|---|---|---|---|---|
| `Corn_Cercospora_Gray_Leaf_Spot` | Corn | Lab / Neutral Background | 513 | 6.68% |
| `Corn_Common_Rust` | Corn | Lab / Neutral Background | 1,192 | 15.52% |
| `Corn_Northern_Leaf_Blight` | Corn | Lab / Neutral Background | 985 | 12.82% |
| `Corn_Healthy` | Corn | Lab / Neutral Background | 1,162 | 15.13% |
| `Rice_Bacterial_Leaf_Blight` | Rice | Semi-field | 636 | 8.28% |
| `Rice_Brown_Spot` | Rice | Semi-field | 646 | 8.41% |
| `Rice_Leaf_Blast` | Rice | Semi-field | 634 | 8.25% |
| `Rice_Leaf_Scald` | Rice | Semi-field | 628 | 8.18% |
| `Rice_Sheath_Blight` | Rice | Semi-field | 632 | 8.23% |
| `Rice_Healthy` | Rice | Semi-field | 653 | 8.50% |
| **Total** | **Combined** | — | **7,681** | **100.0%** |

```
                       Class Distribution (Post-Deduplication)
Corn_Cercospora_Gray_Leaf_Spot:  513 ███████
              Corn_Common_Rust: 1192 ████████████████
     Corn_Northern_Leaf_Blight:  985 █████████████
                  Corn_Healthy: 1162 ███████████████
    Rice_Bacterial_Leaf_Blight:  636 ████████
               Rice_Brown_Spot:  646 ████████
               Rice_Leaf_Blast:  634 ████████
               Rice_Leaf_Scald:  628 ████████
            Rice_Sheath_Blight:  632 ████████
                  Rice_Healthy:  653 █████████
```
*(Reference visual: [eda_class_distribution.png](file:///D:/03_Projects/HCL_Project/artifacts/figures/eda_class_distribution.png))*

### 2.3 Class Imbalance Analysis
Rice classes are remarkably balanced (~628–653 images/class). Corn classes exhibit a mild natural imbalance (ranging from 513 images in Cercospora to 1,192 in Common Rust), presenting a maximum imbalance ratio of **~2.32:1**. Because this imbalance is moderate, neither synthetic oversampling (SMOTE) nor undersampling was employed. Instead, imbalance is handled directly via class-weighted Cross-Entropy loss during model training.

---

## 3. Methodology

```mermaid
flowchart LR
    A["Raw Data (7,681 imgs)"] --> B["pHash Deduplication"]
    B --> C["Group-Aware 70/15/15 Split"]
    C --> D["Dataset Stats Normalization"]
    D --> E["Geometric-Only Augmentation"]
    E --> F["ScratchCropCNN Architecture"]
    F --> G["Adam + ReduceLROnPlateau"]
    G --> H["Macro-F1 & CM Analysis"]
    H --> I["Streamlit App + OOD Rejection"]
```
*(Reference visual: [pipeline_workflow.mmd](file:///D:/03_Projects/HCL_Project/artifacts/diagrams/pipeline_workflow.mmd))*

### 3.1 Deduplication & Leakage Prevention (Module 3a)
Near-duplicate images (e.g., serial burst captures of the exact same leaf from slightly altered distances) present a severe data leakage hazard if randomly distributed between train and test sets. 
- **Method:** Perceptual hashing (`imagehash.phash`) was computed on all 7,681 images.
- **Clustering:** A pairwise Hamming distance threshold of $d \le 5$ was applied within each class.
- **Result:** Identified **78 near-duplicate clusters** containing 2 or more visually identical specimens across 7,579 unique perceptual groups.

### 3.2 Group-Aware Stratified Partitioning (Module 3c)
To guarantee strict statistical independence, a custom group-aware allocator assigned all members of any given cluster strictly to a single partition:
- **Training Set (70%):** 5,373 images
- **Validation Set (15%):** 1,146 images
- **Test Set (15%):** 1,162 images
- **Leakage Check:** An automated assertion confirmed zero group overlap between training, validation, and test sets:
  $$\text{Overlap}(\text{Train}, \text{Val}, \text{Test}) = \emptyset$$

### 3.3 Dataset-Specific Normalization (Module 3d)
Because the network is trained from scratch rather than initialized from ImageNet weights, borrowing ImageNet normalization statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$) is mathematically unjustified. Instead, channel-wise mean and variance were calculated over the training partition:
$$\mu = [0.4093, 0.5035, 0.3134], \quad \sigma = [0.2068, 0.2053, 0.2021]$$
These values accurately reflect the elevated green channel intensity ($\mu_G = 0.5035$) characteristic of agricultural foliar canopies.

### 3.4 Data Augmentation Strategy (Module 4)
Augmentations were applied **strictly to the training split**; validation and test splits were kept unaltered.
- **Policy:** Strictly geometric augmentations.
  - Horizontal Flip ($p=0.5$)
  - Random Rotation ($\pm 20^\circ$)
  - Random Resized Crop (`scale=(0.8, 1.0)`, size $224 \times 224$): selects 80% to 100% of the image area (~1.12× zoom, ~15–20% edge crop).
- **Exclusion of Photometric Jitter:** Color jitter, brightness, saturation, and contrast shifts were explicitly excluded. Foliar disease diagnostics rely heavily on specific necrotic discoloration (e.g., chlorotic yellow halos, grey spots, copper rust pustules); artificial hue distortion risks corrupting critical discriminative features.

### 3.5 Model Architecture: `ScratchCropCNN` (Module 5)
The model consists of 4 convolutional feature extraction blocks followed by Global Average Pooling and a single regularized dense classification head:

```
[Input: 224 × 224 × 3 RGB]
       │
       ▼
[ConvBlock 1: Conv2d(3→32, 3×3, pad 1) → BatchNorm → ReLU → MaxPool(2×2)]  ==> 112 × 112 × 32
       │
       ▼
[ConvBlock 2: Conv2d(32→64, 3×3, pad 1) → BatchNorm → ReLU → MaxPool(2×2)] ==> 56 × 56 × 64
       │
       ▼
[ConvBlock 3: Conv2d(64→128, 3×3, pad 1) → BatchNorm → ReLU → MaxPool(2×2)] ==> 28 × 28 × 128
       │
       ▼
[ConvBlock 4: Conv2d(128→256, 3×3, pad 1) → BatchNorm → ReLU → MaxPool(2×2)] ==> 14 × 14 × 256
       │
       ▼
[AdaptiveAvgPool2d((1, 1))]  ==> 1 × 1 × 256
       │
       ▼
[Flatten]  ==> 256-dimensional vector
       │
       ▼
[Dropout(p = 0.5)]
       │
       ▼
[Linear(256 → 10)]  ==> 10 Unnormalized Class Logits
```
*(Reference visual: [model_architecture.mmd](file:///D:/03_Projects/HCL_Project/artifacts/diagrams/model_architecture.mmd))*

**Design Rationale:**
- **Spatial Reduction:** 4 pooling stages reduce spatial dimensions from $224 \to 112 \to 56 \to 28 \to 14$, sufficient to capture fine-grained spot and pustule textures without spatial over-compression.
- **Global Average Pooling (GAP):** Eliminates massive fully-connected parameter matrices, drastically reducing total trainable parameters to **~1.58 MB** and conferring translational invariance.
- **Logit Output:** Returns raw unnormalized logits for compatibility with `nn.CrossEntropyLoss` (which incorporates `LogSoftmax` numerically). Softmax is computed during inference.

### 3.6 Training Routine & Reproducibility (Module 6)
- **Loss Function:** Cross-Entropy with inverse class weighting:
  $$w_c = \frac{N_{\text{total}}}{K \cdot N_c}$$
- **Optimizer:** Adam ($\text{lr} = 10^{-3}$, $\beta_1 = 0.9, \beta_2 = 0.999$).
- **Learning Rate Schedule:** `ReduceLROnPlateau` (metric = validation loss, factor = 0.5, patience = 3).
- **Batch Size:** 32 (yielding ~168 gradient updates per epoch on 5,373 training samples).
- **Early Stopping:** Monitored on validation loss with patience = 8.
- **Reproducibility Guarantee:** Deterministic seeds fixed to `42` across Python, NumPy, PyTorch, CUDA, and cuDNN (`deterministic = True`, `benchmark = False`).

---

## 4. Experimental Results & Analysis

### 4.1 Training Dynamics
Training proceeded across 50 epochs on an NVIDIA GPU. The learning rate automatically decayed from $1.0 \times 10^{-3}$ at Epoch 1 down to $1.25 \times 10^{-4}$ at Epoch 50 as validation loss converged:

- **Epoch 01:** Train Loss: `1.1776`, Train Acc: `62.48%` | Val Loss: `0.8093`, Val Acc: `68.76%`
- **Epoch 15:** Train Loss: `0.5699`, Train Acc: `82.45%` | Val Loss: `0.4534`, Val Acc: `82.55%`
- **Epoch 30:** Train Loss: `0.3884`, Train Acc: `87.16%` | Val Loss: `0.3199`, Val Acc: `87.00%`
- **Epoch 50:** Train Loss: `0.2414`, Train Acc: `92.65%` | Val Loss: `0.2022`, Val Acc: `93.28%`

```
   Training & Validation Loss                              Training & Validation Accuracy
1.2 ┼──*                                                1.0 ┼                             *---* (Val Acc: 93.3%)
    │   \                                                   │                        /---/
0.8 ┼    *---\                                          0.8 ┼               /-------/      *---* (Train Acc: 92.7%)
    │         \---*                                         │          /---/
0.4 ┼              \---*                                0.6 ┼    *----/
    │                   *---* (Val Loss: 0.2022)            │   /
0.0 ┼─────────────────────────                          0.4 ┼──*
    0        25       50 (Epochs)                           0        25       50 (Epochs)
```
*(Reference visual: [training_curves.png](file:///D:/03_Projects/HCL_Project/artifacts/figures/training_curves.png))*

### 4.2 Test Set Quantitative Performance (Module 7)
Evaluated on the completely unseen held-out test split (**1,162 images**):
- **Overall Test Accuracy:** **94.0%**
- **Macro-Averaged Precision:** **0.93**
- **Macro-Averaged Recall:** **0.93**
- **Macro-Averaged F1 Score:** **0.9321**

#### Comprehensive Classification Report:
| Target Disease Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Corn Cercospora Gray Leaf Spot** | 0.88 | 0.90 | 0.89 | 78 |
| **Corn Common Rust** | 1.00 | 0.99 | 1.00 | 180 |
| **Corn Northern Leaf Blight** | 0.94 | 0.93 | 0.94 | 149 |
| **Corn Healthy** | 1.00 | 1.00 | 1.00 | 175 |
| **Rice Bacterial Leaf Blight** | 0.91 | 0.93 | 0.92 | 96 |
| **Rice Brown Spot** | 0.93 | 0.81 | 0.86 | 98 |
| **Rice Leaf Blast** | 0.84 | 0.92 | 0.88 | 96 |
| **Rice Leaf Scald** | 0.94 | 0.94 | 0.94 | 95 |
| **Rice Sheath Blight** | 0.99 | 0.92 | 0.95 | 96 |
| **Rice Healthy** | 0.92 | 1.00 | 0.96 | 99 |
| **Overall Summary** | **Macro: 0.93** | **Macro: 0.93** | **Macro: 0.9321** | **Total: 1,162** |

### 4.3 Explainable Confusion Matrix Analysis
Four explainability techniques were incorporated into the confusion matrix:
1. **Row Normalization:** Displays true conditional recall percentages ($P(\hat{y} | y)$).
2. **Crop-Grouped Ordering:** Classes 0–3 form a contiguous Corn block; Classes 4–9 form a contiguous Rice block.
3. **Heatmap Styling:** Highlighting off-diagonal confusion densities.
4. **Automated Confusion Flagging:** Flagging any error where $\ge 10\%$ of true instances are misclassified.

*(Reference visual: [confusion_matrix.png](file:///D:/03_Projects/HCL_Project/artifacts/figures/confusion_matrix.png))*

**Key Diagnostic Insights:**
- **Zero Inter-Crop Cross-Talk:** There is virtually 0% confusion between Corn and Rice classes. The network cleanly separates the two crop species before resolving specific disease symptoms.
- **Intra-Crop Confusion (Cercospora $\to$ Northern Leaf Blight):** **10.3%** of true Corn Cercospora Gray Leaf Spot samples were predicted as Northern Leaf Blight. This is biologically expected: both pathologies present as elongated, necrotic, grayish-tan foliar lesions running parallel to leaf veins.
- **Rice Brown Spot Recall (81%):** Rice Brown Spot exhibited minor confusion with Leaf Blast (small oval lesions with dark margins), accounting for its 0.86 F1 score.

---

## 5. Deployment & Out-of-Distribution (OOD) Rejection

### 5.1 Interactive Web Application (Module 8)
An interactive application was deployed using **Streamlit** ([`Project/app.py`](file:///D:/03_Projects/HCL_Project/Project/app.py)), featuring:
- Upload support for JPEG, PNG, and JFIF leaf photographs.
- Strict training-time transform parity (`Resize(224, 224)` $\to$ `ToTensor()` $\to$ `Normalize(mean, std)`).
- Instantaneous visual rendering and confidence scoring.

### 5.2 Out-of-Distribution Vulnerability & Multi-Tier Resolution
Standard closed-set softmax neural networks inherently assign 100% of probability mass across known classes. When presented with arbitrary non-plant images (e.g., human faces, household objects, synthetic textures), standard models frequently output >95% confidence for an arbitrary class (often "Rice Healthy" due to broad background feature representation).

To resolve this critical vulnerability, an inference rejection module ([`Project/inference.py`](file:///D:/03_Projects/HCL_Project/Project/inference.py)) was integrated:

```
[Uploaded Image]
       │
       ▼
[Stage 1: Chromatic Foliar Pre-Filter] ──(Foliage Ratio < 15%)──► [🛑 REJECT: Non-Plant / OOD Image]
       │
       │ (Foliage Ratio ≥ 15%)
       ▼
[CNN Model Forward Pass & Softmax]
       │
       ▼
[Stage 2: Confidence & Entropy Gates] ──(Conf < 60% OR Entropy > 0.55)──► [⚠️ REJECT: Uncertainty Fallback]
       │
       │ (Conf ≥ 60% AND Entropy ≤ 0.55)
       ▼
[✅ ACCEPT: Diagnosis & Confidence Rendered]
```

#### Empirical Validation of OOD Gate:
- **Synthetic Non-Foliar Inputs (Grayscale, Blue Sky, Red Car, Skin Tone):** Achieved **0.0% foliage ratio** and were rejected immediately at Stage 1 before misleading disease diagnosis could occur.
- **Legitimate In-Distribution Leaf Samples:** All 10 test samples in [`sample/`](file:///D:/03_Projects/HCL_Project/sample/) passed Stage 1 (foliage ratios: 37.1% to 97.6%) and were accepted with average confidence > 93%.

---

## 6. Limitations & Honest Disclosures

1. **Lab vs. Semi-Field Backdrop Mismatch:** Corn images were captured under plain lab backdrops (PlantVillage), whereas rice images feature semi-field environments. The model may utilize subtle background contrast cues as shortcut features to distinguish corn from rice.
2. **Geographic and Cultivar Narrowness:** Datasets predominantly reflect specific regional cultivars; performance on heirloom varieties or under drought/salinity stress remains unverified.
3. **Asymptomatic & Early-Stage Detection:** Training images predominantly display well-developed, visible lesions; very early asymptomatic incubation phases cannot be reliably detected.
4. **Binary Absence/Presence:** The model outputs class categorical probabilities, not quantitative disease severity indexes or percentage coverage metrics.
5. **Closed-Set Assumption Boundary:** While the foliar chromatic filter catches non-plant imagery, non-corn/non-rice agricultural crops (e.g., wheat, potato, soybean) will still be mapped to the closest of the 10 known classes.

---

## 7. Future Work

1. **Explainable Spatial Saliency (Grad-CAM):** Integrate Gradient-weighted Class Activation Mapping into the Streamlit UI to highlight the exact leaf pixels driving each classification.
2. **Expansion to Wheat & Sorghum:** Extend the unified model to additional major cereal grains to form a comprehensive Poaceae grain diagnostic suite.
3. **Edge Deployment via ONNX & TFLite:** Quantize the PyTorch model to INT8 precision and export to TFLite for offline mobile inference in rural farming areas without internet access.
4. **Deep Feature-Space OOD Detection:** Implement Mahalanobis distance rejection in the 256-dimensional GAP feature space to reject non-target plant species (e.g., tomato, grapevine) automatically.

---

## 8. Artifacts & Verification Reference

- **Model Checkpoint:** [`Project/crop_disease_scratch_cnn.pth`](file:///D:/03_Projects/HCL_Project/Project/crop_disease_scratch_cnn.pth) (1.58 MB)
- **Training Pipeline Script:** [`crop_disease_pipeline.py`](file:///D:/03_Projects/HCL_Project/crop_disease_pipeline.py)
- **Experimental Notebook:** [`hcl-project.ipynb`](file:///D:/03_Projects/HCL_Project/hcl-project.ipynb)
- **Inference & OOD Module:** [`Project/inference.py`](file:///D:/03_Projects/HCL_Project/Project/inference.py)
- **Deployment Application:** [`Project/app.py`](file:///D:/03_Projects/HCL_Project/Project/app.py)
- **Visual Artifacts:** [`artifacts/figures/`](file:///D:/03_Projects/HCL_Project/artifacts/figures/)
- **Architecture Diagrams:** [`artifacts/diagrams/`](file:///D:/03_Projects/HCL_Project/artifacts/diagrams/)
