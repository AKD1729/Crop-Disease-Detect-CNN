# System Architecture & Pipeline Workflow Diagrams

This directory contains the structural and procedural diagrams for the **Crop Disease Detection Using CNN** system.

---

## 1. Convolutional Neural Network Architecture (`ScratchCropCNN`)

```mermaid
flowchart TD
    subgraph InputStage["1. Input Stage"]
        IN["Input Image<br/>224 × 224 × 3 RGB"]
    end

    subgraph Block1["2. Convolutional Block 1"]
        C1["Conv2d(3 → 32, 3×3, stride 1, pad 1)"] --> BN1["BatchNorm2d(32)"]
        BN1 --> R1["ReLU"]
        R1 --> MP1["MaxPool2d(2×2, stride 2)<br/>Output: 112 × 112 × 32"]
    end

    subgraph Block2["3. Convolutional Block 2"]
        C2["Conv2d(32 → 64, 3×3, stride 1, pad 1)"] --> BN2["BatchNorm2d(64)"]
        BN2 --> R2["ReLU"]
        R2 --> MP2["MaxPool2d(2×2, stride 2)<br/>Output: 56 × 56 × 64"]
    end

    subgraph Block3["4. Convolutional Block 3"]
        C3["Conv2d(64 → 128, 3×3, stride 1, pad 1)"] --> BN3["BatchNorm2d(128)"]
        BN3 --> R3["ReLU"]
        R3 --> MP3["MaxPool2d(2×2, stride 2)<br/>Output: 28 × 28 × 128"]
    end

    subgraph Block4["5. Convolutional Block 4"]
        C4["Conv2d(128 → 256, 3×3, stride 1, pad 1)"] --> BN4["BatchNorm2d(256)"]
        BN4 --> R4["ReLU"]
        R4 --> MP4["MaxPool2d(2×2, stride 2)<br/>Output: 14 × 14 × 256"]
    end

    subgraph ClassifierHead["6. Classification Head"]
        GAP["AdaptiveAvgPool2d((1, 1))<br/>Output: 1 × 1 × 256"]
        FLAT["Flatten → 256-d Feature Vector"]
        DO["Dropout(p=0.5)"]
        FC["Linear(256 → 10)<br/>Raw Unnormalized Logits"]
        GAP --> FLAT --> DO --> FC
    end

    subgraph InferencePipeline["7. Inference Post-Processing & Gate"]
        SM["Softmax Activation<br/>p = softmax(logits)"]
        OOD{"OOD / Uncertainty Rejection<br/>• Foliage Ratio ≥ 15%<br/>• Confidence ≥ 60%<br/>• Normalized Entropy ≤ 0.55"}
        PRED["Accepted Prediction<br/>Top-1 Disease Class & Confidence"]
        FALLBACK["Uncertainty / Non-Foliar Fallback<br/>Warning / Re-upload Prompt"]
        FC -.-> SM
        SM --> OOD
        OOD -- "Pass" --> PRED
        OOD -- "Fail" --> FALLBACK
    end

    IN --> C1
    MP1 --> C2
    MP2 --> C3
    MP3 --> C4
    MP4 --> GAP
```

---

## 2. End-to-End Pipeline Workflow

```mermaid
flowchart LR
    subgraph DataSources["1. Data Sources (~7,681 Images)"]
        D1["PlantVillage (Corn)<br/>4 classes, 3,852 images<br/>(Lab background)"]
        D2["AgML Kaggle (Rice)<br/>6 classes, 3,829 images<br/>(Semi-field background)"]
    end

    subgraph PreprocessingEDA["2. Preprocessing & Deduplication"]
        PHASH["Perceptual Hash Deduplication<br/>(pHash distance ≤ 5)<br/>78 near-duplicate clusters"]
        EDA["Exploratory Data Analysis<br/>• Class distributions<br/>• Aspect ratios & resolutions"]
    end

    subgraph Splitting["3. Group-Aware Stratified Partition"]
        SPLIT["Stratified 70 / 15 / 15 Split<br/>(Clusters kept in same split)"]
        TR_DF["Train Split<br/>5,373 images"]
        VAL_DF["Val Split<br/>1,146 images"]
        TE_DF["Test Split<br/>1,162 images"]
    end

    subgraph NormAug["4. Transforms & Augmentation"]
        STATS["Dataset Normalization Stats<br/>Computed strictly from Train split:<br/>Mean: [0.4093, 0.5035, 0.3134]<br/>Std:  [0.2068, 0.2053, 0.2021]"]
        TR_AUG["Train Augmentation (Geometric only):<br/>• Resize(224, 224)<br/>• RandomHorizontalFlip(p=0.5)<br/>• RandomRotation(±20°)<br/>• RandomResizedCrop(0.8 - 1.0)"]
        EVAL_TRANS["Eval / Test Transforms:<br/>• Resize(224, 224)<br/>• ToTensor()<br/>• Normalize(Mean, Std)"]
    end

    subgraph ModelTrain["5. CNN Model Training"]
        MODEL["ScratchCropCNN<br/>4 Conv Blocks (32→64→128→256)<br/>GAP + Dropout(0.5) + Linear(10)"]
        OPT["Optimizer & Scheduler:<br/>• Adam (LR = 1e-3)<br/>• ReduceLROnPlateau (factor 0.5, patience 3)<br/>• Class-Weighted CrossEntropyLoss<br/>• Early Stopping (patience 8)"]
    end

    subgraph Evaluation["6. Held-Out Evaluation"]
        TEST_METRICS["Evaluation Metrics on Test Partition:<br/>• Overall Accuracy: 94.0%<br/>• Macro-F1 Score: 0.9321<br/>• Explainable Confusion Matrix (4 techniques)"]
    end

    subgraph Deployment["7. Deployment & Inference Gate"]
        APP["Streamlit Web Application<br/>(Project/app.py)"]
        OOD_GATE["OOD & Uncertainty Rejection Gate:<br/>1. Foliar Chromatic HSV Filter (≥15%)<br/>2. Softmax Confidence Threshold (≥60%)<br/>3. Normalized Entropy Filter (≤0.55)"]
    end

    D1 & D2 --> PHASH --> EDA --> SPLIT
    SPLIT --> TR_DF & VAL_DF & TE_DF
    TR_DF --> STATS --> TR_AUG
    VAL_DF & TE_DF --> EVAL_TRANS
    TR_AUG --> MODEL
    OPT --> MODEL
    MODEL --> EVAL_TRANS --> TEST_METRICS
    MODEL --> APP --> OOD_GATE
```
