import ast
import os
import torch
import torch.nn as nn
import streamlit as st
from torchvision import transforms
from PIL import Image

try:
    from inference import predict_with_rejection
except ImportError:
    from Project.inference import predict_with_rejection

CLASSES = [
    'Corn Cercospora Gray Leaf Spot', 'Corn Common Rust', 'Corn Northern Leaf Blight', 'Corn Healthy',
    'Rice Bacterial Leaf Blight', 'Rice Brown Spot', 'Rice Leaf Blast', 'Rice Leaf Scald',
    'Rice Sheath Blight', 'Rice Healthy'
]


class ScratchCropCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ScratchCropCNN, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block4 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


@st.cache_resource
def load_model():
    model = ScratchCropCNN(num_classes=10)
    checkpoint_path = "crop_disease_scratch_cnn.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(os.path.dirname(__file__), "crop_disease_scratch_cnn.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    return model


@st.cache_resource
def load_normalization_stats():
    """
    Loads dataset-specific RGB mean/std computed in Module 3d.
    """
    stats_path = "normalization_stats.txt"
    if not os.path.exists(stats_path):
        stats_path = os.path.join(os.path.dirname(__file__), "normalization_stats.txt")
    with open(stats_path, "r") as f:
        lines = f.readlines()
    mean = ast.literal_eval(lines[0].split("=", 1)[1].strip())
    std = ast.literal_eval(lines[1].split("=", 1)[1].strip())
    return mean, std


model = load_model()
DATASET_MEAN, DATASET_STD = load_normalization_stats()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
])

st.title("🌾 Crop Disease Detection Classifier")
st.caption("Custom CNN trained from scratch on Corn (Maize) and Rice foliar diseases (10 classes).")

uploaded_file = st.file_uploader("Upload a Maize/Corn or Rice Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    result = predict_with_rejection(
        image=image,
        model=model,
        transform=transform,
        classes=CLASSES,
        conf_threshold=60.0,
        max_entropy_ratio=0.55,
        min_foliage_ratio=0.15,
    )

    if result['status'] == 'accepted':
        st.success(f"**Prediction:** {result['predicted_class']}")
        st.info(f"**Confidence Score:** {result['confidence_pct']:.2f}% | **Normalized Entropy:** {result['normalized_entropy']:.2f}")
    elif result['status'] == 'rejected_non_foliar':
        st.error(f"🛑 {result['message']}")
    else:  # rejected_uncertain
        st.warning(f"⚠️ {result['message']}")
