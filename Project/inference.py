"""
Inference and Out-of-Distribution (OOD) Rejection Module for Crop Disease Detection.

Provides:
  1. Foliar chromatic pre-filtering (HSV-based leaf presence check)
  2. Prediction with multi-tier uncertainty and OOD rejection:
     - Stage 1: Foliar presence verification (filters non-plant images like faces, objects)
     - Stage 2: Softmax confidence thresholding (Module 8: >= 60.0%)
     - Stage 3: Normalized Shannon entropy filtering (catches dispersed probability distributions)
"""

import math
import numpy as np
import torch
from PIL import Image


def check_foliar_presence(pil_img: Image.Image, min_foliage_ratio: float = 0.15) -> tuple[bool, float]:
    """
    Evaluates whether the image contains a sufficient fraction of plant foliage
    (green, yellow, or chlorotic/necrotic foliar hues in HSV color space).

    Returns:
        (is_foliar, foliage_ratio)
    """
    hsv_img = pil_img.convert('HSV')
    arr = np.array(hsv_img)
    h, s, v = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # In PIL HSV: H is in [0, 255] (corresponds to 0-360 degrees).
    # Foliar range (yellows, greens, chlorotic olive): ~18 to 118 (approx 25° to 165°)
    # Require minimal saturation (>= 30) and value (>= 30) to eliminate shadows/pure white/grays.
    foliar_mask = (h >= 18) & (h <= 118) & (s >= 30) & (v >= 30)
    ratio = float(np.mean(foliar_mask))
    return (ratio >= min_foliage_ratio), ratio


def predict_with_rejection(
    image: Image.Image,
    model: torch.nn.Module,
    transform,
    classes: list[str],
    conf_threshold: float = 60.0,
    max_entropy_ratio: float = 0.55,
    min_foliage_ratio: float = 0.15,
) -> dict:
    """
    Executes inference with defensible OOD and uncertainty rejection gates.

    Returns a structured dictionary:
        {
            'status': 'accepted' | 'rejected_non_foliar' | 'rejected_uncertain',
            'predicted_class': str | None,
            'predicted_idx': int | None,
            'confidence_pct': float,
            'normalized_entropy': float,
            'foliage_ratio': float,
            'probabilities': dict[str, float],
            'message': str
        }
    """
    # Stage 1: Chromatic foliar check
    is_foliar, foliage_ratio = check_foliar_presence(image, min_foliage_ratio=min_foliage_ratio)
    if not is_foliar:
        return {
            'status': 'rejected_non_foliar',
            'predicted_class': None,
            'predicted_idx': None,
            'confidence_pct': 0.0,
            'normalized_entropy': 1.0,
            'foliage_ratio': foliage_ratio,
            'probabilities': {},
            'message': (
                f"Out-of-Distribution Rejection: Insufficient foliar characteristics detected "
                f"({foliage_ratio * 100:.1f}% foliage pixels, minimum required is {min_foliage_ratio * 100:.0f}%). "
                f"Please ensure the image contains a clear maize or rice crop leaf."
            ),
        }

    # Model forward pass
    input_tensor = transform(image).unsqueeze(0)
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)

    confidence_pct = conf.item() * 100.0
    num_classes = len(classes)
    max_possible_entropy = math.log(num_classes)
    entropy = -float(torch.sum(probs * torch.log(probs + 1e-9)).item())
    normalized_entropy = float(entropy / max_possible_entropy)

    prob_dict = {classes[i]: float(probs[i].item() * 100.0) for i in range(num_classes)}

    # Stage 2 & 3: Confidence threshold and Entropy verification
    if confidence_pct < conf_threshold or normalized_entropy > max_entropy_ratio:
        return {
            'status': 'rejected_uncertain',
            'predicted_class': None,
            'predicted_idx': pred_idx.item(),
            'confidence_pct': confidence_pct,
            'normalized_entropy': normalized_entropy,
            'foliage_ratio': foliage_ratio,
            'probabilities': prob_dict,
            'message': (
                f"Uncertainty Fallback: Model prediction confidence is {confidence_pct:.2f}% "
                f"(threshold: {conf_threshold:.0f}%) with normalized entropy of {normalized_entropy:.2f}. "
                f"Sample is ambiguous or out-of-domain. Please provide a clearer leaf photograph."
            ),
        }

    return {
        'status': 'accepted',
        'predicted_class': classes[pred_idx.item()],
        'predicted_idx': pred_idx.item(),
        'confidence_pct': confidence_pct,
        'normalized_entropy': normalized_entropy,
        'foliage_ratio': foliage_ratio,
        'probabilities': prob_dict,
        'message': f"Classified as {classes[pred_idx.item()]} with {confidence_pct:.2f}% confidence.",
    }
