# src/features/gabor2D.py
import torch
import numpy as np
import itertools
from params import DEVICE, DTYPE, GB_KER_SIZE

GABOR_PARAMS_2D = {
    "sigma": np.linspace(1, 4, 2),
    "lambda": np.linspace(10, 10, 1),
    "psi": np.linspace(0, np.pi, 4),
    "gamma": np.linspace(0.5, 2.0, 2),
    "theta": np.linspace(0, np.pi, 6),
}

def gabor_filter_2d(sigma, theta, lambd, psi, gamma, size):
    theta = torch.tensor(theta, dtype=DTYPE, device=DEVICE)
    coords = torch.linspace(-size, size, 2 * size + 1, device=DEVICE, dtype=DTYPE)
    y, x = torch.meshgrid(coords, coords, indexing="ij")

    x_prime = x * torch.cos(theta) + y * torch.sin(theta)
    y_prime = -x * torch.sin(theta) + y * torch.cos(theta)

    envelope = torch.exp(-0.5 * ((x_prime**2 / sigma**2) + (y_prime**2 / (sigma/gamma)**2)))
    sinusoid = torch.cos(2 * np.pi * x_prime / lambd + psi)
    return envelope * sinusoid

def generate_gabor_features_2d(image_tensor):
    image_tensor = image_tensor.to(DEVICE)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    param_combinations = list(itertools.product(
        GABOR_PARAMS_2D["sigma"],
        GABOR_PARAMS_2D["lambda"],
        GABOR_PARAMS_2D["psi"],
        GABOR_PARAMS_2D["gamma"],
        GABOR_PARAMS_2D["theta"]
    ))

    gabor_filters = torch.stack([
        gabor_filter_2d(sigma, theta, lambd, psi, gamma, GB_KER_SIZE)
        for sigma, lambd, psi, gamma, theta in param_combinations
    ]).unsqueeze(1)

    filtered_images = torch.nn.functional.conv2d(
        image_tensor, gabor_filters, padding=GB_KER_SIZE
    )
    gabor_features = filtered_images.squeeze(0).permute(1, 2, 0)
    return gabor_features

def compute_gabor_features_per_modality_2d(image_3d):
    image_tensor = image_3d.to(DEVICE)
    num_modalities = image_tensor.shape[-1]
    gabor_features_modalities = []
    feature_labels = []

    for m in range(num_modalities):
        modality_tensor = image_tensor[..., m]
        modality_features = generate_gabor_features_2d(modality_tensor)
        gabor_features_modalities.append(modality_features)
        num_filters = modality_features.shape[-1]
        feature_labels.extend([f"gabor_modality_{m+1}_filter_{i}" for i in range(num_filters)])

    gabor_features = torch.cat([f for f in gabor_features_modalities], dim=-1)
    return gabor_features, feature_labels
