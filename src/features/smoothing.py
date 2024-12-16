# src/features/smoothing.py
import torch
import torch.nn.functional as F
from params import DEVICE, DTYPE

def gaussian_smoothing(image, sigma):
    sigma_tensor = torch.tensor(sigma, device=image.device, dtype=image.dtype)
    kernel_size = int(2 * torch.ceil(2 * sigma_tensor).item() + 1)
    x = torch.arange(-kernel_size//2+1, kernel_size//2+1, device=image.device, dtype=image.dtype)
    gaussian_kernel = torch.exp(-(x**2)/(2*sigma_tensor**2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    gaussian_kernel = gaussian_kernel.view(1, 1, -1)

    smoothed = F.conv2d(image.unsqueeze(0).unsqueeze(0), gaussian_kernel.unsqueeze(1), padding="same")
    smoothed = F.conv2d(smoothed, gaussian_kernel.unsqueeze(2), padding="same")
    return smoothed.squeeze()

def gaussian_smoothing_per_modalities(image_3d, sigmas):
    image_tensor = image_3d.to(DEVICE)
    num_modalities = image_tensor.shape[-1]
    smoothed_features_modalities = []
    feature_labels = []

    for m in range(num_modalities):
        modality_tensor = image_tensor[..., m]
        for s_idx, sigma in enumerate(sigmas):
            smoothed_modality = gaussian_smoothing(modality_tensor, sigma)
            smoothed_features_modalities.append(smoothed_modality.unsqueeze(-1))
            feature_labels.append(f"smoothed_modality_{m+1}_sigma_{s_idx+1}")

    smoothed_features = torch.cat(smoothed_features_modalities, dim=-1)
    flattened = smoothed_features.view(-1, smoothed_features.shape[-1])
    return flattened, feature_labels
