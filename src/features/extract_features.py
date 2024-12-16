# extract_features.py

import torch
import os
import logging
import matplotlib.pyplot as plt
from params import (
    DEVICE, SLICE_INDEX, OUTPUT_DIR, DISTANCE_WEITGHTS, CONTEXT, SAMPLE_SIZE
)
from features.cluster_features import extract_cluster_features
from features.gabor2D import compute_gabor_features_per_modality_2d
from features.smoothing import gaussian_smoothing_per_modalities
from clustering.MDPSO import MDPSO
from data_loading import load_and_prepare_slice, load_seg_slice


def extract_cluster_features_and_labels(code):
    """
    Extract cluster-based features for a given subject slice using MDPSO clustering.
    Returns flat cluster features and their corresponding feature labels.
    """
    images = load_and_prepare_slice(code, SLICE_INDEX).to(DEVICE)
    segmentation = load_seg_slice(code, SLICE_INDEX).to(DEVICE)

    weights = DISTANCE_WEITGHTS
    bounds = (0.0, 1.0)
    optimizer = MDPSO(images, bounds, weights, DEVICE)
    cluster_image = optimizer.optimize()

    output_file = os.path.join(str(OUTPUT_DIR), f"clusterisation_{code}.png")
    plt.imsave(output_file, cluster_image.cpu().numpy(), cmap="gray")
    print(f"Cluster image saved at {output_file}")

    cluster_features_tensor, _, cluster_labels = extract_cluster_features(
        cluster_image, images.shape[:-1], images, segmentation
    )

    # Flatten cluster features: (H, W, F) -> (H*W, F)
    flat_features = cluster_features_tensor.view(-1, cluster_features_tensor.shape[-1])
    return flat_features, cluster_labels


def extract_features_and_labels(code, sample=False):
    """
    Extract features and corresponding labels for training or testing.
    If CONTEXT is True, also incorporate cluster features and Gaussian-smoothed features.
    
    Args:
        code: Subject code.
        sample (bool): Whether to sample and balance tumor and non-tumor voxels.
    
    Returns:
        (features, labels, feature_labels)
    """
    try:
        logging.info(f"Loading images and segmentation for subject: {code}")
        images = load_and_prepare_slice(code, SLICE_INDEX).to(DEVICE)
        segmentation = load_seg_slice(code, SLICE_INDEX).to(DEVICE)

        # Compute Gabor features: expects (H, W, filters*C) directly
        gabor_feats, gabor_labels = compute_gabor_features_per_modality_2d(images)
        gabor_feats = gabor_feats.to(DEVICE)

        H, W, g_dim = gabor_feats.shape

        original_labels = [f"original_modality_{i}" for i in range(images.shape[-1])]
        feature_labels = original_labels + gabor_labels

        # Flatten images and gabor features
        images_flat = images.view(-1, images.shape[-1])  # (H*W, C)
        gabor_features_flat = gabor_feats.view(-1, g_dim)  # (H*W, filters*modalities)

        if CONTEXT:
            # Integrate cluster and Gaussian-smoothed features
            cluster_features, cluster_fe_labels = extract_cluster_features_and_labels(code)
            gaussian_smoothed, gs_labels = gaussian_smoothing_per_modalities(images, [1, 2, 4, 8])

            # Update feature labels
            feature_labels.extend(gs_labels)
            feature_labels.extend(cluster_fe_labels)

            # Check for dimension consistency
            n_vox = images_flat.size(0)
            if not (n_vox == gabor_features_flat.size(0) == gaussian_smoothed.size(0) == cluster_features.size(0)):
                logging.error(
                    "Mismatch in tensor sizes for concatenation: "
                    f"images_flat: {images_flat.size(0)}, "
                    f"gabor_features_flat: {gabor_features_flat.size(0)}, "
                    f"gaussian_smoothed: {gaussian_smoothed.size(0)}, "
                    f"cluster_features: {cluster_features.size(0)}"
                )
                raise ValueError("Tensor sizes must match for concatenation along dim=-1.")

            # Concatenate all features
            combined_features = torch.cat([images_flat, gabor_features_flat, gaussian_smoothed, cluster_features], dim=-1)
        else:
            # Only original and gabor features
            combined_features = torch.cat([images_flat, gabor_features_flat], dim=-1)

        # Free memory
        del images, images_flat, gabor_feats, gabor_features_flat
        torch.cuda.empty_cache()

        labels = segmentation.flatten()

        if sample:
            # Identify non-zero modalities and tumor/non-tumor masks
            non_zero_mask = torch.any(combined_features[..., :4] > 0, dim=-1)
            tumor_mask = labels > 0
            brain_mask = non_zero_mask & (labels == 0)

            # Sample indices
            brain_indices = torch.nonzero(brain_mask, as_tuple=True)[0]
            tumor_indices = torch.nonzero(tumor_mask, as_tuple=True)[0]

            sample_size = int(SAMPLE_SIZE)
            sampled_tumor_indices = tumor_indices[torch.randperm(len(tumor_indices))[:sample_size]]
            sampled_non_tumor_indices = brain_indices[torch.randperm(len(brain_indices))[:sample_size]]

            # Balance tumor and non-tumor samples
            selected_indices = torch.cat([sampled_tumor_indices, sampled_non_tumor_indices])

            features = combined_features[selected_indices]
            labels = labels[selected_indices]

            del combined_features, brain_indices, tumor_indices, sampled_tumor_indices, sampled_non_tumor_indices
            torch.cuda.empty_cache()
        else:
            features = combined_features

        return features, labels, feature_labels

    except Exception as e:
        logging.error(f"Error in extract_features_and_labels: {e}")
        raise
