# modules/cluster_features.py
import torch
from params import DEVICE, MODALITIES
from features.fractal import calculate_fractal_dimension

def extract_cluster_features(cluster_image, image_shape, modalities, segmentation):
    """
    Extract cluster-based features for each voxel, including fractal dimension and various shape-based and intensity-based features.
    Args:
        cluster_image (torch.Tensor): Precomputed cluster assignments (H x W).
        image_shape (tuple): Shape of the original image, e.g. (H, W).
        modalities (torch.Tensor): Input modalities tensor (H, W, C).
        segmentation (torch.Tensor): Segmentation labels (H, W).
    Returns:
        cluster_features (torch.Tensor): A tensor containing cluster features for each voxel.
        cluster_procent_labels (torch.Tensor): A tensor of labels (or derived values) per voxel.
        cluster_labels (list): A list of feature label names.
    """
    num_modalities = modalities.shape[-1]
    per_modality_features = 4  # mean, stddev, contrast, homogeneity
    global_features = 14  # cluster-level features like fractal dimension, etc.
    num_features = num_modalities * per_modality_features + global_features

    cluster_features = torch.zeros((*image_shape, num_features), device=DEVICE)
    cluster_procent_labels = torch.zeros(image_shape, device=DEVICE)

    # Labels for features
    cluster_labels = []
    for modality in MODALITIES:
        cluster_labels.extend([
            f"mean_intensity_{modality}",
            f"stddev_intensity_{modality}",
            f"contrast_{modality}",
            f"homogeneity_{modality}"
        ])
    cluster_labels.extend([
        "cluster_size", "bbox_size_x", "bbox_size_y", 
        "num_neighbors", "num_zero_neighbors", "stddev_x", "stddev_y",
        "density", "neighbor_ratio", "norm_mean_intensity",
        "aspect_ratio", "perimeter_to_area", "excentricity", "fractal_dimension"
    ])

    # Define box sizes for fractal dimension calculation
    scales = [1, 2, 4, 8, 16]

    unique_clusters = torch.unique(cluster_image)
    for cluster_id in unique_clusters:
        if cluster_id == 0:
            continue  # Skip background

        cluster_mask = (cluster_image == cluster_id)
        coords = torch.nonzero(cluster_mask, as_tuple=False).float()
        cluster_size = len(coords)

        # Bounding box
        bbox_min = coords.min(dim=0).values
        bbox_max = coords.max(dim=0).values
        bbox_size_x, bbox_size_y = (bbox_max - bbox_min).tolist()

        # Fractal dimension
        fractal_dimension = calculate_fractal_dimension(cluster_mask.float(), scales, device=DEVICE)

        # Neighbors
        padded_image = torch.nn.functional.pad(cluster_image, (1, 1, 1, 1), mode='constant', value=0)
        shifted_neighbors = torch.stack([
            padded_image[1:-1, :-2], padded_image[1:-1, 2:],
            padded_image[:-2, 1:-1], padded_image[2:, 1:-1],
            padded_image[:-2, :-2], padded_image[:-2, 2:],
            padded_image[2:, :-2], padded_image[2:, 2:]
        ], dim=0)

        foreign_neighbor_mask = (shifted_neighbors != cluster_id) & (shifted_neighbors != 0)
        num_neighbors = foreign_neighbor_mask[:, cluster_mask].sum().item()

        zero_neighbor_mask = (shifted_neighbors == 0)
        num_zero_neighbors = zero_neighbor_mask[:, cluster_mask].sum().item()

        # Density
        density = cluster_size / (bbox_size_x * bbox_size_y + 1e-6)

        # Aspect ratio
        aspect_ratio = bbox_size_y / (bbox_size_x + 1e-6)

        # Perimeter-to-area ratio (approximation)
        perimeter_to_area = cluster_size / (bbox_size_x + bbox_size_y + 1e-6)

        # Excentricity
        stddev_x = coords[:, 0].std().item()
        stddev_y = coords[:, 1].std().item()
        excentricity = stddev_y / (stddev_x + 1e-6)

        # Modality-based features
        modality_features = []
        for i in range(num_modalities):
            modality_mask = modalities[..., i][cluster_mask]
            mean_intensity = modality_mask.mean()
            stddev_intensity = modality_mask.std()
            contrast = modality_mask.max() - modality_mask.min()
            homogeneity = 1 / (1 + stddev_intensity)

            modality_features.extend([mean_intensity.item(), stddev_intensity.item(), contrast.item(), homogeneity.item()])

        # Normalized mean intensity
        norm_mean_intensity = sum(modality_features[0::4]) / len(modality_features[0::4]) / (density + 1e-6)

        # Combine all features
        cluster_values = torch.tensor(modality_features + [
            cluster_size, bbox_size_x, bbox_size_y, num_neighbors, num_zero_neighbors,
            stddev_x, stddev_y, density, num_neighbors / cluster_size,
            norm_mean_intensity, aspect_ratio, perimeter_to_area, excentricity, fractal_dimension
        ], device=DEVICE)

        cluster_features[cluster_mask] = cluster_values.unsqueeze(0).expand(cluster_mask.sum(), -1)
        cluster_procent_labels[cluster_mask] = segmentation[cluster_mask].float().mean()

    return cluster_features, cluster_procent_labels, cluster_labels
