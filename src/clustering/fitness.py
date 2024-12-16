# src/utils/fitness.py
import torch
from params import DEVICE, DTYPE, DISTANCE_WEITGHTS

scale_euclidian = 4.3

def stable_exp_metric(x, y, weights, just_color=False, scale_euclidian=scale_euclidian):
    if just_color:
        weights = weights.clone()
        weights[0] = weights[1] = 0

    batch_size, feature_dim = x.shape
    num_centroids = y.shape[0]

    x_expanded = x.unsqueeze(1).expand(batch_size, num_centroids, feature_dim)
    y_expanded = y.unsqueeze(0).expand(batch_size, num_centroids, feature_dim)

    differences = x_expanded - y_expanded
    differences[:, :, 0] *= scale_euclidian
    differences[:, :, 1] *= scale_euclidian

    distances = torch.sqrt(torch.sum(weights * torch.exp(torch.clamp(differences**2, max=50)), dim=2))
    return distances

def fitness_function(centroids, flat_image, weights, metric_func=stable_exp_metric):
    num_clusters = centroids.size(0)
    centroids = centroids.view(-1, flat_image.shape[1])
    sqrt_num_clusters = torch.pow(torch.tensor(float(num_clusters), dtype=torch.float32, device=DEVICE), 1/2)

    all_feature_distances = metric_func(flat_image, centroids, weights, just_color=False)
    _, cluster_indices = torch.min(all_feature_distances, dim=1)

    color_distances = metric_func(flat_image, centroids, weights, just_color=False)
    assigned_distances = color_distances[torch.arange(flat_image.size(0)), cluster_indices]

    F_sum = 0.0
    for i in range(num_clusters):
        cluster_mask = (cluster_indices == i)
        A_i = cluster_mask.sum()
        A_i_sqrt = torch.sqrt(A_i.float())
        if A_i_sqrt.item() > 0:
            e_i = assigned_distances[cluster_mask].sum()
            F_sum += (e_i**2 / A_i_sqrt)

    F = sqrt_num_clusters * F_sum
    image_size = float(flat_image.size(0))
    F /= image_size
    return F, cluster_indices

def construct_clustering(flat_image, image_shape, cluster_indices, device=DEVICE):
    x_coords = (flat_image[:, 0] * image_shape[0]).long().clamp(0, image_shape[0]-1).to(device)
    y_coords = (flat_image[:, 1] * image_shape[1]).long().clamp(0, image_shape[1]-1).to(device)
    cluster_map = torch.zeros(image_shape, dtype=torch.long, device=device)
    cluster_map[y_coords, x_coords] = (cluster_indices + 1)
    return cluster_map
