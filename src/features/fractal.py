# src/features/fractal.py
import torch

def calculate_fractal_dimension(cluster_mask, scales, device):
    scales = torch.tensor(scales, device=device).int()
    cluster_mask = cluster_mask.to(device)
    cluster_mask = (cluster_mask > 0).float()

    box_counts = torch.zeros(len(scales), device=device)
    streams = [torch.cuda.Stream(device=device) for _ in scales]

    def compute_box_count(scale, idx, stream):
        with torch.cuda.stream(stream):
            pooled_mask = torch.nn.functional.avg_pool2d(
                cluster_mask.unsqueeze(0).unsqueeze(0),
                kernel_size=scale.item(),
                stride=scale.item()
            )
            box_counts[idx] = (pooled_mask > 0).sum() + 1e-6

    for i, scale in enumerate(scales):
        compute_box_count(scale, i, streams[i])

    torch.cuda.synchronize()

    valid_mask = box_counts > 0
    valid_box_counts = box_counts[valid_mask]
    valid_scales = scales[valid_mask]

    if len(valid_box_counts) < 2:
        raise ValueError("Not enough valid box counts.")

    log_scales = torch.log(valid_scales.float())[1:]
    log_counts = torch.log(valid_box_counts)[1:]

    A = torch.stack([log_scales, torch.ones_like(log_scales)], dim=1)
    solution = torch.linalg.lstsq(A, log_counts.unsqueeze(1)).solution
    slope = solution[0].item()
    fractal_dimension = abs(slope)
    return fractal_dimension
