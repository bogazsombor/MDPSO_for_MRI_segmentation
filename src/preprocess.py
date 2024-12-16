import os
import nibabel as nib
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from params import (
    RAW_DATA_DIR,
    PREPROCESSED_DATA_DIR,
    MRI_TYPES,
    MRI_TYPES_TO_SCALE,
    HIST_UPPER,
    HIST_FOR_QUANTILE_75,
    HIST_FOR_QUANTILE_25,
    HIST_LOWER,
    MAX_IMAGES,
    DEVICE
)

def get_codes_from_directory(directory, max_images=MAX_IMAGES):
    codes = []
    i = 0
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):
            last_5_chars = folder_name[-5:]
            codes.append(last_5_chars)
            i += 1
            if i >= max_images:
                break
    return codes

def compute_quantiles_tensor(values, q25=0.25, q75=0.75):
    """
    Compute 25th and 75th percentiles using torch.quantile on GPU.
    values: 1D torch.Tensor of voxel intensities (non-zero).
    """
    q_values = torch.quantile(values, torch.tensor([q25, q75], device=values.device))
    return q_values[0].item(), q_values[1].item()

def scale_intensity(volume_tensor, hist_lower=HIST_LOWER, hist_upper=HIST_UPPER,
                    hist_for_q75=HIST_FOR_QUANTILE_75, hist_for_q25=HIST_FOR_QUANTILE_25):
    """
    Scale and clamp the intensity values of the given volume tensor based on quantiles.
    volume_tensor: torch.Tensor on GPU (H x W x D).
    """
    # Extract nonzero values for quantile computation
    nonzero_values = volume_tensor[volume_tensor != 0]
    if nonzero_values.numel() == 0:
        # No scaling if everything is zero
        return volume_tensor

    q25, q75 = compute_quantiles_tensor(nonzero_values, 0.25, 0.75)

    if q75 != q25:
        multiply = (hist_for_q75 - hist_for_q25) / (q75 - q25)
        shift = (hist_for_q25 * q75 - hist_for_q75 * q25) / (q75 - q25)

        # Apply scaling on non-zero voxels
        volume_tensor[volume_tensor != 0] = volume_tensor[volume_tensor != 0] * multiply + shift

        # Clamp values
        volume_tensor.clamp_(min=hist_lower, max=hist_upper)

    return volume_tensor

def preprocess_subject(subject_code):
    os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)
    subject_folder = os.path.join(RAW_DATA_DIR, f"BraTS2021_{subject_code}")

    # Load all modalities into GPU to determine scaling parameters
    # We'll handle intensity scaling per modality as needed
    modality_volumes = {}

    for modality in MRI_TYPES:
        input_path = os.path.join(subject_folder, f"BraTS2021_{subject_code}{modality}.nii.gz")
        nifti_file = nib.load(input_path)
        loaded = nifti_file.get_fdata().astype(np.float32)

        # Move to GPU as a tensor
        volume_tensor = torch.tensor(loaded, dtype=torch.float32, device=DEVICE)

        # Scale intensities if needed
        if modality in MRI_TYPES_TO_SCALE:
            volume_tensor = scale_intensity(volume_tensor)

        # Move back to CPU for saving
        volume_cpu = volume_tensor.cpu().numpy()

        processed_img = nib.Nifti1Image(volume_cpu, affine=nifti_file.affine)
        processed_img.header.set_data_dtype(np.float32)
        processed_img.header.set_xyzt_units('mm', 'sec')

        save_path = os.path.join(PREPROCESSED_DATA_DIR, f"BraTS2021_{subject_code}{modality}.nii.gz")
        nib.save(processed_img, save_path)
        print(f"Saved: {save_path}")

def main():
    codes = get_codes_from_directory(RAW_DATA_DIR)
    for c_index, c in enumerate(codes):
        print(f"Processing subject {c} ({c_index+1}/{len(codes)})...")
        preprocess_subject(c)

if __name__ == "__main__":
    main()
