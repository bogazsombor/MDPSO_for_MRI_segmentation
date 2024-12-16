import os
import nibabel as nib
import torch
from params import PREPROCESSED_DATA_DIR

def normalize_image(image):
    """
    Normalizes the MRI image data to the range [0-1].
    The input is a 4-channel MRI image (H x W x 4).
    """
    min_val = image.min(dim=(0, 1), keepdim=True).values
    max_val = image.max(dim=(0, 1), keepdim=True).values
    return (image - min_val) / (max_val - min_val)

def load_and_prepare_slice(code, slice_index):
    """
    Loads and prepares the given slice of the MRI images from the specified code.
    Preparation includes normalizing each modality and combining them.
    For debugging, it prints the number of pixels and the number of non-zero values in each channel.

    :param code: The code for the MRI files to be loaded.
    :param slice_index: The index of the desired slice.
    :return: A torch.Tensor (H x W x 4) where the four modalities are concatenated.
    """
    modalities = ['flair', 't1ce', 't1', 't2']
    image_slices = []

    for modality in modalities:
        # File path
        file_name = f"BraTS2021_{code}_{modality}.nii.gz"
        file_path = os.path.join(PREPROCESSED_DATA_DIR, file_name)

        # Load NIfTI file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found.")

        # Load and select slice
        image = nib.load(file_path).get_fdata()
        if slice_index >= image.shape[2]:
            raise ValueError(f"Slice index ({slice_index}) is outside the dimensions ({image.shape[2]}) of the given image ({modality}).")
        slice_image = image[:, :, slice_index]

        # Normalize to [0, 1]
        min_val = slice_image.min()
        max_val = slice_image.max()
        normalized_slice = (slice_image - min_val) / (max_val - min_val)

        # Convert to tensor
        tensor_slice = torch.tensor(normalized_slice, dtype=torch.float32)
        image_slices.append(tensor_slice)

    # Concatenate all modalities (H x W x 4)
    stacked_slices = torch.stack(image_slices, dim=-1)

    return stacked_slices

def load_seg_slice(code, slice_index):
    """
    Loads and prepares the given segmentation slice from the specified code.
    The preparation includes normalization and combination of each modality.
    For debugging, it prints the number of pixels and the number of non-zero values in each channel.

    :param code: The code for the MRI files to be loaded.
    :param slice_index: The index of the desired slice.
    :return: A torch.Tensor (H x W), the segmentation map.
    """

    modality = 'seg'
    # File path
    file_name = f"BraTS2021_{code}_{modality}.nii.gz"
    file_path = os.path.join(PREPROCESSED_DATA_DIR, file_name)

    # Load NIfTI file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    # Load and select slice
    image = nib.load(file_path).get_fdata()
    if slice_index >= image.shape[2]:
        raise ValueError(f"Slice index ({slice_index}) is outside the dimensions ({image.shape[2]}) of the given image ({modality}).")
    slice_image = image[:, :, slice_index]

    # Convert to tensor (binarize >0)
    return (torch.tensor(slice_image, dtype=torch.float32) > 0).long()


import matplotlib.pyplot as plt
import numpy as np
import math
import torch

def save_image(image, output_path, rand_color=False):
    """
    Saves the given image to a file. If multi-channel, saves all channels in a grid.

    Args:
        image (torch.Tensor): Image tensor in format (H x W or H x W x C).
        output_path (str): File name or path.
        rand_color (bool): If True, generate a colorful image with random colors (only valid for single-channel images).
    """
    if len(image.shape) == 3 and image.shape[-1] > 1:
        if rand_color:
            raise ValueError("rand_color=True is only supported for single-channel images.")

        # Multi-channel image handling
        num_channels = image.shape[-1]
        grid_size = math.ceil(math.sqrt(num_channels))  # Grid size for arrangement

        # Create a figure with subplots
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
        axes = axes.flatten()  # Flatten the 2D array of axes for easy indexing

        for i in range(num_channels):
            channel = image[:, :, i]
            # Normalize and display grayscale
            channel = channel / channel.max() if channel.max() > 0 else channel
            axes[i].imshow(channel.cpu().numpy(), cmap="gray")
            axes[i].axis("off")
            axes[i].set_title(f"Channel {i+1}")

        # Hide unused subplots
        for i in range(num_channels, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    else:
        # Single-channel image handling
        if rand_color:
            # Apply random colors
            unique_values = torch.unique(image)
            color_map = {value.item(): np.random.rand(3) for value in unique_values if value.item() > 0}
            color_map[0] = np.array([0, 0, 0])  # Ensure 0 intensity is black

            # Generate the colorful image
            color_image = np.zeros((image.shape[0], image.shape[1], 3))  # H x W x 3 for RGB
            for value, color in color_map.items():
                color_image[image.cpu().numpy() == value] = color

            plt.figure(figsize=(8, 8))
            plt.imshow(color_image)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
            plt.close()
        else:
            # Grayscale mode
            image = image / image.max() if image.max() > 0 else image  # Normalize to [0-1]
            plt.figure(figsize=(8, 8))
            plt.imshow(image.cpu().numpy(), cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
            plt.close()

    print(f"Image saved to: {output_path}")
