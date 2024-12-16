# modeling/test.py

import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from features.extract_features import extract_features_and_labels
from params import OUTPUT_DIR, SLICE_INDEX
from data_loading import load_seg_slice
import matplotlib.pyplot as plt
import logging

def dice_score(y_true, y_pred):
    """
    Calculate the Dice coefficient.
    """
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)
def test_model_on_subjects(rf_model, test_codes, output_dir=OUTPUT_DIR):
    """
    Test a trained Random Forest model on multiple subjects, save predictions as PNG files, 
    and compute metrics for each subject with filtering.
    
    Args:
        rf_model: Trained Random Forest model.
        test_codes: List of subject codes for testing.
        output_dir: Directory to save prediction results.
    """
    os.makedirs(output_dir, exist_ok=True)
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    metrics_data = []

    for code in test_codes:
        logging.info(f"Testing subject {code}...")
        try:
            # Extract features and labels
            logging.info(f"Extracting features and labels for subject {code}.")
            features, labels, _ = extract_features_and_labels(code, sample=False)

            # Identify and filter out voxels where first 4 features are zero
            non_zero_mask = (features[:, :4] != 0).any(dim=1)
            features_filtered = features[non_zero_mask]
            labels_filtered = labels[non_zero_mask]

            # Predict using the trained model
            logging.info(f"Making predictions for subject {code}.")
            predictions = torch.tensor(rf_model.predict(features_filtered.cpu().numpy()), device=features.device)

            # Reconstruct prediction image
            logging.info(f"Reconstructing prediction image for subject {code}.")
            segmentation_shape = load_seg_slice(code, SLICE_INDEX).shape
            prediction_image = torch.zeros(segmentation_shape, dtype=torch.uint8, device=features.device)
            prediction_image.view(-1)[non_zero_mask] = predictions.to(torch.uint8)

            # Save prediction as PNG
            output_path = os.path.join(predictions_dir, f"prediction_{code}.png")
            plt.imsave(output_path, prediction_image.cpu().numpy(), cmap="gray")
            logging.info(f"Saved prediction for subject {code} at {output_path}.")

            # Compute metrics
            logging.info(f"Computing metrics for subject {code}.")
            labels_flat = labels_filtered.cpu()
            predictions_flat = predictions.cpu()

            if len(torch.unique(labels_flat)) == 1:
                if torch.all(labels_flat == 0):
                    # No positive pixels in ground truth
                    tn = len(labels_flat)
                    tp = fp = fn = 0
                else:
                    # No negative pixels in ground truth
                    tp = len(labels_flat)
                    tn = fp = fn = 0
            else:
                tn, fp, fn, tp = confusion_matrix(labels_flat.numpy(), predictions_flat.numpy(), labels=[0, 1]).ravel()

            total_positive = tp + fn
            total_negative = tn + fp

            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn)>0 else 1.0
            precision = tp / (tp + fp) if (tp+fp)>0 else 1.0
            recall = tp / (tp + fn) if (tp+fn)>0 else 1.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision+recall)>0 else 1.0
            dice = (2 * tp) / (2 * tp + fp + fn) if (2*tp+fp+fn)>0 else 1.0

            metrics_data.append({
                "Subject": code,
                "Total Positive": total_positive,
                "Total Negative": total_negative,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Dice Score": dice
            })

            # Free memory
            del features, labels, predictions, features_filtered, labels_filtered, prediction_image
            torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Error encountered for subject {code}: {e}")
            torch.cuda.empty_cache()
            continue

    # Save metrics to CSV
    metrics_path = os.path.join(output_dir, "metrics.csv")
    df = pd.DataFrame(metrics_data)
    df.to_csv(metrics_path, index=False)
    logging.info(f"Metrics saved to {metrics_path}")
