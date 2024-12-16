# modeling/train.py
import torch
from joblib import dump
from cuml.ensemble import RandomForestClassifier as cuRF
from params import OUTPUT_DIR
from features.extract_features import extract_features_and_labels
import os

def train_gpu_random_forest(features, labels):
    """
    Train a GPU-based Random Forest using cuML.
    Returns the trained model.
    """
    rf = cuRF(
        n_estimators=500,
        max_depth=35,
        max_features='sqrt',
        min_samples_split=5,
        min_samples_leaf=2,
        split_criterion='entropy',
        bootstrap=True,
        max_samples=0.8,
        n_bins=256,
        n_streams=8,
        random_state=42,
        verbose=False,
    )
    trained_rf = rf.fit(features.cpu().numpy(), labels.cpu().numpy())
    return trained_rf

def train_model(training_codes):
    all_features = []
    all_labels = []
    all_feature_labels = None

    print("Starting training feature extraction...")
    for code in training_codes:
        features, labels, feature_labels = extract_features_and_labels(code, sample=True)
        all_features.append(features)
        all_labels.append(labels)
        if all_feature_labels is None:
            all_feature_labels = feature_labels

    # Concatenate all training data
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print("Training Random Forest model...")
    rf_model = train_gpu_random_forest(all_features, all_labels)

    # Save the model
    model_path = os.path.join(str(OUTPUT_DIR), "trained_model.joblib")
    dump(rf_model, model_path)
    print(f"Model saved to {model_path}")

    return rf_model
