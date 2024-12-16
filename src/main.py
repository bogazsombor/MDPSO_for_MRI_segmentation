# main.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import OUTPUT_DIR, training_codes, testing_codes
from modeling.train import train_model
from modeling.test import test_model_on_subjects

def main():
    # Ensure output directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Train the model
    model = train_model(training_codes)

    # Test the model
    test_model_on_subjects(model, testing_codes, OUTPUT_DIR)

if __name__ == "__main__":
    main()
