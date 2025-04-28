# Image Segmentation Project

## Overview  
This project is focused on medical image segmentation, specifically for the BraTS dataset. It uses a Random Forest model to predict and evaluate region segmentation in MRI scans for multiple modalities.  The project is currently being updated to include 3D segmentation.

### Features:  
- Data loading and preprocessing of four MRI modalities.  
- Extracts both voxel-wise and regional features.  
- Cluster-based feature extraction using MDPSO.  
- Prediction with a model trained on voxel-wise features.  
- Outputs include predicted segmentation images (as PNGs) and evaluation metrics (CSV).  

## Setup  
1. **Clone the repository**  
   ```bash  
   git clone https://github.com/bogazsombor/MDPSO_for_MRI_segmentation
   ```  

2. **Navigate to the project directory**  
   ```bash  
   cd MDPSO_for_MRI_segmentation  
   ```  

3. **Create and activate a virtual environment**  
   ```bash  
   python -m venv venv  
   source venv/bin/activate  # On Windows: venv\Scripts\activate  
   ```  

4. **Install dependencies**  
   ```bash  
   pip install -r requirements.txt  
   ```  

## Dataset  
The project uses the **BraTS 2021 dataset**. Ensure the dataset is downloaded and placed into the `data/raw/` directory. Preprocessing scripts will generate `data/preprocessed/` as needed.  

## Running the Training  
To train the model on the training data:  
```bash  
python src/main.py  
```  

## Running Predictions  
After training, test the model and generate predictions:  
```bash  
python src/modeling/test.py  
```  

## Metrics  
The testing process generates a CSV file containing the following metrics:  
- **accuracy**: Proportion of correctly labeled voxels.  
- **precision**: Correctly predicted tumors / all predicted tumors.  
- **recall**: Correctly predicted tumors / all actual tumors.  
- **Dice score**: Overlap measure between predicted and ground truth segmentation regions.  

## Outputs  
- **Prediction images**: Saved as PNG files in `output/predictions/`.  
- **Evaluation metrics**: Saved as a CSV file in `output/metrics.csv`.  

## License  
This project is licensed under the MIT License.  
