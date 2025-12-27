# Fraud Detection System

## Project Overview
This project implements machine learning models to detect fraudulent transactions in e-commerce and banking systems. The system uses advanced feature engineering, geolocation analysis, and ensemble learning to accurately identify fraud while minimizing false positives.

## Features
- **Data Integration**: Merges transaction data with IP geolocation
- **Feature Engineering**: Creates time-based and behavioral features
- **Model Training**: Multiple models with hyperparameter tuning
- **Explainability**: SHAP analysis for model interpretation
- **Real-time Ready**: Pipeline designed for production deployment

## Installation
```bash
# Clone repository
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download data (requires gdown)
python scripts/download_data.py

# Fraud Detection MLOps Project

## Project Structure
fraud-detection-mlops/
├── data/ # Data storage
├── notebooks/ # Jupyter notebooks for analysis
├── src/ # Source code modules
├── tests/ # Unit tests
├── models/ # Trained model artifacts
└── reports/ # Generated reports and visualizations

text

## Tasks Completed

### Task 1: Exploratory Data Analysis ✓
- E-commerce fraud data analysis
- Credit card fraud data analysis
- Location: `notebooks/Task1_EDA.ipynb`

### Task 2: Model Building & Evaluation ✓
- Built 5+ models for each dataset
- Best models: HistGradientBoosting (e-commerce) and LightGBM (credit)
- Location: `notebooks/Task2_Modeling.ipynb`

### Task 3: Model Explainability (Next)
- SHAP analysis for model interpretation
- Location: `notebooks/Task3_SHAP.ipynb`

## How to Run

1. Clone repository:
```bash
git clone [your-repo-url]
cd fraud-detection-mlops
Install dependencies:

bash
pip install -r requirements.txt
Run notebooks in order:

bash
jupyter notebook notebooks/Task1_EDA.ipynb
Key Results
E-commerce PR-AUC: 0.9916 (HistGradientBoosting)

Credit Card PR-AUC: 0.9915 (LightGBM)

All models saved in models/ directory