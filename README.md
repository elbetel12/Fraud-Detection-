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