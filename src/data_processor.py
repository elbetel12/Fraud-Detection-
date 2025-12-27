# src/data_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    """Process and clean fraud detection data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def clean_data(self, df):
        """Clean raw data"""
        # Your cleaning logic here
        return df
    
    def engineer_features(self, df):
        """Create new features"""
        # Your feature engineering logic here
        return df

# src/model_trainer.py
from sklearn.model_selection import train_test_split
import joblib

class ModelTrainer:
    """Train and save fraud detection models"""
    
    def train_model(self, X, y, model_type='rf'):
        """Train specified model type"""
        # Your training logic here
        return model
    
    def save_model(self, model, path):
        """Save model to disk"""
        joblib.dump(model, path)