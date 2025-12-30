import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Robust data processor with error handling"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        
    def clean_and_scale(self, X, y=None, fit=True):
        """Clean and scale data with error handling"""
        try:
            # Validate input
            if X is None or len(X) == 0:
                raise ValueError("Input data is empty")
            
            # Convert to DataFrame if not already
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            
            if self.verbose:
                print(f"Cleaning data: {X.shape}")
            
            # Handle missing values
            if self.imputer is None or fit:
                self.imputer = SimpleImputer(strategy='median')
                X_clean = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
            else:
                X_clean = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
            
            # Scale features
            if self.scaler is None or fit:
                self.scaler = StandardScaler()
                X_scaled = pd.DataFrame(self.scaler.fit_transform(X_clean), columns=X_clean.columns)
            else:
                X_scaled = pd.DataFrame(self.scaler.transform(X_clean), columns=X_clean.columns)
            
            self.feature_names = X_scaled.columns.tolist()
            
            if self.verbose:
                print(f"✓ Data cleaned and scaled: {X_scaled.shape}")
                print(f"  Missing values: {X_scaled.isnull().sum().sum()}")
            
            if y is not None:
                return X_scaled, y
            return X_scaled
            
        except Exception as e:
            print(f"❌ Error in clean_and_scale: {e}")
            raise