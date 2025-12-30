import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score)
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Train and evaluate models with error handling"""
    
    def __init__(self, random_state=42, verbose=True):
        self.random_state = random_state
        self.verbose = verbose
        self.models = {}
        self.results = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name="model"):
        """Evaluate a trained model"""
        try:
            # Predict
            y_pred = model.predict(X_test)
            
            # Get probabilities if available
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except:
                y_pred_proba = y_pred
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
            
            # Try AUC metrics
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                metrics['pr_auc'] = average_precision_score(y_test, y_pred_proba)
            except:
                metrics['roc_auc'] = np.nan
                metrics['pr_auc'] = np.nan
            
            if self.verbose:
                print(f"  {model_name}: PR-AUC={metrics['pr_auc']:.4f}, "
                      f"Recall={metrics['recall']:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error evaluating model: {e}")
            return {k: np.nan for k in ['accuracy', 'precision', 'recall', 
                                       'f1', 'roc_auc', 'pr_auc']}