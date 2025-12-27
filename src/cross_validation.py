import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                           f1_score, precision_score, recall_score, accuracy_score)
import warnings
warnings.filterwarnings('ignore')
import os

class CrossValidator:
    """
    Perform stratified k-fold cross-validation with comprehensive metrics
    """
    
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.results = {}
        
    def cross_validate(self, X, y, model, model_name="model"):
        """
        Perform stratified k-fold cross-validation
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        y : array-like, shape (n_samples,)
            Target labels
        model : sklearn estimator
            Model to validate
        model_name : str
            Name of the model for results tracking
            
        Returns:
        --------
        dict: Aggregated metrics across folds
        """
        print(f"\n{'='*60}")
        print(f"Stratified {self.n_splits}-Fold CV for {model_name}")
        print(f"{'='*60}")
        
        # Validate inputs
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be pandas DataFrame or numpy array")
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be pandas Series or numpy array")
        
        # Convert to DataFrame/Series for consistency
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        # Initialize metric storage
        fold_metrics = {
            'accuracy': [], 'precision': [], 'recall': [], 
            'f1': [], 'roc_auc': [], 'pr_auc': []
        }
        
        fold_results = []
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(X, y), 1):
            try:
                # Split data
                X_train_fold = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
                X_val_fold = X.iloc[val_idx] if isinstance(X, pd.DataFrame) else X[val_idx]
                y_train_fold = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
                y_val_fold = y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx]
                
                # Clone model to avoid contamination between folds
                from sklearn.base import clone
                model_clone = clone(model)
                
                # Train model
                model_clone.fit(X_train_fold, y_train_fold)
                
                # Predict
                y_pred = model_clone.predict(X_val_fold)
                
                # Handle models that might not have predict_proba
                try:
                    y_pred_proba = model_clone.predict_proba(X_val_fold)[:, 1]
                except (AttributeError, IndexError):
                    # For models without predict_proba, use decision function
                    try:
                        y_pred_proba = model_clone.decision_function(X_val_fold)
                    except:
                        y_pred_proba = y_pred  # Fallback to binary predictions
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_val_fold, y_pred, y_pred_proba)
                
                # Store fold results
                for metric_name, value in metrics.items():
                    fold_metrics[metric_name].append(value)
                
                fold_results.append({
                    'fold': fold,
                    'train_size': len(X_train_fold),
                    'val_size': len(X_val_fold),
                    'fraud_rate_train': y_train_fold.mean() if hasattr(y_train_fold, 'mean') else np.mean(y_train_fold),
                    'fraud_rate_val': y_val_fold.mean() if hasattr(y_val_fold, 'mean') else np.mean(y_val_fold),
                    **metrics
                })
                
                print(f"  Fold {fold}: PR-AUC = {metrics['pr_auc']:.4f}, "
                      f"Recall = {metrics['recall']:.4f}, "
                      f"Train Fraud Rate = {fold_results[-1]['fraud_rate_train']:.4f}")
                
            except Exception as e:
                print(f"  ⚠️  Error in fold {fold}: {str(e)[:100]}...")
                # Store NaN for failed fold
                failed_metrics = {k: np.nan for k in fold_metrics.keys()}
                for metric_name in fold_metrics.keys():
                    fold_metrics[metric_name].append(np.nan)
                fold_results.append({
                    'fold': fold,
                    'train_size': 0,
                    'val_size': 0,
                    'fraud_rate_train': np.nan,
                    'fraud_rate_val': np.nan,
                    **failed_metrics,
                    'error': str(e)
                })
        
        # Calculate aggregated metrics
        aggregated = self._aggregate_metrics(fold_metrics)
        
        # Store results
        self.results[model_name] = {
            'fold_results': fold_results,
            'aggregated': aggregated,
            'fold_metrics': fold_metrics
        }
        
        # Print summary
        self._print_summary(model_name, aggregated)
        
        return aggregated
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate all evaluation metrics with error handling"""
        try:
            # Ensure arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_pred_proba = np.array(y_pred_proba)
            
            # Handle edge cases
            if len(np.unique(y_true)) < 2:
                return {k: np.nan for k in ['accuracy', 'precision', 'recall', 
                                           'f1', 'roc_auc', 'pr_auc']}
            
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0)
            }
            
            # Try ROC-AUC (might fail if only one class)
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = np.nan
            
            # Try PR-AUC
            try:
                metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
            except:
                metrics['pr_auc'] = np.nan
                
            return metrics
            
        except Exception as e:
            print(f"  Error calculating metrics: {e}")
            return {k: np.nan for k in ['accuracy', 'precision', 'recall', 
                                       'f1', 'roc_auc', 'pr_auc']}
    
    def _aggregate_metrics(self, fold_metrics):
        """Aggregate metrics across folds"""
        aggregated = {}
        for metric_name, values in fold_metrics.items():
            values_array = np.array(values)
            valid_values = values_array[~np.isnan(values_array)]
            
            if len(valid_values) > 0:
                aggregated[f'{metric_name}_mean'] = np.mean(valid_values)
                aggregated[f'{metric_name}_std'] = np.std(valid_values)
                aggregated[f'{metric_name}_min'] = np.min(valid_values)
                aggregated[f'{metric_name}_max'] = np.max(valid_values)
                aggregated[f'{metric_name}_n_folds'] = len(valid_values)
            else:
                aggregated[f'{metric_name}_mean'] = np.nan
                aggregated[f'{metric_name}_std'] = np.nan
                aggregated[f'{metric_name}_min'] = np.nan
                aggregated[f'{metric_name}_max'] = np.nan
                aggregated[f'{metric_name}_n_folds'] = 0
        return aggregated
    
    def _print_summary(self, model_name, aggregated):
        """Print summary of cross-validation results"""
        print(f"\n{'='*40}")
        print(f"CV SUMMARY - {model_name}")
        print(f"{'='*40}")
        
        if aggregated['pr_auc_n_folds'] > 0:
            print(f"PR-AUC:    {aggregated['pr_auc_mean']:.4f} ± {aggregated['pr_auc_std']:.4f} "
                  f"(n={aggregated['pr_auc_n_folds']})")
            print(f"Recall:    {aggregated['recall_mean']:.4f} ± {aggregated['recall_std']:.4f}")
            print(f"ROC-AUC:   {aggregated['roc_auc_mean']:.4f} ± {aggregated['roc_auc_std']:.4f}")
            print(f"F1-Score:  {aggregated['f1_mean']:.4f} ± {aggregated['f1_std']:.4f}")
            print(f"Accuracy:  {aggregated['accuracy_mean']:.4f} ± {aggregated['accuracy_std']:.4f}")
        else:
            print("⚠️  No valid folds to aggregate")
    
    def save_results(self, model_name, filepath):
        """Save cross-validation results to CSV"""
        try:
            if model_name not in self.results:
                print(f"⚠️  No results found for model: {model_name}")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save fold-level results
            fold_results = pd.DataFrame(self.results[model_name]['fold_results'])
            fold_file = filepath.replace('.csv', '_folds.csv')
            fold_results.to_csv(fold_file, index=False)
            
            # Save aggregated results
            aggregated = pd.DataFrame([self.results[model_name]['aggregated']])
            aggregated.to_csv(filepath, index=False)
            
            print(f"  ✓ Results saved to {filepath}")
            print(f"  ✓ Fold details saved to {fold_file}")
            return True
            
        except Exception as e:
            print(f"  ❌ Error saving results: {e}")
            return False
    
    def get_model_ranking(self, metric='pr_auc_mean'):
        """Rank models based on specified metric"""
        try:
            model_scores = []
            for model_name, results in self.results.items():
                if 'aggregated' in results:
                    score = results['aggregated'].get(metric, np.nan)
                    n_folds = results['aggregated'].get(f'{metric.split("_")[0]}_n_folds', 0)
                    model_scores.append({
                        'model': model_name,
                        'score': score,
                        'n_folds': n_folds
                    })
            
            # Sort by score (descending), handle NaN
            model_scores.sort(key=lambda x: x['score'] if not np.isnan(x['score']) else -np.inf, reverse=True)
            
            return model_scores
            
        except Exception as e:
            print(f"Error ranking models: {e}")
            return []
    
    def plot_cv_results(self, metric='pr_auc', save_path=None):
        """Create visualization of CV results"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(12, 6))
            
            # Prepare data for plotting
            plot_data = []
            for model_name, results in self.results.items():
                if 'fold_metrics' in results and metric in results['fold_metrics']:
                    for fold, value in enumerate(results['fold_metrics'][metric], 1):
                        if not np.isnan(value):
                            plot_data.append({
                                'Model': model_name,
                                'Fold': f'Fold {fold}',
                                metric.upper(): value
                            })
            
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                
                # Create box plot
                plt.subplot(1, 2, 1)
                sns.boxplot(data=plot_df, x='Model', y=metric.upper())
                plt.xticks(rotation=45)
                plt.title(f'{metric.upper()} Distribution Across Folds')
                
                # Create bar plot with error bars
                plt.subplot(1, 2, 2)
                model_stats = []
                for model_name in plot_df['Model'].unique():
                    model_values = plot_df[plot_df['Model'] == model_name][metric.upper()]
                    model_stats.append({
                        'Model': model_name,
                        'Mean': model_values.mean(),
                        'Std': model_values.std()
                    })
                
                stats_df = pd.DataFrame(model_stats)
                stats_df = stats_df.sort_values('Mean', ascending=False)
                
                plt.bar(range(len(stats_df)), stats_df['Mean'], 
                       yerr=stats_df['Std'], capsize=5, alpha=0.7)
                plt.xticks(range(len(stats_df)), stats_df['Model'], rotation=45)
                plt.title(f'{metric.upper()} Mean ± Std')
                plt.ylabel(metric.upper())
                
                plt.tight_layout()
                
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"  ✓ Plot saved to {save_path}")
                
                plt.show()
            else:
                print("  ⚠️  No valid data to plot")
                
        except Exception as e:
            print(f"  ⚠️  Error creating plot: {e}")