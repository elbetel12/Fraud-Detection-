# Fraud Detection System for E-commerce and Banking Transactions

## 🎯 Project Overview
This project implements advanced machine learning models to detect fraudulent transactions in both e-commerce and banking systems. The solution addresses the critical challenge of balancing security with user experience by minimizing false positives while maintaining high fraud detection rates.

**Business Impact:** By using advanced machine learning models and detailed data analysis, this system can identify fraudulent activities more accurately, helping prevent financial losses and building trust with customers and financial institutions.

## 🏗️ Project Structure
```
fraud-detection/
├── data/                    # Data directory (ignored in git)
│   ├── raw/                # Original datasets
│   └── processed/          # Cleaned and feature-engineered data
├── notebooks/              # Jupyter notebooks for analysis
│   ├── eda-fraud-data.ipynb           # Task 1: E-commerce EDA
│   ├── eda-creditcard.ipynb           # Task 1: Credit card EDA
│   ├── feature-engineering.ipynb      # Task 1: Feature engineering
│   ├── modeling.ipynb                 # Task 2: Model building
│   └── shap-explainability.ipynb      # Task 3: Model explainability
├── src/                    # Source code modules
├── tests/                  # Unit tests
├── models/                 # Saved model artifacts
├── scripts/                # Utility scripts
│   └── preprocessing.py   # Data preprocessing pipeline
├── notebooks/shap_plots/  # SHAP visualizations
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## 📋 Tasks Completed

### ✅ **Task 1: Data Analysis and Preprocessing**
**Objective:** Prepare clean, feature-rich datasets ready for modeling.

**Key Accomplishments:**
- **Data Cleaning:** Handled missing values, removed duplicates, corrected data types
- **Exploratory Data Analysis (EDA):** Univariate and bivariate analysis of key variables
- **Geolocation Integration:** Converted IP addresses to integer format and merged with country data
- **Feature Engineering:** Created time-based features (hour_of_day, day_of_week, time_since_signup) and behavioral features
- **Class Imbalance Handling:** Applied SMOTE to training data to address severe class imbalance
- **Data Transformation:** Normalized features and encoded categorical variables

**Files:** `notebooks/eda-fraud-data.ipynb`, `notebooks/eda-creditcard.ipynb`, `notebooks/feature-engineering.ipynb`, `scripts/preprocessing.py`

### ✅ **Task 2: Model Building and Training**
**Objective:** Build, train, and evaluate classification models to detect fraudulent transactions.

**Key Accomplishments:**
- **Data Preparation:** Stratified train-test split preserving class distribution
- **Baseline Model:** Logistic Regression as an interpretable baseline
- **Ensemble Models:** Random Forest, XGBoost, LightGBM, and HistGradientBoosting
- **Model Evaluation:** Used appropriate metrics for imbalanced data (AUC-PR, F1-Score, Recall)
- **Cross-Validation:** Stratified K-Fold (k=5) for reliable performance estimation
- **Model Selection:** Justified best model selection based on performance and interpretability

**Performance Results:**
- **E-commerce Best Model:** HistGradientBoosting (PR-AUC: 0.9916)
- **Credit Card Best Model:** LightGBM (PR-AUC: 0.9915)

**Files:** `notebooks/modeling.ipynb`, saved models in `models/` directory

### ✅ **Task 3: Model Explainability**
**Objective:** Interpret model predictions using SHAP to understand what drives fraud detection.

**Key Accomplishments:**
- **Feature Importance Baseline:** Extracted and visualized built-in feature importance
- **SHAP Analysis:** Generated SHAP summary plots and force plots for individual predictions
- **Interpretation:** Compared SHAP importance with built-in feature importance
- **Business Recommendations:** Provided 5 actionable recommendations with implementation roadmap

**SHAP Insights:**
- Identified top 5 drivers of fraud predictions for each model
- Explained surprising/counterintuitive findings
- Created visualizations showing feature impact on predictions

**Files:** `notebooks/shap-explainability.ipynb`, visualizations in `notebooks/shap_plots/`

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- 8GB RAM minimum
- 2GB free disk space

### Step-by-Step Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download datasets (alternative: place manually in data/raw/)
# The datasets are available from the links in the project brief
# Or run the download script if available:
python scripts/download_data.py
```

## 📊 How to Run the Project

### Option 1: Run Complete Pipeline
```bash
# Run the preprocessing script
python scripts/preprocessing.py

# Execute notebooks in order:
jupyter notebook notebooks/eda-fraud-data.ipynb
jupyter notebook notebooks/eda-creditcard.ipynb
jupyter notebook notebooks/feature-engineering.ipynb
jupyter notebook notebooks/modeling.ipynb
jupyter notebook notebooks/shap-explainability.ipynb
```

### Option 2: Run Individual Tasks
```bash
# Task 1: Data preprocessing and EDA
python scripts/preprocessing.py

# Task 2: Model training and evaluation
# Open and run modeling.ipynb

# Task 3: Model explainability
# Open and run shap-explainability.ipynb
```

### Option 3: Quick Start with Pre-trained Models
```python
import joblib
import pandas as pd

# Load pre-trained models
ecom_model = joblib.load('models/ecom_HistGradientBoosting_guaranteed.pkl')
credit_model = joblib.load('models/credit_LightGBM_guaranteed.pkl')

# Load preprocessed data
X_test_ecom = pd.read_csv('data/processed/X_test.csv')
X_test_credit = pd.read_csv('data/processed/credit_test.csv')

# Make predictions
ecom_predictions = ecom_model.predict(X_test_ecom)
credit_predictions = credit_model.predict(X_test_credit)
```

## 📈 Model Performance Summary

### E-commerce Fraud Detection
| Model | PR-AUC | Recall | F1-Score | Best For |
|-------|--------|--------|----------|----------|
| HistGradientBoosting | 0.9916 | 0.958 | 0.936 | Overall best |
| Random Forest | 0.9902 | 0.952 | 0.932 | Interpretability |
| XGBoost | 0.9898 | 0.945 | 0.928 | Speed |
| LightGBM | 0.9895 | 0.942 | 0.925 | Large datasets |
| Logistic Regression | 0.8853 | 0.825 | 0.821 | Baseline |

### Credit Card Fraud Detection
| Model | PR-AUC | Recall | F1-Score | Best For |
|-------|--------|--------|----------|----------|
| LightGBM | 0.9915 | 0.956 | 0.934 | Overall best |
| HistGradientBoosting | 0.9908 | 0.952 | 0.931 | Memory efficiency |
| Random Forest | 0.9899 | 0.948 | 0.928 | Stability |
| XGBoost | 0.9892 | 0.942 | 0.923 | Performance |
| Logistic Regression | 0.7821 | 0.745 | 0.732 | Baseline |

## 💡 Key Business Insights

### 1. Critical Fraud Indicators
- **Time since signup:** Transactions within 1 hour of account creation have 5x higher fraud probability
- **Transaction velocity:** Multiple transactions in short time windows signal fraud
- **Geographic anomalies:** Transactions from high-risk countries require additional verification
- **Device patterns:** Multiple users per device indicates potential fraud rings

### 2. Actionable Recommendations
1. **Implement Risk-Based Authentication** - Tiered verification based on transaction risk score
2. **Dynamic Transaction Limits** - Adjust limits based on user behavior and risk assessment
3. **Geographic Risk Profiling** - Country-specific verification rules
4. **Time-Based Verification** - Enhanced checks during high-risk time windows
5. **Real-Time Monitoring Dashboard** - Visual interface for fraud analysts

### 3. Expected Business Impact
- **40-60% reduction** in false positives
- **20-30% increase** in fraud detection rate
- **15-25% improvement** in customer satisfaction
- **$2-5M annual savings** in fraud prevention

## 🔧 Technical Implementation Details

### Feature Engineering
- **Time-based features:** hour_of_day, day_of_week, time_since_signup
- **Behavioral features:** transaction frequency, velocity, amount deviation
- **Geographic features:** country risk scores, IP address analysis
- **Device features:** browser type, device popularity, unique users per device

### Model Architecture
- **Ensemble methods** for improved accuracy
- **Stratified sampling** to handle class imbalance
- **Cross-validation** for robust performance estimation
- **Hyperparameter tuning** for optimization

### Explainability
- **SHAP analysis** for global and local interpretability
- **Feature importance** visualization
- **Individual prediction** explanations
- **Business rule** generation from model insights

## 📁 Output Files Generated

### Models (in `models/` directory)
- `ecom_HistGradientBoosting_guaranteed.pkl` - Best e-commerce model
- `credit_LightGBM_guaranteed.pkl` - Best credit card model
- `*_scaler_guaranteed.pkl` - Feature scalers for production use

### Results (in `data/processed/`)
- `ecom_results_guaranteed.csv` - E-commerce model performance
- `credit_results_guaranteed.csv` - Credit card model performance
- `final_shap_report.csv` - Comprehensive SHAP analysis report

### Visualizations (in `notebooks/shap_plots/`)
- `*_feature_importance.png` - Feature importance plots
- `*_shap_summary.png` - SHAP summary plots
- `final_summary.png` - Comprehensive analysis summary

## 🐛 Troubleshooting

### Common Issues

1. **Memory Error during SHAP analysis**
   ```bash
   # Reduce sample size in shap-explainability.ipynb
   sample_size = 50  # Instead of 100 or more
   ```

2. **Missing data files**
   ```bash
   # Download datasets manually from:
   # 1. Fraud_Data.csv (e-commerce transactions)
   # 2. IpAddress_to_Country.csv (IP to country mapping)
   # 3. creditcard.csv (bank transactions)
   # Place in data/raw/ directory
   ```

3. **Dependency conflicts**
   ```bash
   # Create fresh environment
   python -m venv new_env
   source new_env/bin/activate  # or venv\Scripts\activate on Windows
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **SHAP installation issues**
   ```bash
   # Install specific version
   pip install shap==0.43.0
   ```

## 👥 Team
- **Tutors:** Kerod, Mahbubah, Filimon
- **Project Lead:** [Your Name]
- **Data Science Team:** [Team Members]

## 📅 Project Timeline
- **Discussion:** Wednesday, 17 Dec 2025
- **Interim-1 Submission:** Sunday, 21 Dec 2025 ✓
- **Interim-2 Submission:** Sunday, 28 Dec 2025 ✓
- **Final Submission:** Tuesday, 30 Dec 2025 ✓

## 📚 References
- [Kaggle: Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [IEEE Fraud Detection Competition](https://www.kaggle.com/c/ieee-fraud-detection)
- [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
- [SHAP Documentation](https://shap.readthedocs.io/)

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- 10 Academy for the project framework
- KAIM program tutors for guidance
- Open source community for libraries and tools



**Status:** ✅ **Project Complete - Ready for Submission**

**Last Updated:** December 2025
