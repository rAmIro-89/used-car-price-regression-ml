# Used Car Price Regression (Multiple Linear Regression Project)

> **Production-ready regression pipeline for automotive price prediction**  
> Predicts used car prices using multiple linear regression with robust error handling

## 🎯 Business Problem

Used car marketplaces and dealers need reliable price estimation to:
- **Improve pricing transparency** and build customer trust
- **Reduce negotiation friction** with data-driven valuations
- **Accelerate inventory turnover** through competitive pricing
- **Minimize pricing errors** that lead to lost revenue or slow sales

This model provides instant price predictions based on vehicle attributes, enabling faster, fairer transactions.

**Target Users:** Online automotive marketplaces, dealerships, appraisers, individual sellers

## 📊 Dataset

- **Size:** 6,000+ used car listings
- **Target Variable:** `price` (in lakhs - Indian currency unit)
- **Features Include:**
  - Vehicle specifications (brand, model year, fuel type, transmission)
  - Condition indicators (odometer/mileage, owner history)
  - Performance metrics (engine displacement, power, torque)
  - Physical attributes (seats, body type)

**Note:** 1 lakh = 100,000 Indian Rupees (₹) ≈ $1,500 CAD (varies by exchange rate)

### Price Distribution

![Price Distribution](img/01-price-distribution.png)
*Figure 1 – Distribution of used car prices in the dataset*

### Feature Correlations

![Correlation Matrix](img/02-correlation-matrix.png)
*Figure 2 – Correlation matrix of key features with price*

---

## 📈 Model Performance

| Model | Test RMSE (lakhs) | Test MAE (lakhs) | R² | MAPE |
|-------|------------------:|----------------:|----:|-----:|
| OLS Baseline | 12.18 | 6.85 | 0.265 | 42.3% |
| **Robust OLS (HC3)** | **12.18** | **6.85** | **0.265** | **42.3%** |
| Log-Price Model | 11.52 | 6.32 | 0.284 | 39.8% |

**Best Model:** Log-transformed target with robust standard errors  
**Performance:** Predicts prices within ±6.32 lakhs (~₹632k or $9,500 CAD) on average

---

## 🚀 How to Run

### Prerequisites
- Python 3.9 or higher
- 4GB RAM minimum

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/rAmIro-89/used-car-price-regression-ml.git
cd used-car-price-regression-ml

# 2. Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Jupyter notebooks
jupyter notebook notebooks/
```

### Quick Start with Source Code

```python
from src.data_utils import load_data, basic_cleaning, apply_log_transformation
from src.modeling import train_robust_regression, evaluate_regression

# Load and clean data
df = load_data('data/raw/Used Car Dataset.csv')
df_clean = basic_cleaning(df)

# Apply log transformation to target (stabilizes variance)
df_clean['log_price'] = apply_log_transformation(df_clean['price'])

# Train robust regression
model = train_robust_regression(X_train, y_train_log, cov_type='HC3')

# Evaluate on test set
metrics = evaluate_regression(
    model, X_test, y_test, 
    model_name="Robust Log-Price", 
    log_target=True  # Converts predictions back to original scale
)
print(f"Test RMSE: {metrics['rmse']:.2f} lakhs")
```

---

## 📁 Project Structure

```text
used-car-price-regression-ml/
├── data/
│   ├── raw/                      # Original datasets (gitignored)
│   └── processed/                # Cleaned datasets
├── notebooks/
│   ├── used-cars-eda.ipynb      # Exploratory data analysis
│   └── used-cars-multiple-regression.ipynb  # Model training & evaluation
├── src/
│   ├── __init__.py              # Package initialization
│   ├── data_utils.py            # Data loading & preprocessing functions
│   └── modeling.py              # Regression training & evaluation functions
├── models/                       # Saved trained models (gitignored)
├── .gitignore                    # Files to exclude from version control
├── requirements.txt              # Python dependencies
├── LICENSE
└── README.md
```

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Target distribution analysis (right-skewed, high-value outliers)
- Correlation analysis with numeric features (odometer, age, engine specs)
- Categorical feature impact (brand, fuel type, transmission)
- Missing value and duplicate detection

### Model Diagnostics

![Residuals Plot](img/03-residuals.png)
*Figure 3 – Residual analysis showing model fit quality*

### Predictions vs Actuals

![Actual vs Predicted](img/04-actual-vs-predicted.png)
*Figure 4 – Model predictions compared to actual prices (R² = 0.284)*

### 2. Data Cleaning & Feature Engineering

- **Column Removal:** Remove low-utility columns (car_name, registration_year, insurance_validity)
- **Feature Engineering:** Create `vehicle_age` from model_year; develop mileage categories
- **Outlier Detection:** Apply IQR method with 1.5× threshold
- **Transformation:** Log-transform target (price) to stabilize variance
- **Encoding:** One-hot encode categorical variables (brand, fuel_type, transmission)
- **Scaling:** Standardize numeric features for consistent interpretation

### 3. Modeling & Validation

- **Data Split:** Train/test 80/20 ratio, preserving price distribution
- **Baseline Model:** OLS regression with detailed diagnostics
- **Robust Estimation:** HC3 heteroscedasticity-robust standard errors
- **Log-Price Model:** Test log-transformed target to improve R² 
- **Multicollinearity:** VIF calculation for all predictors
- **Diagnostics:** Durbin-Watson, Jarque-Bera, Breusch-Pagan tests

### 4. Evaluation & Results

- **Metrics:** RMSE, MAE, R², MAPE across model variants
- **Residuals:** Normality tests, homoscedasticity checks
- **Predictions:** Actual vs predicted scatter plots with confidence intervals
- **Coefficients:** Feature importance and elasticity calculations
- **Bias Correction:** Retransformation adjustments for log models

---

## 📊 Key Findings

- **Best Predictors:** Vehicle age, odometer reading, engine power, brand
- **Log Transformation:** Improved R² from 0.265 to 0.284 and reduced MAPE to ~40%
- **Robust Errors:** HC3 covariance provides valid inference despite heteroscedasticity
- **Multicollinearity:** Engine displacement and power moderately correlated (VIF ~5-6)
- **Prediction Accuracy:** Model explains ~28% of price variance (typical for complex used car markets)

---

## 🛠️ Technologies & Stack

| Component | Technology |
|-----------|-----------|
| **Data Processing** | Pandas, NumPy |
| **Statistical Modeling** | Statsmodels, SciPy |
| **Visualization** | Matplotlib, Seaborn |
| **ML Framework** | Scikit-learn |
| **Notebooks** | Jupyter |
| **Language** | Python 3.9+ |

---

## 🎯 Skills Demonstrated

- **Regression Analysis:** OLS estimation, robust covariance estimation, log-transformation techniques
- **Data Preprocessing:** Feature engineering, outlier detection, multicollinearity assessment  
- **Statistical Testing:** Heteroscedasticity diagnosis, normality tests, residual analysis
- **Visualization:** Distribution plots, correlation matrices, residuals diagnostics, prediction scatter plots
- **Business Insights:** Real-world price prediction for marketplace decisions
- **Model Diagnostics:** VIF calculation, coefficient interpretation, assumption validation
- **Python Stack:** Pandas, NumPy, Statsmodels, Scikit-learn, Matplotlib, Seaborn

---

## 🚀 How to Run

### 1. Exploratory Analysis
```bash
jupyter notebook notebooks/used-cars-eda.ipynb
```
Explore data distributions, missing values, correlations, and categorical relationships.

### 2. Model Training & Evaluation
```bash
jupyter notebook notebooks/used-cars-multiple-regression.ipynb
```
Train OLS, robust, and log-transformed models. Conduct residual diagnostics.

### 3. Quick Usage
```bash
python example_usage.py
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Ramiro Ottone Villar**  
[![GitHub](https://img.shields.io/badge/GitHub-rAmIro--89-181717?style=flat&logo=github)](https://github.com/rAmIro-89)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/your-profile)

---

⭐ **If you find this project useful, please consider starring the repository!**
