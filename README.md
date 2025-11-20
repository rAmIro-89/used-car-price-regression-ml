# Used Car Price Regression (Multiple Linear Regression Project)

Professional end-to-end regression project to predict **used car prices** using multiple linear regression, robust regression techniques, diagnostics, and model comparison.

## 📊 Dataset
- Located in: `data/raw/`
- Target: **Price**
- Features include brand, age, mileage, fuel type, engine specs, and other vehicle attributes.

## 🔬 Methodology
1. **EDA**
   - Distribution analysis
   - Outliers
   - Correlations
   - Feature relationships

2. **Feature Engineering**
   - Dummy encoding
   - Log transformations
   - Outlier handling
   - Scaling when necessary

3. **Modeling**
   - OLS (statsmodels)
   - HC3 robust regression
   - VIF multicollinearity checks
   - Residual diagnostics

4. **Model Comparison**
   - RMSE, MAE, R²
   - Predictions vs actual plots
   - Influence of features

## 📁 Project Structure
```text
used-car-price-regression-ml/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── reports/
│   └── figures/
├── src/
├── models/
└── README.md
```
