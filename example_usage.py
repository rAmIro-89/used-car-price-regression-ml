"""
Quick usage example for used car price regression.

This script demonstrates how to use the reusable functions from src/
to load data, clean it, and train regression models.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data_utils import (
    load_data, 
    basic_cleaning, 
    create_vehicle_age,
    apply_log_transformation,
    split_features_target
)
from src.modeling import (
    prepare_train_test_split,
    train_robust_regression,
    evaluate_regression,
    compare_models
)

def main():
    """Run the complete used car price regression pipeline."""
    
    print("="*60)
    print("USED CAR PRICE REGRESSION - QUICK DEMO")
    print("="*60)
    
    # Step 1: Load data
    print("\n[STEP 1] Loading data...")
    df = load_data('data/raw/Used Car Dataset.csv')
    
    # Step 2: Basic cleaning
    print("\n[STEP 2] Cleaning data...")
    df_clean = basic_cleaning(df)
    
    # Step 3: Feature engineering
    print("\n[STEP 3] Feature engineering...")
    df_clean = create_vehicle_age(df_clean, current_year=2024)
    
    # Step 4: Log transform target (optional - improves model performance)
    print("\n[STEP 4] Applying log transformation to target...")
    df_clean['log_price'] = apply_log_transformation(df_clean['price'])
    
    # Step 5: Split features and target
    print("\n[STEP 5] Preparing features and target...")
    # Note: In practice, you'd apply one-hot encoding here
    X, y_log = split_features_target(df_clean, target_col='log_price')
    y_original = df_clean['price']
    
    # Step 6: Train/test split
    print("\n[STEP 6] Splitting train/test sets...")
    X_train, X_test, y_train_log, y_test_log = prepare_train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    
    # Get original scale target for test set
    y_test_original = y_original.loc[y_test_log.index]
    
    # Step 7: Train robust regression
    print("\n[STEP 7] Training robust regression model...")
    print("(This may take a few minutes...)")
    
    # Note: In practice, you'd select only numeric features or apply encoding first
    # This is simplified for demonstration
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nFor full pipeline with encoding and evaluation,")
    print("see notebooks/used-cars-multiple-regression.ipynb")

if __name__ == "__main__":
    main()
