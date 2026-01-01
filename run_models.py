import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb

# Paths
BASE_DIR = r"d:/SLIIT Study Materials/Y4S1/Research/New model/C2"
DATA_PATH = os.path.join(BASE_DIR, "price prediction", "Data", "coffee_price_simulation", "coffee_price_monthly_dataset_CORRECTED.csv")
PRICE_MODEL_PATH = os.path.join(BASE_DIR, "price prediction", "coffee_price_xgboost_model.pkl")
DEMAND_MODEL_PATH = os.path.join(BASE_DIR, "demand prediction", "coffee_demand_model.pkl")

def run_predictions():
    print("--- Coffee Demand and Price Prediction ---")
    
    # 1. Load Data
    print(f"Loading data from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print("Error: Dataset not found.")
        return
    
    df = pd.read_csv(DATA_PATH)
    
    # 2. Feature Engineering
    print("Performing feature engineering...")
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 3. Load Models
    print("Loading models...")
    price_model = joblib.load(PRICE_MODEL_PATH)
    demand_model = joblib.load(DEMAND_MODEL_PATH)
    
    # 4. Predict Price
    # Features required for Price Model (from notebook):
    # ['predicted_yield_kg', 'global_price_usd_kg', 'usd_lkr_rate', 'demand_index', 'month_sin', 'month_cos']
    PRICE_FEATURES = [
        'predicted_yield_kg', 
        'global_price_usd_kg', 
        'usd_lkr_rate', 
        'demand_index', 
        'month_sin', 
        'month_cos'
    ]
    X_price = df[PRICE_FEATURES]
    df['predicted_price_lkr'] = price_model.predict(X_price)
    
    # 5. Predict Demand
    # Features required for Demand Model (from notebook):
    # ['local_coffee_price_lkr_per_kg', 'global_price_usd_kg', 'usd_lkr_rate', 'month_sin', 'month_cos']
    # Note: The notebook uses 'local_coffee_price_lkr_per_kg' (historical) for training.
    # We will use the same for the index, but could potentially use the predicted price too if desired for future.
    DEMAND_FEATURES = [
        'local_coffee_price_lkr_per_kg', 
        'global_price_usd_kg', 
        'usd_lkr_rate', 
        'month_sin', 
        'month_cos'
    ]
    X_demand = df[DEMAND_FEATURES]
    df['predicted_demand_idx'] = demand_model.predict(X_demand)
    
    # 6. Display Results
    print("\nSample Predictions (Latest 5 Months):")
    cols_to_show = ['year', 'month', 'local_coffee_price_lkr_per_kg', 'predicted_price_lkr', 'predicted_demand_idx']
    print(df[cols_to_show].tail().to_string(index=False))
    
    print("\nSummary statistics for Predictions:")
    print(f"Mean Predicted Price: {df['predicted_price_lkr'].mean():.2f} LKR/kg")
    print(f"Mean Predicted Demand Index: {df['predicted_demand_idx'].mean():.2f}")

if __name__ == "__main__":
    run_predictions()
