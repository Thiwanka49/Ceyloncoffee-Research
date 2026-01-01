import joblib
import numpy as np

def extract_demand():
    model = joblib.load('C2/demand prediction/coffee_demand_model.pkl')
    r = model.named_steps['ridge']
    s = model.named_steps['scaler']
    f = ['Price_LKR', 'Global_USD', 'USD_LKR', 'Month_Sin', 'Month_Cos']
    
    print("--- Demand Model (Ridge) ---")
    print(f"Intercept: {r.intercept_}")
    for feature, coef in zip(f, r.coef_):
        print(f"{feature} Coefficient: {coef}")
    
    print("\n--- Scaler Parameters ---")
    for feature, m, sc in zip(f, s.mean_, s.scale_):
        print(f"{feature}: Mean={m}, Scale={sc}")

if __name__ == "__main__":
    extract_demand()
