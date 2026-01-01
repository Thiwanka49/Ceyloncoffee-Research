import joblib
import numpy as np
import json

def extract_to_json():
    model = joblib.load('C2/demand prediction/coffee_demand_model.pkl')
    r = model.named_steps['ridge']
    s = model.named_steps['scaler']
    f = ['Price_LKR', 'Global_USD', 'USD_LKR', 'Month_Sin', 'Month_Cos']
    
    data = {
        "intercept": float(r.intercept_),
        "coefficients": {feature: float(c) for feature, c in zip(f, r.coef_)},
        "scaler": {
            feature: {"mean": float(m), "scale": float(sc)} 
            for feature, m, sc in zip(f, s.mean_, s.scale_)
        }
    }
    
    with open('params.json', 'w') as f_out:
        json.dump(data, f_out, indent=4)

if __name__ == "__main__":
    extract_to_json()
