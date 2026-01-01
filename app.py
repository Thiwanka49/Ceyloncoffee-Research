import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "C2")
PRICE_MODEL_PATH = os.path.join(MODELS_DIR, "price prediction", "coffee_price_xgboost_model.pkl")
DEMAND_MODEL_PATH = os.path.join(MODELS_DIR, "demand prediction", "coffee_demand_model.pkl")

# Load Models
try:
    price_model = joblib.load(PRICE_MODEL_PATH)
    demand_model = joblib.load(DEMAND_MODEL_PATH)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    price_model = None
    demand_model = None

def get_seasonal_features(month):
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    return month_sin, month_cos

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not price_model or not demand_model:
        return jsonify({"error": "Models not loaded on server."}), 500

    try:
        data = request.json
        year = int(data.get('year', 2025))
        month = int(data.get('month', 1))
        global_price_usd_kg = float(data.get('global_price', 8.0))
        usd_lkr_rate = float(data.get('usd_lkr', 300.0))
        predicted_yield_kg = float(data.get('yield', 600.0))

        # Feature Engineering
        m_sin, m_cos = get_seasonal_features(month)

        # 1. Predict Demand Index First
        # Demand Features: ['local_coffee_price_lkr_per_kg', 'global_price_usd_kg', 'usd_lkr_rate', 'month_sin', 'month_cos']
        # Note: We need a 'local_price' to predict demand index according to the model training.
        # However, the user usually wants to see price based on demand.
        # In the training data, demand index was calculated using historical price.
        # Let's estimate local price for demand prediction based on a simplified formula if not provided,
        # OR just use the global price equivalent as a proxy if appropriate.
        # Actually, let's look at how the generator did it.
        # Generator: price = base_price * supply_factor * demand_index
        # Demand: base + noise (seasonal)
        
        # To keep it simple and robust for the UI:
        # We will use the models as trained. 
        # For the Demand model, we'll use a conservative estimate for 'local_coffee_price_lkr_per_kg'
        # or ask for it. But usually, these factors are interdependent.
        # Let's use (global_price * usd_lkr_rate) as the 'local_price' proxy for demand prediction.
        local_price_proxy = global_price_usd_kg * usd_lkr_rate

        demand_features = np.array([[
            local_price_proxy,
            global_price_usd_kg,
            usd_lkr_rate,
            m_sin,
            m_cos
        ]])
        
        demand_idx = float(demand_model.predict(demand_features)[0])
        
        # Calibrate Demand Index:
        # The raw model output tends to hover around 1.08 for average inputs.
        # We apply an offset to center the baseline around 1.0 (Stable).
        calibration_offset = 0.08
        demand_idx = demand_idx - calibration_offset
        
        # Widen clipping range to allow for more variation
        demand_idx = round(np.clip(demand_idx, 0.5, 1.5), 2)

        # 2. Predict Price
        # Price Features: ['predicted_yield_kg', 'global_price_usd_kg', 'usd_lkr_rate', 'demand_index', 'month_sin', 'month_cos']
        price_features = np.array([[
            predicted_yield_kg,
            global_price_usd_kg,
            usd_lkr_rate,
            demand_idx,
            m_sin,
            m_cos
        ]])
        
        predicted_price = float(price_model.predict(price_features)[0])
        predicted_price = round(predicted_price, 2)

        return jsonify({
            "status": "success",
            "predictions": {
                "demand_index": demand_idx,
                "local_price_lkr": predicted_price
            },
            "inputs": {
                "year": year,
                "month": month,
                "global_price": global_price_usd_kg,
                "usd_lkr": usd_lkr_rate,
                "yield": predicted_yield_kg
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
