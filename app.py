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

def get_ai_advisory(price, demand_idx, global_price_lkr):
    # Determine Demand Level
    # More sensitive thresholds
    if demand_idx > 1.05:
        demand_lvl = "High"
    elif demand_idx < 0.95:
        demand_lvl = "Low"
    else:
        demand_lvl = "Stable"
    
    # Determine Price Level (relative to global parity)
    # Refined parity check: Local price is usually 90-100% of global parity
    parity_ratio = price / global_price_lkr
    if parity_ratio > 0.98:
        price_lvl = "High"
    elif parity_ratio < 0.92:
        price_lvl = "Low"
    else:
        price_lvl = "Stable"

    advisory = {
        ("High", "High"): {
            "title": "Maximize Export Volume",
            "message": "Market conditions are ideal. Both demand and prices are at peak levels. We recommend fast-tracking all pending shipments and negotiating premium long-term contracts.",
            "action": "Increase Shipments",
            "color": "#4ade80"
        },
        ("High", "Stable"): {
            "title": "Capitalize on Demand",
            "message": "Strong market demand with steady pricing. This is an excellent time to expand your buyer network and push high-volume inventory to clear stocks.",
            "action": "Liquidate Stock",
            "color": "#fbbf24"
        },
        ("High", "Low"): {
            "title": "Focus on Market Share",
            "message": "Demand is high but prices are currently suppressed. Prioritize building relationships and fulfilling urgent orders, but avoid committing your entire stock to low-price contracts.",
            "action": "Selective Shipping",
            "color": "#f87171"
        },
        ("Stable", "High"): {
            "title": "Secure Profit Margins",
            "message": "Prices are favorable despite stable demand. Focus on quality-conscious buyers who are willing to pay the current premium. Ensure efficient logistics to maximize net gains.",
            "action": "Optimize Logistics",
            "color": "#60a5fa"
        },
        ("Stable", "Stable"): {
            "title": "Maintain Steady Supply",
            "message": "Market is in equilibrium. Continue with planned export schedules. Focus on operational efficiency and maintaining consistent quality to keep repeat customers.",
            "action": "Consistent Flow",
            "color": "#94a3b8"
        },
        ("Stable", "Low"): {
            "title": "Defensive Strategy",
            "message": "Stable demand but disappointing prices. Minimize overheads and fulfill only necessary contracts. It may be wise to delay non-urgent shipments until prices recover.",
            "action": "Reduce Overheads",
            "color": "#fca5a5"
        },
        ("Low", "High"): {
            "title": "Prioritize Quality over Volume",
            "message": "Prices are high but demand is sluggish. Target specialty niche markets and premium boutiques that value high-grade Ceylon coffee. Every kilogram sold now yields high returns.",
            "action": "Target Niche",
            "color": "#a78bfa"
        },
        ("Low", "Stable"): {
            "title": "Build Inventory & Brand",
            "message": "Low demand with stable prices. Use this period for processing, advanced grading, and enhancing your brand story. Prepare for the next demand cycle.",
            "action": "Brand Building",
            "color": "#94a3b8"
        },
        ("Low", "Low"): {
            "title": "Hold Stock & Invest in Quality",
            "message": "Both demand and prices are at a seasonal low. We advise holding back large shipments. Focus on improving farm infrastructure and coffee processing techniques for future gains.",
            "action": "Strategic Holding",
            "color": "#ef4444"
        }
    }

    return advisory.get((demand_lvl, price_lvl), advisory[("Stable", "Stable")])

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
        local_price_proxy = global_price_usd_kg * usd_lkr_rate
        
        demand_df = pd.DataFrame([{
            'local_coffee_price_lkr_per_kg': local_price_proxy,
            'global_price_usd_kg': global_price_usd_kg,
            'usd_lkr_rate': usd_lkr_rate,
            'month_sin': m_sin,
            'month_cos': m_cos
        }])
        
        demand_idx = float(demand_model.predict(demand_df)[0])
        
        # Calibrate Demand Index:
        # The raw model output tends to hover around 1.08 for average inputs.
        calibration_offset = 0.08
        demand_idx = demand_idx - calibration_offset
        demand_idx = round(np.clip(demand_idx, 0.5, 1.5), 2)

        # 2. Predict Price
        # Price Features: ['predicted_yield_kg', 'global_price_usd_kg', 'usd_lkr_rate', 'demand_index', 'month_sin', 'month_cos']
        price_df = pd.DataFrame([{
            'predicted_yield_kg': predicted_yield_kg,
            'global_price_usd_kg': global_price_usd_kg,
            'usd_lkr_rate': usd_lkr_rate,
            'demand_index': demand_idx,
            'month_sin': m_sin,
            'month_cos': m_cos
        }])
        
        predicted_price = float(price_model.predict(price_df)[0])
        predicted_price = round(predicted_price, 2)

        # 3. Generate Advisory
        global_price_lkr = global_price_usd_kg * usd_lkr_rate
        advisory = get_ai_advisory(predicted_price, demand_idx, global_price_lkr)

        return jsonify({
            "status": "success",
            "predictions": {
                "demand_index": demand_idx,
                "local_price_lkr": predicted_price
            },
            "advisory": advisory,
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
