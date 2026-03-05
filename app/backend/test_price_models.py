"""Test if price/demand models load correctly"""

import os
import joblib

print("\n" + "="*70)
print("🧪 TESTING PRICE & DEMAND MODELS")
print("="*70)

# Check files
print("\n📁 Checking files...")
files = {
    'Price GB': 'models/price_demand/coffee_price_gb_model.pkl',
    'Price RF': 'models/price_demand/coffee_price_rf_model.pkl',
    'Price Ridge': 'models/price_demand/coffee_price_ridge_model.pkl',
    'Price Scaler': 'models/price_demand/price_feature_scaler.pkl',
    'Price Region Enc': 'models/price_demand/price_region_encoder.pkl',
    'Price Quality Enc': 'models/price_demand/price_quality_encoder.pkl',
    'Price Ensemble': 'models/price_demand/price_ensemble_config.pkl',
    'Price Features': 'models/price_demand/price_feature_names.txt',
    
    'Demand GB': 'models/price_demand/coffee_demand_gb_model.pkl',
    'Demand RF': 'models/price_demand/coffee_demand_rf_model.pkl',
    'Demand Ridge': 'models/price_demand/coffee_demand_ridge_model.pkl',
    'Demand Scaler': 'models/price_demand/demand_feature_scaler.pkl',
    'Demand Region Enc': 'models/price_demand/demand_region_encoder.pkl',
    'Demand Quality Enc': 'models/price_demand/demand_quality_encoder.pkl',
    'Demand Ensemble': 'models/price_demand/demand_ensemble_config.pkl',
    'Demand Features': 'models/price_demand/demand_feature_names.txt',
}

for name, path in files.items():
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    if exists:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"{status} {name:20s} | {size_mb:6.2f} MB")
    else:
        print(f"{status} {name:20s} | NOT FOUND")

# Try loading
print("\n🔬 Testing model loading...")
try:
    price_gb = joblib.load('models/price_demand/coffee_price_gb_model.pkl')
    print(f"✅ Price GB loaded: {type(price_gb)}")
except Exception as e:
    print(f"❌ Price GB failed: {e}")

try:
    demand_gb = joblib.load('models/price_demand/coffee_demand_gb_model.pkl')
    print(f"✅ Demand GB loaded: {type(demand_gb)}")
except Exception as e:
    print(f"❌ Demand GB failed: {e}")

try:
    price_scaler = joblib.load('models/price_demand/price_feature_scaler.pkl')
    print(f"✅ Price Scaler loaded: {type(price_scaler)}")
except Exception as e:
    print(f"❌ Price Scaler failed: {e}")

try:
    with open('models/price_demand/price_feature_names.txt', 'r') as f:
        features = [line.strip() for line in f.readlines()]
    print(f"✅ Price Features loaded: {len(features)} features")
except Exception as e:
    print(f"❌ Price Features failed: {e}")

print("\n" + "="*70)
print("✅ TEST COMPLETE")
print("="*70 + "\n")