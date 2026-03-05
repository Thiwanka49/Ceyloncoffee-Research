from flask import Blueprint, request, jsonify
from utils.decorators import token_required
from config.firebase import db
from utils.market_api import get_usd_lkr_rate, get_global_coffee_price
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import os

price_demand_bp = Blueprint('price_demand', __name__)

# ============================================================================
# MODEL PATHS
# ============================================================================
PRICE_MODEL_PATHS = {
    'gb_model': 'models/price_demand/coffee_price_gb_model.pkl',
    'rf_model': 'models/price_demand/coffee_price_rf_model.pkl',
    'ridge_model': 'models/price_demand/coffee_price_ridge_model.pkl',
    'scaler': 'models/price_demand/price_feature_scaler.pkl',
    'region_encoder': 'models/price_demand/price_region_encoder.pkl',
    'quality_encoder': 'models/price_demand/price_quality_encoder.pkl',
    'source_encoder': 'models/price_demand/price_source_encoder.pkl',
    'ensemble_config': 'models/price_demand/price_ensemble_config.pkl',
    'feature_names': 'models/price_demand/price_feature_names.txt'
}

DEMAND_MODEL_PATHS = {
    'gb_model': 'models/price_demand/coffee_demand_gb_model.pkl',
    'rf_model': 'models/price_demand/coffee_demand_rf_model.pkl',
    'ridge_model': 'models/price_demand/coffee_demand_ridge_model.pkl',
    'scaler': 'models/price_demand/demand_feature_scaler.pkl',
    'region_encoder': 'models/price_demand/demand_region_encoder.pkl',
    'quality_encoder': 'models/price_demand/demand_quality_encoder.pkl',
    'source_encoder': 'models/price_demand/demand_source_encoder.pkl',
    'ensemble_config': 'models/price_demand/demand_ensemble_config.pkl',
    'feature_names': 'models/price_demand/demand_feature_names.txt'
}

# ============================================================================
# LOAD MODELS
# ============================================================================
price_models = {}
demand_models = {}

print("\n" + "="*70)
print("📦 LOADING PRICE & DEMAND PREDICTION MODELS")
print("="*70)

# Load Price Models
print("\n💰 Loading Price Prediction Models...")
try:
    for key, path in PRICE_MODEL_PATHS.items():
        if os.path.exists(path):
            if key == 'feature_names':
                with open(path, 'r') as f:
                    price_models[key] = [line.strip() for line in f.readlines()]
                print(f"✅ {key}: {len(price_models[key])} features")
            else:
                price_models[key] = joblib.load(path)
                print(f"✅ {key} loaded")
        else:
            print(f"❌ {key} not found: {path}")
except Exception as e:
    print(f"❌ Error loading price models: {e}")
    import traceback
    traceback.print_exc()

# Load Demand Models
print("\n📈 Loading Demand Prediction Models...")
try:
    for key, path in DEMAND_MODEL_PATHS.items():
        if os.path.exists(path):
            if key == 'feature_names':
                with open(path, 'r') as f:
                    demand_models[key] = [line.strip() for line in f.readlines()]
                print(f"✅ {key}: {len(demand_models[key])} features")
            else:
                demand_models[key] = joblib.load(path)
                print(f"✅ {key} loaded")
        else:
            print(f"❌ {key} not found: {path}")
except Exception as e:
    print(f"❌ Error loading demand models: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*70)
print("📊 MODEL LOADING SUMMARY")
print("="*70)
print(f"Price Models:  {'✅ Loaded' if len(price_models) > 0 else '❌ Failed'}")
print(f"Demand Models: {'✅ Loaded' if len(demand_models) > 0 else '❌ Failed'}")
print("="*70 + "\n")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_price_features(input_data):
    """
    Create all features required for price prediction
    Based on your training notebook feature engineering
    """
    
    # Calculate temporal features
    month_sin = np.sin(2 * np.pi * input_data['month'] / 12)
    month_cos = np.cos(2 * np.pi * input_data['month'] / 12)
    time_index = (input_data['year'] - 2020) * 12 + input_data['month']
    
    # Ratio features
    price_per_usd_rate = input_data['global_price_usd_kg'] / (input_data['usd_lkr_rate'] / 100)
    yield_to_demand_ratio = input_data['predicted_yield_kg'] / (input_data['demand_index'] + 0.001)
    
    # Normalized features
    global_price_mean = 2.5  # Approximate mean from training data
    exchange_mean = 320.0    # Approximate mean from training data
    
    global_price_normalized = input_data['global_price_usd_kg'] / global_price_mean
    exchange_normalized = input_data['usd_lkr_rate'] / exchange_mean
    
    # Trend features
    price_trend = input_data['global_price_usd_kg'] * input_data['usd_lkr_rate']
    
    # Economic multipliers
    cost_multiplier = (
        input_data['global_price_usd_kg'] *
        input_data['usd_lkr_rate'] *
        input_data['demand_index'] *
        input_data['regional_factor'] *
        input_data['quality_factor']
    )
    
    # Supply pressure
    mean_yield = 800.0  # Approximate mean from training
    supply_pressure = np.clip(mean_yield / (input_data['predicted_yield_kg'] + 1), 0.5, 2.0)
    
    # Interaction features
    exchange_demand = input_data['usd_lkr_rate'] * input_data['demand_index']
    quality_price_interaction = input_data['quality_factor'] * input_data['global_price_usd_kg']
    regional_demand = input_data['regional_factor'] * input_data['demand_index']
    
    # Polynomial features
    global_price_sq = input_data['global_price_usd_kg'] ** 2
    exchange_sq = input_data['usd_lkr_rate'] ** 2
    
    # Log transformations
    log_global_price = np.log1p(input_data['global_price_usd_kg'])
    log_exchange = np.log1p(input_data['usd_lkr_rate'])
    log_yield = np.log1p(input_data['predicted_yield_kg'])
    
    # Lagged features (use provided or defaults)
    prev_month_price = input_data.get('prev_month_price', 2000.0)
    price_change = input_data.get('price_change', 0.0)
    price_change_pct = input_data.get('price_change_pct', 0.0)
    
    # Create feature dictionary matching training order
    features = {
        'year': input_data['year'],
        'month': input_data['month'],
        'month_sin': month_sin,
        'month_cos': month_cos,
        'time_index': time_index,
        'global_price_usd_kg': input_data['global_price_usd_kg'],
        'log_global_price': log_global_price,
        'global_price_normalized': global_price_normalized,
        'global_price_sq': global_price_sq,
        'usd_lkr_rate': input_data['usd_lkr_rate'],
        'log_exchange': log_exchange,
        'exchange_normalized': exchange_normalized,
        'exchange_sq': exchange_sq,
        'predicted_yield_kg': input_data['predicted_yield_kg'],
        'log_yield': log_yield,
        'demand_index': input_data['demand_index'],
        'supply_pressure': supply_pressure,
        'yield_to_demand_ratio': yield_to_demand_ratio,
        'region_encoded': input_data['region_encoded'],
        'quality_encoded': input_data['quality_encoded'],
        'regional_factor': input_data['regional_factor'],
        'quality_factor': input_data['quality_factor'],
        'avg_disease_severity': input_data['avg_disease_severity'],
        'price_volatility_index': input_data.get('price_volatility_index', 0.15),
        'cost_multiplier': cost_multiplier,
        'price_trend': price_trend,
        'exchange_demand': exchange_demand,
        'quality_price_interaction': quality_price_interaction,
        'regional_demand': regional_demand,
        'prev_month_price': prev_month_price,
        'price_change': price_change,
        'price_change_pct': price_change_pct
    }
    
    return features


def create_demand_features(input_data):
    """
    Create all features required for demand prediction
    """
    
    # Calculate temporal features
    month_sin = np.sin(2 * np.pi * input_data['month'] / 12)
    month_cos = np.cos(2 * np.pi * input_data['month'] / 12)
    time_index = (input_data['year'] - 2020) * 12 + input_data['month']
    
    # Ratio features
    yield_to_disease_ratio = input_data['predicted_yield_kg'] / (input_data['avg_disease_severity'] + 0.001)
    
    # Normalized features
    disease_mean = 0.3  # Approximate mean
    disease_normalized = input_data['avg_disease_severity'] / disease_mean
    
    # Interaction features
    region_quality_interaction = input_data['regional_factor'] * input_data['quality_factor']
    disease_quality_interaction = input_data['avg_disease_severity'] * input_data['quality_factor']
    
    # Polynomial features
    disease_sq = input_data['avg_disease_severity'] ** 2
    
    # Log transformations
    log_disease = np.log1p(input_data['avg_disease_severity'])
    log_yield = np.log1p(input_data['predicted_yield_kg'])
    
    features = {
        'year': input_data['year'],
        'month': input_data['month'],
        'month_sin': month_sin,
        'month_cos': month_cos,
        'time_index': time_index,
        'predicted_yield_kg': input_data['predicted_yield_kg'],
        'log_yield': log_yield,
        'avg_disease_severity': input_data['avg_disease_severity'],
        'log_disease': log_disease,
        'disease_normalized': disease_normalized,
        'disease_sq': disease_sq,
        'price_volatility_index': input_data.get('price_volatility_index', 0.15),
        'region_encoded': input_data['region_encoded'],
        'quality_encoded': input_data['quality_encoded'],
        'source_encoded': input_data.get('source_encoded', 0),
        'regional_factor': input_data['regional_factor'],
        'quality_factor': input_data['quality_factor'],
        'yield_to_disease_ratio': yield_to_disease_ratio,
        'region_quality_interaction': region_quality_interaction,
        'disease_quality_interaction': disease_quality_interaction
    }
    
    return features


# ============================================================================
# PRICE PREDICTION ENDPOINT
# ============================================================================

@price_demand_bp.route('/predict-price', methods=['POST'])
@token_required
def predict_price():
    """Predict coffee price for next month"""
    try:
        # Check if models are loaded
        if not price_models or 'gb_model' not in price_models:
            return jsonify({
                'error': 'Price prediction models not available'
            }), 503
        
        data = request.json
        print(f"\n💰 Price prediction request: {data}")
        
        # Encode categorical variables
        region = data['region']
        quality = data['quality_grade']
        
        region_encoded = price_models['region_encoder'].transform([region])[0]
        quality_encoded = price_models['quality_encoder'].transform([quality])[0]
        
        # Prepare input data
        input_data = {
            'year': int(data['year']),
            'month': int(data['month']),
            'global_price_usd_kg': float(data['global_price_usd_kg']),
            'usd_lkr_rate': float(data['usd_lkr_rate']),
            'predicted_yield_kg': float(data['predicted_yield_kg']),
            'demand_index': float(data.get('demand_index', 1.0)),
            'avg_disease_severity': float(data['avg_disease_severity']),
            'regional_factor': float(data.get('regional_factor', 1.0)),
            'quality_factor': float(data.get('quality_factor', 1.0)),
            'region_encoded': region_encoded,
            'quality_encoded': quality_encoded
        }
        
        # Create features
        features = create_price_features(input_data)
        
        # Create DataFrame with correct feature order
        feature_names = price_models['feature_names']
        feature_array = [features[name] for name in feature_names]
        X = pd.DataFrame([feature_array], columns=feature_names)
        
        # Scale features
        X_scaled = price_models['scaler'].transform(X)
        
        # Make predictions with ensemble
        pred_gb = price_models['gb_model'].predict(X_scaled)[0]
        pred_rf = price_models['rf_model'].predict(X_scaled)[0]
        pred_ridge = price_models['ridge_model'].predict(X_scaled)[0]
        
        # Get ensemble weights
        weights = price_models['ensemble_config']['weights']
        predicted_price = (
            weights['gb'] * pred_gb +
            weights['rf'] * pred_rf +
            weights['ridge'] * pred_ridge
        )
        
        print(f"✅ Predicted price: {predicted_price:.2f} LKR/kg")
        
        # Save to Firestore
        prediction_data = {
            'userId': request.user_id,
            'predictionType': 'price',
            'inputData': {
                'year': input_data['year'],
                'month': input_data['month'],
                'region': region,
                'quality_grade': quality,
                'global_price_usd_kg': input_data['global_price_usd_kg'],
                'usd_lkr_rate': input_data['usd_lkr_rate'],
                'predicted_yield_kg': input_data['predicted_yield_kg'],
                'avg_disease_severity': input_data['avg_disease_severity']
            },
            'predicted_price_lkr_per_kg': float(predicted_price),
            'timestamp': datetime.utcnow()
        }
        
        prediction_ref = db.collection('price_predictions').add(prediction_data)
        
        return jsonify({
            'success': True,
            'data': {
                'id': prediction_ref[1].id,
                'predicted_price_lkr_per_kg': float(predicted_price),
                'prediction_month': f"{input_data['year']}-{input_data['month']:02d}"
            }
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# DEMAND PREDICTION ENDPOINT
# ============================================================================

@price_demand_bp.route('/predict-demand', methods=['POST'])
@token_required
def predict_demand():
    """Predict coffee demand index for next month"""
    try:
        # Check if models are loaded
        if not demand_models or 'gb_model' not in demand_models:
            return jsonify({
                'error': 'Demand prediction models not available'
            }), 503
        
        data = request.json
        print(f"\n📈 Demand prediction request: {data}")
        
        # Encode categorical variables
        region = data['region']
        quality = data['quality_grade']
        
        region_encoded = demand_models['region_encoder'].transform([region])[0]
        quality_encoded = demand_models['quality_encoder'].transform([quality])[0]
        
        # Prepare input data
        input_data = {
            'year': int(data['year']),
            'month': int(data['month']),
            'predicted_yield_kg': float(data['predicted_yield_kg']),
            'avg_disease_severity': float(data['avg_disease_severity']),
            'regional_factor': float(data.get('regional_factor', 1.0)),
            'quality_factor': float(data.get('quality_factor', 1.0)),
            'region_encoded': region_encoded,
            'quality_encoded': quality_encoded
        }
        
        # Create features
        features = create_demand_features(input_data)
        
        # Create DataFrame with correct feature order
        feature_names = demand_models['feature_names']
        feature_array = [features[name] for name in feature_names]
        X = pd.DataFrame([feature_array], columns=feature_names)
        
        # Scale features
        X_scaled = demand_models['scaler'].transform(X)
        
        # Make predictions with ensemble
        pred_gb = demand_models['gb_model'].predict(X_scaled)[0]
        pred_rf = demand_models['rf_model'].predict(X_scaled)[0]
        pred_ridge = demand_models['ridge_model'].predict(X_scaled)[0]
        
        # Get ensemble weights
        weights = demand_models['ensemble_config']['weights']
        predicted_demand = (
            weights['gb'] * pred_gb +
            weights['rf'] * pred_rf +
            weights['ridge'] * pred_ridge
        )
        
        print(f"✅ Predicted demand index: {predicted_demand:.4f}")
        
        # Save to Firestore
        prediction_data = {
            'userId': request.user_id,
            'predictionType': 'demand',
            'inputData': {
                'year': input_data['year'],
                'month': input_data['month'],
                'region': region,
                'quality_grade': quality,
                'predicted_yield_kg': input_data['predicted_yield_kg'],
                'avg_disease_severity': input_data['avg_disease_severity']
            },
            'predicted_demand_index': float(predicted_demand),
            'timestamp': datetime.utcnow()
        }
        
        prediction_ref = db.collection('demand_predictions').add(prediction_data)
        
        return jsonify({
            'success': True,
            'data': {
                'id': prediction_ref[1].id,
                'predicted_demand_index': float(predicted_demand),
                'prediction_month': f"{input_data['year']}-{input_data['month']:02d}"
            }
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# HISTORY ENDPOINTS
# ============================================================================

@price_demand_bp.route('/price/history', methods=['GET'])
@token_required
def get_price_history():
    """Get price prediction history"""
    try:
        predictions_ref = db.collection('price_predictions')
        query = predictions_ref.where('userId', '==', request.user_id).order_by('timestamp', direction='DESCENDING')
        docs = query.stream()
        
        predictions = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            if 'timestamp' in data:
                data['timestamp'] = data['timestamp'].isoformat()
            predictions.append(data)
        
        return jsonify({
            'success': True,
            'data': predictions
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@price_demand_bp.route('/demand/history', methods=['GET'])
@token_required
def get_demand_history():
    """Get demand prediction history"""
    try:
        predictions_ref = db.collection('demand_predictions')
        query = predictions_ref.where('userId', '==', request.user_id).order_by('timestamp', direction='DESCENDING')
        docs = query.stream()
        
        predictions = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            if 'timestamp' in data:
                data['timestamp'] = data['timestamp'].isoformat()
            predictions.append(data)
        
        return jsonify({
            'success': True,
            'data': predictions
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# MARKET DATA ENDPOINT
# ============================================================================

@price_demand_bp.route('/market-data', methods=['GET'])
def get_market_data():
    """Get current market data: exchange rate and coffee price"""
    try:
        print("📊 Fetching market data...")
        
        # Fetch exchange rate
        exchange_data = get_usd_lkr_rate()
        
        # Fetch coffee price
        coffee_data = get_global_coffee_price()
        
        if not exchange_data['success']:
            return jsonify({
                'success': False,
                'error': 'Failed to fetch exchange rate'
            }), 500
        
        return jsonify({
            'success': True,
            'data': {
                'usd_lkr_rate': exchange_data['rate'],
                'exchange_rate_date': exchange_data.get('date'),
                'exchange_rate_source': exchange_data.get('source'),
                
                'global_coffee_price_usd_kg': coffee_data.get('price_usd_per_kg') if coffee_data['success'] else None,
                'coffee_price_note': coffee_data.get('note'),
                'coffee_price_source': coffee_data.get('source'),
                'coffee_price_date': coffee_data.get('date'),
            }
        }), 200
        
    except Exception as e:
        print(f"❌ Market data error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# HEALTH CHECK
# ============================================================================

@price_demand_bp.route('/health', methods=['GET'])
def health_check():
    """Check price & demand prediction service health"""
    return jsonify({
        'success': True,
        'models': {
            'price_gb': 'gb_model' in price_models,
            'price_rf': 'rf_model' in price_models,
            'price_ridge': 'ridge_model' in price_models,
            'demand_gb': 'gb_model' in demand_models,
            'demand_rf': 'rf_model' in demand_models,
            'demand_ridge': 'ridge_model' in demand_models
        }
    }), 200