from flask import Blueprint, request, jsonify
from utils.decorators import token_required
from utils.weather_api import get_historical_weather, get_available_regions as get_regions_data
from config.firebase import db
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import os

fertilizer_bp = Blueprint('fertilizer', __name__)

# ============================================================================
# MODEL PATHS
# ============================================================================
MODEL_PATHS = {
    'rf_model': 'models/fertilizer/fertilizer_rf_model_per_perch.pkl',
    'gb_model': 'models/fertilizer/fertilizer_gb_model_per_perch.pkl',
    'ensemble_weights': 'models/fertilizer/ensemble_weights.pkl',
    'feature_names': 'models/fertilizer/feature_names.txt'
}

# ============================================================================
# LOAD MODELS
# ============================================================================
models = {}

print("\n" + "="*70)
print("📦 LOADING FERTILIZER PREDICTION MODEL")
print("="*70)

try:
    for key, path in MODEL_PATHS.items():
        if os.path.exists(path):
            if key == 'feature_names':
                with open(path, 'r') as f:
                    models[key] = [line.strip() for line in f.readlines()]
                print(f"✅ {key}: {len(models[key])} features")
            else:
                models[key] = joblib.load(path)
                print(f"✅ {key} loaded")
        else:
            print(f"❌ {key} not found: {path}")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    import traceback
    traceback.print_exc()

print("="*70 + "\n")


# ============================================================================
# HELPER FUNCTION: PREPARE INPUT DATA
# ============================================================================

def prepare_fertilizer_features(input_data):
    """
    Prepare features matching the training notebook's feature engineering.
    """
    
    # Temporal features
    month = input_data['month']
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    is_rainy_season = 1 if month in [5, 6, 7, 8, 9, 10] else 0
    
    # Yield-related features
    yield_per_plant = input_data['predicted_yield_kg_per_perch'] / input_data['plants_per_perch']
    
    # Plant density category
    plants_per_perch = input_data['plants_per_perch']
    if plants_per_perch <= 5:
        plant_density_category = 'low'
    elif plants_per_perch <= 7:
        plant_density_category = 'medium'
    else:
        plant_density_category = 'high'
    
    # Soil balance indicators
    soil_N_P_ratio = input_data['soil_n_mg_kg'] / (input_data['soil_p_mg_kg'] + 1)
    soil_N_K_ratio = input_data['soil_n_mg_kg'] / (input_data['soil_k_mg_kg'] + 1)
    soil_nutrient_sum = (
        input_data['soil_n_mg_kg'] + 
        input_data['soil_p_mg_kg'] + 
        input_data['soil_k_mg_kg']
    )
    
    # Environmental stress indicators
    temp_stress = abs(input_data['avg_temperature_c'] - 23)  # Optimal is ~23°C
    
    # Rainfall category
    rainfall = input_data['rainfall_mm']
    if rainfall <= 100:
        rainfall_category = 'low'
    elif rainfall <= 200:
        rainfall_category = 'moderate'
    else:
        rainfall_category = 'high'
    
    # Create feature dictionary matching training order
    features = {
        'plot_size_perches': input_data['plot_size_perches'],
        'plants_per_perch': input_data['plants_per_perch'],
        'yield_per_plant': yield_per_plant,
        'predicted_yield_kg_per_perch': input_data['predicted_yield_kg_per_perch'],
        'soil_n_mg_kg': input_data['soil_n_mg_kg'],
        'soil_p_mg_kg': input_data['soil_p_mg_kg'],
        'soil_k_mg_kg': input_data['soil_k_mg_kg'],
        'soil_N_P_ratio': soil_N_P_ratio,
        'soil_N_K_ratio': soil_N_K_ratio,
        'soil_nutrient_sum': soil_nutrient_sum,
        'rainfall_mm': input_data['rainfall_mm'],
        'avg_temperature_c': input_data['avg_temperature_c'],
        'temp_stress': temp_stress,
        'month': month,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'is_rainy_season': is_rainy_season,
        'disease_severity': input_data['disease_severity'],
        'growth_stage': input_data['growth_stage'],
        'coffee_variety': input_data['coffee_variety'],
        'plant_density_category': plant_density_category,
        'rainfall_category': rainfall_category
    }
    
    return features


# ============================================================================
# REGIONS ENDPOINT
# ============================================================================

@fertilizer_bp.route('/regions', methods=['GET'])
def get_available_regions():
    """Get list of available Sri Lankan coffee regions"""
    try:
        data = get_regions_data()
        regions = [
            {'key': r['key'], 'name': r['name']}
            for r in data['regions']
        ]
        return jsonify({
            'success': True,
            'data': regions
        }), 200
    except Exception as e:
        print(f"❌ Error getting regions: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# WEATHER ENDPOINT
# ============================================================================

@fertilizer_bp.route('/weather', methods=['GET'])
def get_weather_data():
    """Get historical weather data for a region"""
    try:
        region = request.args.get('region', 'kandy')
        months = int(request.args.get('months', 12))
        
        print(f"\n🌤️ Weather request: region={region}, months={months}")
        result = get_historical_weather(region, months)
        
        if result['success']:
            return jsonify({
                'success': True,
                'data': result['data'],
                'region': result['region'],
                'period': result['period'],
                'source': result['source']
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Failed to fetch weather data')
            }), 500
            
    except Exception as e:
        print(f"❌ Weather API error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# PREDICT ENDPOINT
# ============================================================================

@fertilizer_bp.route('/predict', methods=['POST'])
@token_required
def predict_fertilizer():
    """Predict NPK fertilizer requirements"""
    try:
        # Check if models are loaded
        if not models or 'rf_model' not in models:
            return jsonify({
                'error': 'Fertilizer prediction model not available'
            }), 503
        
        data = request.json
        print(f"\n🌱 Fertilizer prediction request received")
        print(f"Input data: {data}")
        
        # Validate required fields
        required_fields = [
            'plot_size_perches', 'plants_per_perch', 'predicted_yield_kg_per_perch',
            'soil_n_mg_kg', 'soil_p_mg_kg', 'soil_k_mg_kg',
            'rainfall_mm', 'avg_temperature_c', 'month',
            'growth_stage', 'coffee_variety', 'disease_severity'
        ]
        
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({
                'error': 'Missing required fields',
                'missing': missing
            }), 400
        
        # Prepare input dict
        input_data = {
            'plot_size_perches': float(data['plot_size_perches']),
            'plants_per_perch': float(data['plants_per_perch']),
            'predicted_yield_kg_per_perch': float(data['predicted_yield_kg_per_perch']),
            'soil_n_mg_kg': float(data['soil_n_mg_kg']),
            'soil_p_mg_kg': float(data['soil_p_mg_kg']),
            'soil_k_mg_kg': float(data['soil_k_mg_kg']),
            'rainfall_mm': float(data['rainfall_mm']),
            'avg_temperature_c': float(data['avg_temperature_c']),
            'month': int(data['month']),
            'growth_stage': data['growth_stage'],
            'coffee_variety': data['coffee_variety'],
            'disease_severity': float(data['disease_severity']),
        }
        
        # Prepare features
        features = prepare_fertilizer_features(input_data)
        
        # Create DataFrame with correct feature order
        feature_names = models['feature_names']
        feature_array = [features[name] for name in feature_names]
        X = pd.DataFrame([feature_array], columns=feature_names)
        
        print(f"Feature vector shape: {X.shape}")
        
        # Make predictions with ensemble
        pred_rf = models['rf_model'].predict(X)[0]
        pred_gb = models['gb_model'].predict(X)[0]
        
        # Get ensemble weights
        weights = models['ensemble_weights']
        
        # Weighted ensemble for each nutrient (N, P, K)
        predictions = np.zeros(3)
        for i in range(3):
            w_rf, w_gb = weights[i]
            predictions[i] = w_rf * pred_rf[i] + w_gb * pred_gb[i]
        
        # Ensure predictions are positive and within reasonable bounds
        N_kg = max(0.10, min(0.70, predictions[0]))
        P_kg = max(0.03, min(0.30, predictions[1]))
        K_kg = max(0.10, min(0.70, predictions[2]))
        
        print(f"✅ Predicted fertilizer (kg/perch):")
        print(f"   N: {N_kg:.3f}, P: {P_kg:.3f}, K: {K_kg:.3f}")
        
        # Save prediction to Firestore
        prediction_data = {
            'userId': request.user_id,
            'inputData': {
                'plot_size_perches': input_data['plot_size_perches'],
                'plants_per_perch': input_data['plants_per_perch'],
                'predicted_yield_kg_per_perch': input_data['predicted_yield_kg_per_perch'],
                'coffee_variety': input_data['coffee_variety'],
                'growth_stage': input_data['growth_stage'],
                'soil_n_mg_kg': input_data['soil_n_mg_kg'],
                'soil_p_mg_kg': input_data['soil_p_mg_kg'],
                'soil_k_mg_kg': input_data['soil_k_mg_kg'],
            },
            'N_kg_per_perch': float(N_kg),
            'P_kg_per_perch': float(P_kg),
            'K_kg_per_perch': float(K_kg),
            'total_NPK_kg_per_perch': float(N_kg + P_kg + K_kg),
            'timestamp': datetime.utcnow()
        }
        
        prediction_ref = db.collection('fertilizer_predictions').add(prediction_data)
        
        return jsonify({
            'success': True,
            'data': {
                'id': prediction_ref[1].id,
                'N_kg_per_perch': float(N_kg),
                'P_kg_per_perch': float(P_kg),
                'K_kg_per_perch': float(K_kg),
                'total_NPK_kg_per_perch': float(N_kg + P_kg + K_kg)
            }
        }), 200
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500


# ============================================================================
# HISTORY ENDPOINT
# ============================================================================

@fertilizer_bp.route('/history', methods=['GET'])
@token_required
def get_fertilizer_history():
    """Get fertilizer prediction history for current user"""
    try:
        predictions_ref = db.collection('fertilizer_predictions')
        query = predictions_ref.where('userId', '==', request.user_id)
        docs = query.stream()
        
        predictions = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            if 'timestamp' in data:
                data['timestamp'] = data['timestamp'].isoformat()
            predictions.append(data)
        
        # Sort by timestamp descending
        predictions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify({
            'success': True,
            'data': predictions
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# HEALTH CHECK
# ============================================================================

@fertilizer_bp.route('/health', methods=['GET'])
def health_check():
    """Check fertilizer prediction service health"""
    return jsonify({
        'success': True,
        'rf_model_loaded': 'rf_model' in models,
        'gb_model_loaded': 'gb_model' in models,
        'features_count': len(models.get('feature_names', []))
    }), 200