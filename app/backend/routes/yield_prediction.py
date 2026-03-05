from flask import Blueprint, request, jsonify
from utils.decorators import token_required
from config.firebase import db
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import os
from utils.weather_api import get_historical_weather, get_available_regions as get_regions_data


yield_bp = Blueprint('yield', __name__)

# ============================================================================
# MODEL PATHS
# ============================================================================
MODEL_PATHS = {
    'model': 'models/yield/yield_xgboost_model.pkl',
    'variety_encoder': 'models/yield/variety_encoder.pkl',
    'disease_encoder': 'models/yield/disease_encoder.pkl',
    # 'variety_encoder': 'models/yield/variety_encoder_compatible.pkl',
    # 'disease_encoder': 'models/yield/disease_encoder_compatible.pkl',
    'feature_names': 'models/yield/feature_names.txt'
}

# ============================================================================
# LOAD MODELS
# ============================================================================
models = {}

print("\n" + "="*70)
print("📦 LOADING YIELD PREDICTION MODEL")
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

def prepare_input_data(input_dict):
    """
    Prepare a DataFrame from inputs, compute derived features, and encode categoricals.
    Matches the exact inference logic.
    """
    # Create DataFrame from inputs
    df = pd.DataFrame([input_dict])
    
    # Compute derived features
    df['temp_range'] = df['max_temp'] - df['min_temp']
    df['is_peak_age'] = ((df['plant_age_years'] >= 5) & (df['plant_age_years'] <= 10)).astype(int)
    df['rainfall_per_rainy_day'] = df['total_rainfall_mm'] / (df['rainy_days'] + 1)  # Avoid division by zero
    df['humidity_rainfall'] = df['avg_humidity'] * df['total_rainfall_mm'] / 100
    df['temp_solar'] = df['avg_temp'] * df['avg_solarradiation'] / 100
    
    # Encode categorical features
    variety = df['variety'].iloc[0]
    disease_type = df['disease_type'].iloc[0]
    
    if variety not in models['variety_encoder'].classes_:
        raise ValueError(f"Invalid variety: {variety}. Must be one of {models['variety_encoder'].classes_}")
    if disease_type not in models['disease_encoder'].classes_:
        raise ValueError(f"Invalid disease_type: {disease_type}. Must be one of {models['disease_encoder'].classes_}")
    
    df['variety'] = models['variety_encoder'].transform(df['variety'])
    df['disease_type'] = models['disease_encoder'].transform(df['disease_type'])
    
    # Select only the required features in order
    feature_names = models['feature_names']
    X = df[feature_names]
    
    return X

@yield_bp.route('/regions', methods=['GET'])
def get_available_regions():
    """Get list of available Sri Lankan coffee regions"""
    try:
        data = get_regions_data()
        
        # Reformat to match what frontend expects:
        # frontend does: regions.map(r => ({ label: r.name, value: r.key }))
        regions = [
            {
                'key': r['key'],
                'name': r['name'],
            }
            for r in data['regions']
        ]
        
        return jsonify({
            'success': True,
            'data': regions
        }), 200
        
    except Exception as e:
        print(f"❌ Error getting regions: {str(e)}")
        return jsonify({'error': str(e)}), 500


@yield_bp.route('/weather', methods=['GET'])
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

@yield_bp.route('/predict', methods=['POST'])
@token_required
def predict_yield():
    """Predict coffee yield based on farm and weather data"""
    try:
        # Check if models are loaded
        if not models or 'model' not in models:
            return jsonify({
                'error': 'Yield prediction model not available'
            }), 503
        
        data = request.json
        print(f"\n🌱 Yield prediction request received")
        print(f"Input data: {data}")
        
        # Validate required fields
        required_fields = [
            'plot_area_hectares', 'plants_per_hectare', 'plant_age_years',
            'fertilizer_applied', 'variety', 'avg_temp', 'min_temp', 'max_temp',
            'avg_humidity', 'total_rainfall_mm', 'rainy_days', 'avg_cloudcover',
            'avg_solarradiation', 'avg_uvindex'
        ]
        
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({
                'error': 'Missing required fields',
                'missing': missing
            }), 400
        
        # Prepare input dict (matching inference structure)
        input_dict = {
            'plot_area_hectares': float(data['plot_area_hectares']),
            'plants_per_hectare': int(data['plants_per_hectare']),
            'plant_age_years': int(data['plant_age_years']),
            'fertilizer_applied': int(data['fertilizer_applied']),
            'variety': data['variety'],
            'avg_temp': float(data['avg_temp']),
            'min_temp': float(data['min_temp']),
            'max_temp': float(data['max_temp']),
            'avg_humidity': float(data['avg_humidity']),
            'total_rainfall_mm': float(data['total_rainfall_mm']),
            'rainy_days': int(data['rainy_days']),
            'avg_cloudcover': float(data['avg_cloudcover']),
            'avg_solarradiation': float(data['avg_solarradiation']),
            'avg_uvindex': float(data['avg_uvindex']),
            'disease_type': data.get('disease_type', 'nodisease'),
            'disease_severity': float(data.get('disease_severity', 0.0)),
        }
        
        # Prepare features using inference logic
        X = prepare_input_data(input_dict)
        
        print(f"Feature vector shape: {X.shape}")
        # print(f"Feature values: {X.values[0][:10]}...")  # Print first 10 values
        print(f"Complete feature vector ({len(X.values[0])} values):")
        for i, (name, val) in enumerate(zip(models['feature_names'], X.values[0])):
            print(f"  [{i:2d}] {name:<25} = {val:>10.4f}")
        
        # Make prediction
        predicted_yield = models['model'].predict(X)[0]
        
        print(f"Raw prediction: {predicted_yield:.2f} kg")
        
        # ✅ SAFETY CHECK: Ensure yield is positive and realistic
        MIN_YIELD = 0.0
        MAX_YIELD = 500.0  # Maximum reasonable monthly yield for small plots
        
        if predicted_yield < MIN_YIELD:
            print(f"⚠️ Negative yield detected ({predicted_yield:.2f}), setting to 0")
            predicted_yield = MIN_YIELD
        elif predicted_yield > MAX_YIELD:
            print(f"⚠️ Unrealistic high yield ({predicted_yield:.2f}), capping at {MAX_YIELD}")
            predicted_yield = MAX_YIELD
        
        print(f"✅ Final predicted yield: {predicted_yield:.2f} kg")
        
        # Save prediction to Firestore
        prediction_data = {
            'userId': request.user_id,
            'leafSessionId': data.get('leafSessionId'),
            'inputData': {
                'plot_area_hectares': input_dict['plot_area_hectares'],
                'plants_per_hectare': input_dict['plants_per_hectare'],
                'plant_age_years': input_dict['plant_age_years'],
                'fertilizer_applied': input_dict['fertilizer_applied'],
                'variety': input_dict['variety'],
                'avg_temp': input_dict['avg_temp'],
                'disease_type': input_dict['disease_type'],
                'disease_severity': input_dict['disease_severity']
            },
            'predicted_yield_kg': float(predicted_yield),
            'timestamp': datetime.utcnow()
        }
        
        prediction_ref = db.collection('yield_predictions').add(prediction_data)
        
        return jsonify({
            'success': True,
            'data': {
                'id': prediction_ref[1].id,
                'predicted_yield_kg': float(predicted_yield),
                'disease_type': input_dict['disease_type'],
                'disease_severity': input_dict['disease_severity']
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

@yield_bp.route('/history', methods=['GET'])
@token_required
def get_yield_history():
    """Get yield prediction history for current user"""
    try:
        predictions_ref = db.collection('yield_predictions')
        query = predictions_ref.where('userId', '==', request.user_id)
        docs = query.stream()
        
        predictions = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            if 'timestamp' in data:
                data['timestamp'] = data['timestamp'].isoformat()
            predictions.append(data)
        
        # Sort by timestamp descending (newest first)
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

@yield_bp.route('/health', methods=['GET'])
def health_check():
    """Check yield prediction service health"""
    return jsonify({
        'success': True,
        'model_loaded': 'model' in models,
        'encoders_loaded': 'variety_encoder' in models and 'disease_encoder' in models,
        'features_count': len(models.get('feature_names', []))
    }), 200