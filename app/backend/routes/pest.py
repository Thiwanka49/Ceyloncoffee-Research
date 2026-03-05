from flask import Blueprint, request, jsonify
from utils.decorators import token_required
from utils.weather_api import get_historical_weather, get_available_regions as get_regions_data
from config.firebase import db
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import os

pest_bp = Blueprint('pest', __name__)

# ============================================================================
# MODEL PATHS
# ============================================================================
MODEL_PATHS = {
    # Fungicide models
    'fungicide_xgb': 'models/pest/fungicide_xgb_model.pkl',
    'fungicide_rf': 'models/pest/fungicide_rf_model.pkl',
    'fungicide_gb': 'models/pest/fungicide_gb_model.pkl',
    
    # Insecticide models
    'insecticide_xgb': 'models/pest/insecticide_xgb_model.pkl',
    'insecticide_rf': 'models/pest/insecticide_rf_model.pkl',
    
    # Miticide models
    'miticide_xgb': 'models/pest/miticide_xgb_model.pkl',
    'miticide_rf': 'models/pest/miticide_rf_model.pkl',
    
    # Herbicide models
    'herbicide_xgb': 'models/pest/herbicide_xgb_model.pkl',
    'herbicide_rf': 'models/pest/herbicide_rf_model.pkl',
    'herbicide_gb': 'models/pest/herbicide_gb_model.pkl',
    
    # Application frequency models
    'applications_xgb': 'models/pest/applications_xgb_model.pkl',
    'applications_rf': 'models/pest/applications_rf_model.pkl',
    
    # Encoders and feature sets
    'label_encoders': 'models/pest/pesticide_label_encoders.pkl',
    'feature_sets': 'models/pest/pesticide_feature_sets.pkl',
}

# ============================================================================
# LOAD MODELS
# ============================================================================
models = {}

print("\n" + "="*70)
print("📦 LOADING PEST MANAGEMENT MODELS")
print("="*70)

try:
    for key, path in MODEL_PATHS.items():
        if os.path.exists(path):
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
# DISEASE TREATMENT PROFILES
# ============================================================================

DISEASE_TREATMENT_PROFILE = {
    "nodisease": {
        "requires_fungicide": False,
        "requires_insecticide": False,
        "requires_miticide": False,
        "treatment_urgency": 0.0,
        "active_ingredient": None
    },
    "rust": {
        "requires_fungicide": True,
        "requires_insecticide": False,
        "requires_miticide": False,
        "treatment_urgency": 0.9,
        "active_ingredient": "copper_based",
        "fungicide_type": "copper_based"
    },
    "miner": {
        "requires_fungicide": False,
        "requires_insecticide": True,
        "requires_miticide": False,
        "treatment_urgency": 0.6,
        "active_ingredient": "neonicotinoid",
        "insecticide_type": "neonicotinoid"
    },
    "phoma": {
        "requires_fungicide": True,
        "requires_insecticide": False,
        "requires_miticide": False,
        "treatment_urgency": 0.7,
        "active_ingredient": "azoxystrobin",
        "fungicide_type": "azoxystrobin"
    },
    "brown_eye_spot": {
        "requires_fungicide": True,
        "requires_insecticide": False,
        "requires_miticide": False,
        "treatment_urgency": 0.75,
        "active_ingredient": "copper_based",
        "fungicide_type": "copper_based"
    },
    "red_spider_mite": {
        "requires_fungicide": False,
        "requires_insecticide": False,
        "requires_miticide": True,
        "treatment_urgency": 0.65,
        "active_ingredient": "abamectin",
        "miticide_type": "abamectin"
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def assess_weather_favorability(humidity, rainfall):
    """Determine if weather favors disease"""
    if humidity > 80 and rainfall > 5:
        return "highly_favorable"
    elif humidity > 70 or rainfall > 2:
        return "favorable"
    else:
        return "unfavorable"


def calculate_spray_volume(plot_area_perches, trees_per_perch, spray_method):
    """Calculate water volume needed for spraying"""
    if spray_method == "knapsack":
        base_volume = 8  # liters per perch
    elif spray_method == "motorized":
        base_volume = 12
    else:  # manual
        base_volume = 6
    
    density_adjustment = 1.0 + ((trees_per_perch - 6) * 0.08)
    volume_per_perch = base_volume * density_adjustment
    total_volume = volume_per_perch * plot_area_perches
    
    return round(total_volume, 2)


def prepare_pest_features(input_data, encoders):
    """Prepare features for pest management predictions"""
    
    # Calculate plot area in perches
    trees_per_perch = 6.0  # Assume average
    plot_area_perches = input_data['plot_size_trees'] / trees_per_perch
    
    # Get disease profile
    disease = input_data['detected_disease']
    profile = DISEASE_TREATMENT_PROFILE[disease]
    
    # Temporal features
    month = input_data['month']
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    is_rainy_season = 1 if month in [5, 6, 7, 8, 9, 10] else 0
    
    # Determine season
    season = 'rainy' if is_rainy_season else 'dry'
    
    # Weather favorability
    weather_fav = assess_weather_favorability(
        input_data['avg_humidity_pct'],
        input_data['avg_precipitation_mm'] / 30  # Convert monthly to daily avg
    )
    
    # ✅ FIX: Check if weather_fav is in encoder classes, use fallback if not
    if weather_fav not in encoders['weather_fav'].classes_:
        print(f"⚠️ Weather favorability '{weather_fav}' not in encoder classes: {list(encoders['weather_fav'].classes_)}")
        print(f"   Using 'favorable' as fallback")
        weather_fav = 'favorable'  # Use a safe default that should exist
    
    # Disease favorable conditions
    disease_favorable = 1 if (input_data['avg_humidity_pct'] > 75 and 
                             input_data['avg_precipitation_mm'] > 60) else 0
    
    # Severity level
    severity_score = input_data['disease_severity_score']
    if severity_score <= 0.2:
        severity_level = 'minimal'
    elif severity_score <= 0.4:
        severity_level = 'low'
    elif severity_score <= 0.6:
        severity_level = 'moderate'
    elif severity_score <= 0.8:
        severity_level = 'high'
    else:
        severity_level = 'severe'
    
    # ✅ FIX: Check severity_level too
    if severity_level not in encoders['severity_level'].classes_:
        print(f"⚠️ Severity level '{severity_level}' not in encoder classes: {list(encoders['severity_level'].classes_)}")
        severity_level = 'moderate'  # Use safe default
    
    # Plant age features
    is_peak_age = 1 if (5 <= input_data['plant_age_years'] <= 12) else 0
    is_young_plantation = 1 if input_data['plant_age_years'] < 4 else 0
    
    # Weather stress indices
    heat_stress_index = max(0, input_data['avg_temperature_c'] - 28) / 5 if input_data['avg_temperature_c'] > 28 else 0
    moisture_stress_index = max(0, 80 - input_data['avg_humidity_pct']) / 20 if input_data['avg_humidity_pct'] < 80 else 0
    
    # Treatment intensity (will be calculated after predicting applications)
    # Placeholder for now
    treatment_intensity = severity_score * 2  # Will be updated
    
    # Weed management urgency
    weed_management_urgency = (
        input_data['weed_density_index'] * 
        (input_data['days_since_last_weeding'] / 90)
    )
    
    # Encode categorical variables with safety checks
    variety_encoded = encoders['variety'].transform([input_data['coffee_variety']])[0]
    disease_encoded = encoders['disease'].transform([disease])[0]
    season_encoded = encoders['season'].transform([season])[0]
    weather_fav_encoded = encoders['weather_fav'].transform([weather_fav])[0]
    spray_encoded = encoders['spray'].transform([input_data['spray_method']])[0]
    severity_level_encoded = encoders['severity_level'].transform([severity_level])[0]
    
    return {
        'plot_area_perches': plot_area_perches,
        'trees_per_perch': trees_per_perch,
        'disease_encoded': disease_encoded,
        'disease_severity_score': severity_score,
        'detection_confidence': input_data['detection_confidence'],
        'treatment_urgency': profile['treatment_urgency'],
        'severity_level_encoded': severity_level_encoded,
        'variety_encoded': variety_encoded,
        'plant_age_years': input_data['plant_age_years'],
        'is_peak_age': is_peak_age,
        'is_young_plantation': is_young_plantation,
        'avg_temperature_c': input_data['avg_temperature_c'],
        'avg_humidity_pct': input_data['avg_humidity_pct'],
        'daily_rainfall_mm': input_data['avg_precipitation_mm'] / 30,  # Monthly to daily avg
        'monthly_rainfall_mm': input_data['avg_precipitation_mm'],
        'weather_fav_encoded': weather_fav_encoded,
        'disease_favorable_conditions': disease_favorable,
        'heat_stress_index': heat_stress_index,
        'moisture_stress_index': moisture_stress_index,
        'pest_pressure_index': input_data['pest_pressure_index'],
        'weed_density_index': input_data['weed_density_index'],
        'days_since_last_weeding': input_data['days_since_last_weeding'],
        'weed_management_urgency': weed_management_urgency,
        'month': month,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'season_encoded': season_encoded,
        'is_rainy_season': is_rainy_season,
        'spray_encoded': spray_encoded,
        'requires_fungicide': 1 if profile['requires_fungicide'] else 0,
        'requires_insecticide': 1 if profile['requires_insecticide'] else 0,
        'requires_miticide': 1 if profile['requires_miticide'] else 0,
        'requires_herbicide': 1 if input_data['weed_density_index'] > 0.3 else 0,
        'herbicide_applications': 1 if input_data['weed_density_index'] > 0.3 else 0,
        'treatment_intensity': treatment_intensity,
        'num_applications_needed': 2  # Placeholder, will be predicted
    }, profile


# ============================================================================
# REGIONS ENDPOINT
# ============================================================================

@pest_bp.route('/regions', methods=['GET'])
def get_available_regions():
    """Get list of available regions"""
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

@pest_bp.route('/weather', methods=['GET'])
def get_weather_data():
    """Get historical weather data"""
    try:
        region = request.args.get('region', 'kandy')
        months = int(request.args.get('months', 12))
        
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
        return jsonify({'error': str(e)}), 500


# ============================================================================
# RECOMMEND ENDPOINT
# ============================================================================

@pest_bp.route('/recommend', methods=['POST'])
@token_required
def get_pest_recommendations():
    """Get pesticide and herbicide recommendations"""
    try:
        # Check if models are loaded
        if not models or 'label_encoders' not in models:
            return jsonify({
                'error': 'Pest management models not available'
            }), 503
        
        data = request.json
        print(f"\n🦠 Pest management request received")
        print(f"Input data: {data}")
        
        # Validate required fields
        required_fields = [
            'plot_size_trees', 'coffee_variety', 'plant_age_years',
            'detected_disease', 'disease_severity_score', 'detection_confidence',
            'avg_temperature_c', 'avg_humidity_pct', 'avg_precipitation_mm',
            'pest_pressure_index', 'weed_density_index', 'days_since_last_weeding',
            'spray_method', 'month'
        ]
        
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({
                'error': 'Missing required fields',
                'missing': missing
            }), 400
        
        # Prepare input
        input_data = {
            'plot_size_trees': int(data['plot_size_trees']),
            'coffee_variety': data['coffee_variety'],
            'plant_age_years': int(data['plant_age_years']),
            'detected_disease': data['detected_disease'],
            'disease_severity_score': float(data['disease_severity_score']),
            'detection_confidence': float(data['detection_confidence']),
            'avg_temperature_c': float(data['avg_temperature_c']),
            'avg_humidity_pct': float(data['avg_humidity_pct']),
            'avg_precipitation_mm': float(data['avg_precipitation_mm']),
            'pest_pressure_index': float(data['pest_pressure_index']),
            'weed_density_index': float(data['weed_density_index']),
            'days_since_last_weeding': int(data['days_since_last_weeding']),
            'spray_method': data['spray_method'],
            'month': int(data['month'])
        }
        
        # Prepare features
        encoders = models['label_encoders']
        feature_sets = models['feature_sets']
        features, profile = prepare_pest_features(input_data, encoders)
        
        # STEP 1: Predict number of applications
        X_app = pd.DataFrame([features])[feature_sets['applications']]
        
        num_applications = (
            0.6 * models['applications_xgb'].predict(X_app)[0] +
            0.4 * models['applications_rf'].predict(X_app)[0]
        )
        num_applications = max(1, int(round(num_applications)))
        
        # Update features with predicted applications
        features['num_applications_needed'] = num_applications
        features['treatment_intensity'] = num_applications * features['disease_severity_score']
        
        # STEP 2: Predict fungicide amount
        X_fung = pd.DataFrame([features])[feature_sets['fungicide']]
        
        total_fungicide = 0
        fungicide_type = 'none'
        if profile['requires_fungicide']:
            total_fungicide = (
                0.5 * models['fungicide_xgb'].predict(X_fung)[0] +
                0.3 * models['fungicide_rf'].predict(X_fung)[0] +
                0.2 * models['fungicide_gb'].predict(X_fung)[0]
            )
            total_fungicide = max(0, total_fungicide)
            fungicide_type = profile.get('fungicide_type', 'copper_based')
        
        # STEP 3: Predict insecticide amount
        X_insect = pd.DataFrame([features])[feature_sets['insecticide']]
        
        total_insecticide = 0
        insecticide_type = 'none'
        if profile['requires_insecticide']:
            total_insecticide = (
                0.6 * models['insecticide_xgb'].predict(X_insect)[0] +
                0.4 * models['insecticide_rf'].predict(X_insect)[0]
            )
            total_insecticide = max(0, total_insecticide)
            insecticide_type = profile.get('insecticide_type', 'neonicotinoid')
        
        # STEP 4: Predict miticide amount
        X_mite = pd.DataFrame([features])[feature_sets['miticide']]
        
        total_miticide = 0
        miticide_type = 'none'
        if profile['requires_miticide']:
            total_miticide = (
                0.6 * models['miticide_xgb'].predict(X_mite)[0] +
                0.4 * models['miticide_rf'].predict(X_mite)[0]
            )
            total_miticide = max(0, total_miticide)
            miticide_type = profile.get('miticide_type', 'abamectin')
        
        # STEP 5: Predict herbicide amount
        X_herb = pd.DataFrame([features])[feature_sets['herbicide']]
        
        total_herbicide = (
            0.5 * models['herbicide_xgb'].predict(X_herb)[0] +
            0.3 * models['herbicide_rf'].predict(X_herb)[0] +
            0.2 * models['herbicide_gb'].predict(X_herb)[0]
        )
        total_herbicide = max(0, total_herbicide)
        herbicide_type = 'glyphosate' if total_herbicide > 0 else 'none'
        
        # STEP 6: Calculate spray volume
        spray_volume = calculate_spray_volume(
            features['plot_area_perches'],
            features['trees_per_perch'],
            input_data['spray_method']
        )
        
        print(f"✅ Recommendations generated:")
        print(f"   Fungicide: {total_fungicide:.2f} kg/L")
        print(f"   Insecticide: {total_insecticide:.2f} L")
        print(f"   Miticide: {total_miticide:.2f} L")
        print(f"   Herbicide: {total_herbicide:.2f} L")
        print(f"   Applications: {num_applications}")
        
        # Save to Firestore
        recommendation_data = {
            'userId': request.user_id,
            'inputData': {
                'plot_size_trees': input_data['plot_size_trees'],
                'detected_disease': input_data['detected_disease'],
                'disease_severity_score': input_data['disease_severity_score'],
                'coffee_variety': input_data['coffee_variety']
            },
            'total_fungicide_kg_L': float(total_fungicide),
            'fungicide_type': fungicide_type,
            'total_insecticide_L': float(total_insecticide),
            'insecticide_type': insecticide_type,
            'total_miticide_L': float(total_miticide),
            'miticide_type': miticide_type,
            'total_herbicide_L': float(total_herbicide),
            'herbicide_type': herbicide_type,
            'num_applications': int(num_applications),
            'spray_volume_liters': float(spray_volume),
            'treatment_urgency': float(profile['treatment_urgency']),
            'timestamp': datetime.utcnow()
        }
        
        recommendation_ref = db.collection('pest_recommendations').add(recommendation_data)
        
        return jsonify({
            'success': True,
            'data': {
                'id': recommendation_ref[1].id,
                'total_fungicide_kg_L': float(total_fungicide),
                'fungicide_type': fungicide_type,
                'total_insecticide_L': float(total_insecticide),
                'insecticide_type': insecticide_type,
                'total_miticide_L': float(total_miticide),
                'miticide_type': miticide_type,
                'total_herbicide_L': float(total_herbicide),
                'herbicide_type': herbicide_type,
                'num_applications': int(num_applications),
                'spray_volume_liters': float(spray_volume),
                'treatment_urgency': float(profile['treatment_urgency'])
            }
        }), 200
        
    except Exception as e:
        print(f"❌ Recommendation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Recommendation failed',
            'details': str(e)
        }), 500


# ============================================================================
# HISTORY ENDPOINT
# ============================================================================

@pest_bp.route('/history', methods=['GET'])
@token_required
def get_pest_history():
    """Get pest management recommendation history"""
    try:
        recommendations_ref = db.collection('pest_recommendations')
        query = recommendations_ref.where('userId', '==', request.user_id)
        docs = query.stream()
        
        recommendations = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            if 'timestamp' in data:
                data['timestamp'] = data['timestamp'].isoformat()
            recommendations.append(data)
        
        recommendations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify({
            'success': True,
            'data': recommendations
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# HEALTH CHECK
# ============================================================================

@pest_bp.route('/health', methods=['GET'])
def health_check():
    """Check pest management service health"""
    return jsonify({
        'success': True,
        'models_loaded': {
            'fungicide': 'fungicide_xgb' in models,
            'insecticide': 'insecticide_xgb' in models,
            'miticide': 'miticide_xgb' in models,
            'herbicide': 'herbicide_xgb' in models,
            'applications': 'applications_xgb' in models,
            'encoders': 'label_encoders' in models
        }
    }), 200