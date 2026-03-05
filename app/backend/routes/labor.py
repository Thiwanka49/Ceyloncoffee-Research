from flask import Blueprint, request, jsonify
from utils.decorators import token_required
from utils.weather_api import get_historical_weather, get_available_regions as get_regions_data
from config.firebase import db
from datetime import datetime, timedelta
import joblib
import pandas as pd
import numpy as np
import os

labor_bp = Blueprint('labor', __name__)

# ============================================================================
# MODEL PATHS
# ============================================================================
MODEL_PATHS = {
    # Labor hours models
    'labor_hours_xgb': 'models/labor/labor_hours_xgb_model.pkl',
    'labor_hours_rf': 'models/labor/labor_hours_rf_model.pkl',
    'labor_hours_gb': 'models/labor/labor_hours_gb_model.pkl',
    
    # Pickers models
    'num_pickers_xgb': 'models/labor/num_pickers_xgb_model.pkl',
    'num_pickers_rf': 'models/labor/num_pickers_rf_model.pkl',
    
    # Total workers models
    'total_workers_xgb': 'models/labor/total_workers_xgb_model.pkl',
    'total_workers_rf': 'models/labor/total_workers_rf_model.pkl',
    
    # Completion days models
    'completion_days_xgb': 'models/labor/completion_days_xgb_model.pkl',
    'completion_days_rf': 'models/labor/completion_days_rf_model.pkl',
    'completion_days_gb': 'models/labor/completion_days_gb_model.pkl',
    
    # Encoders and feature sets
    'label_encoders': 'models/labor/label_encoders.pkl',
    'feature_sets': 'models/labor/feature_sets.pkl',
}

# ============================================================================
# LOAD MODELS
# ============================================================================
models = {}

print("\n" + "="*70)
print("📦 LOADING LABOR PREDICTION MODELS")
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
# CONSTANTS
# ============================================================================

# Labor rates (LKR per day)
LABOR_RATES = {
    "picker": 2000,
    "skilled_picker": 2800,
    "sorter": 1800,
    "supervisor": 3500,
    "loader": 1600
}

# Picking rates (kg per hour)
PICKING_RATES = {
    "arabica": {
        "novice": 3.5,
        "intermediate": 5.5,
        "expert": 8.0
    },
    "robusta": {
        "novice": 4.5,
        "intermediate": 7.0,
        "expert": 10.0
    }
}

# Terrain difficulty factors
TERRAIN_FACTORS = {
    "flat": 1.0,
    "gentle_slope": 0.85,
    "moderate_slope": 0.70,
    "steep_slope": 0.55,
    "terraced": 0.75
}

# Ripeness factors
RIPENESS_FACTORS = {
    "under_ripe": 0.60,
    "optimal": 1.0,
    "over_ripe": 0.70,
    "mixed": 0.80
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def assess_weather_suitability(temp, humidity, precip):
    """
    Determine weather suitability for harvesting
    Returns: (category, score)
    """
    # Rain is the biggest factor
    if precip > 10:
        return "unsuitable", 0.20
    elif precip > 5:
        return "poor", 0.50
    elif precip > 2:
        return "fair", 0.75
    
    # Temperature check (optimal: 20-28°C)
    if temp < 15 or temp > 32:
        return "fair", 0.75
    elif temp < 18 or temp > 30:
        return "good", 0.90
    
    # Humidity check
    if humidity > 85:
        return "fair", 0.75
    elif humidity < 60:
        return "excellent", 1.0
    
    return "excellent", 1.0


def calculate_crew_composition(num_pickers):
    """
    Calculate supporting crew based on number of pickers
    """
    # Supervisors: 1 per 10 pickers
    supervisors = int(np.ceil(num_pickers / 10)) if num_pickers > 5 else 0
    
    # Sorters: 1 per 3-4 pickers
    sorters = int(np.ceil(num_pickers / 3.5))
    
    # Loaders: 1 per 5 pickers
    loaders = max(1, int(np.ceil(num_pickers / 5)))
    
    return {
        'supervisors': supervisors,
        'sorters': sorters,
        'loaders': loaders
    }


def calculate_labor_costs(num_pickers, supervisors, sorters, loaders, days):
    """
    Calculate total labor costs
    """
    # Assume 30% are skilled pickers
    skilled_pickers = int(num_pickers * 0.3)
    unskilled_pickers = num_pickers - skilled_pickers
    
    picker_cost = (
        skilled_pickers * days * LABOR_RATES['skilled_picker'] +
        unskilled_pickers * days * LABOR_RATES['picker']
    )
    
    supervisor_cost = supervisors * days * LABOR_RATES['supervisor']
    sorter_cost = sorters * days * LABOR_RATES['sorter']
    loader_cost = loaders * days * LABOR_RATES['loader']
    
    total_cost = picker_cost + supervisor_cost + sorter_cost + loader_cost
    
    return {
        'picker_cost': round(picker_cost, 2),
        'supervisor_cost': round(supervisor_cost, 2),
        'sorter_cost': round(sorter_cost, 2),
        'loader_cost': round(loader_cost, 2),
        'total_cost': round(total_cost, 2)
    }


def prepare_labor_features(input_data, encoders):
    """
    Prepare features for labor prediction models
    """
    # Get base picking rate
    variety = input_data['coffee_variety'].lower()
    experience = input_data['worker_experience_level']
    base_picking_rate = PICKING_RATES[variety][experience]
    
    # Get factors
    terrain_factor = TERRAIN_FACTORS[input_data['terrain_type']]
    ripeness_factor = RIPENESS_FACTORS[input_data['cherry_ripeness']]
    
    # Assess weather
    weather_suit, weather_score = assess_weather_suitability(
        input_data['avg_temperature_c'],
        input_data['avg_humidity_pct'],
        input_data['avg_precipitation_mm']
    )
    
    # Calculate effective picking rate
    effective_picking_rate = (
        base_picking_rate * 
        terrain_factor * 
        weather_score * 
        ripeness_factor
    )
    
    # Calculate avg yield per tree
    avg_yield_per_tree = input_data['expected_harvest_kg'] / input_data['plot_size_trees']
    
    # Temporal features
    month = input_data['month']
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    is_rainy_season = 1 if month in [5, 6, 7, 8, 9, 10] else 0
    is_main_season = 1 if month in [10, 11, 12, 1, 2, 3] else 0
    
    # Determine season
    season = 'main' if is_main_season else 'off'
    
    # Plant age features
    is_peak_age = 1 if (5 <= input_data['plant_age_years'] <= 10) else 0
    
    # Weather impact features
    temp_humidity_index = input_data['avg_temperature_c'] * input_data['avg_humidity_pct'] / 100
    rain_impact_factor = 0.5 if input_data['avg_precipitation_mm'] > 5 else 1.0
    
    # Harvest complexity
    harvest_complexity = (
        (1 - terrain_factor) * 0.4 +
        (1 - weather_score) * 0.3 +
        (1 - ripeness_factor) * 0.3
    )
    
    # Encode categorical variables
    variety_encoded = encoders['variety'].transform([input_data['coffee_variety']])[0]
    terrain_encoded = encoders['terrain'].transform([input_data['terrain_type']])[0]
    ripeness_encoded = encoders['ripeness'].transform([input_data['cherry_ripeness']])[0]
    experience_encoded = encoders['experience'].transform([input_data['worker_experience_level']])[0]
    
    # For weather_suit_encoded, use 'fair' if the actual value isn't in classes
    if weather_suit in encoders['weather_suit'].classes_:
        weather_suit_encoded = encoders['weather_suit'].transform([weather_suit])[0]
    else:
        weather_suit_encoded = encoders['weather_suit'].transform(['fair'])[0]
    
    season_encoded = encoders['season'].transform([season])[0]
    
    return {
        'expected_harvest_kg': input_data['expected_harvest_kg'],
        'plot_size_trees': input_data['plot_size_trees'],
        'avg_yield_per_tree_kg': avg_yield_per_tree,
        'variety_encoded': variety_encoded,
        'terrain_encoded': terrain_encoded,
        'terrain_factor': terrain_factor,
        'plant_age_years': input_data['plant_age_years'],
        'is_peak_age': is_peak_age,
        'ripeness_encoded': ripeness_encoded,
        'ripeness_factor': ripeness_factor,
        'experience_encoded': experience_encoded,
        'base_picking_rate_kg_per_hour': base_picking_rate,
        'effective_picking_rate_kg_per_hour': effective_picking_rate,
        'avg_temperature_c': input_data['avg_temperature_c'],
        'avg_humidity_pct': input_data['avg_humidity_pct'],
        'avg_precipitation_mm': input_data['avg_precipitation_mm'],
        'weather_suitability_score': weather_score,
        'weather_suit_encoded': weather_suit_encoded,
        'temp_humidity_index': temp_humidity_index,
        'rain_impact_factor': rain_impact_factor,
        'month': month,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'season_encoded': season_encoded,
        'is_main_season': is_main_season,
        'harvest_complexity': harvest_complexity,
        'target_completion_days': input_data['target_completion_days']
    }


# ============================================================================
# REGIONS ENDPOINT
# ============================================================================

@labor_bp.route('/regions', methods=['GET'])
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

@labor_bp.route('/weather', methods=['GET'])
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
# PREDICT ENDPOINT
# ============================================================================

@labor_bp.route('/predict', methods=['POST'])
@token_required
def predict_labor():
    """Predict labor requirements for coffee harvesting"""
    try:
        # Check if models are loaded
        if not models or 'label_encoders' not in models:
            return jsonify({
                'error': 'Labor prediction models not available'
            }), 503
        
        data = request.json
        print(f"\n👥 Labor prediction request received")
        print(f"Input data: {data}")
        
        # Validate required fields
        required_fields = [
            'expected_harvest_kg', 'plot_size_trees', 'coffee_variety',
            'terrain_type', 'plant_age_years', 'cherry_ripeness',
            'worker_experience_level', 'avg_temperature_c', 'avg_humidity_pct',
            'avg_precipitation_mm', 'target_completion_days', 'month'
        ]
        
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({
                'error': 'Missing required fields',
                'missing': missing
            }), 400
        
        # Prepare input
        input_data = {
            'expected_harvest_kg': float(data['expected_harvest_kg']),
            'plot_size_trees': int(data['plot_size_trees']),
            'coffee_variety': data['coffee_variety'],
            'terrain_type': data['terrain_type'],
            'plant_age_years': int(data['plant_age_years']),
            'cherry_ripeness': data['cherry_ripeness'],
            'worker_experience_level': data['worker_experience_level'],
            'avg_temperature_c': float(data['avg_temperature_c']),
            'avg_humidity_pct': float(data['avg_humidity_pct']),
            'avg_precipitation_mm': float(data['avg_precipitation_mm']),
            'target_completion_days': int(data['target_completion_days']),
            'month': int(data['month'])
        }
        
        # Prepare features
        encoders = models['label_encoders']
        feature_sets = models['feature_sets']
        features = prepare_labor_features(input_data, encoders)
        
        # STEP 1: Predict labor hours
        X_hours = pd.DataFrame([features])[feature_sets['labor_hours']]
        
        labor_hours = (
            0.5 * models['labor_hours_xgb'].predict(X_hours)[0] +
            0.3 * models['labor_hours_rf'].predict(X_hours)[0] +
            0.2 * models['labor_hours_gb'].predict(X_hours)[0]
        )
        
        # Calculate worker-days
        worker_days = labor_hours / 7  # 7 hours per working day
        
        # Add to features for next predictions
        features['total_labor_hours'] = labor_hours
        features['total_worker_days'] = worker_days
        
        # STEP 2: Predict number of pickers
        X_pickers = pd.DataFrame([features])[feature_sets['pickers']]
        
        num_pickers = (
            0.6 * models['num_pickers_xgb'].predict(X_pickers)[0] +
            0.4 * models['num_pickers_rf'].predict(X_pickers)[0]
        )
        num_pickers = max(1, int(round(num_pickers)))
        
        # Add to features
        features['num_pickers'] = num_pickers
        
        # STEP 3: Predict total workers
        X_workers = pd.DataFrame([features])[feature_sets['total_workers']]
        
        total_workers = (
            0.6 * models['total_workers_xgb'].predict(X_workers)[0] +
            0.4 * models['total_workers_rf'].predict(X_workers)[0]
        )
        total_workers = max(1, int(round(total_workers)))
        
        # Add to features
        features['total_workers'] = total_workers
        
        # STEP 4: Predict actual completion days
        X_completion = pd.DataFrame([features])[feature_sets['completion']]
        
        completion_days = (
            0.5 * models['completion_days_xgb'].predict(X_completion)[0] +
            0.3 * models['completion_days_rf'].predict(X_completion)[0] +
            0.2 * models['completion_days_gb'].predict(X_completion)[0]
        )
        
        # STEP 5: Calculate crew composition
        crew = calculate_crew_composition(num_pickers)
        
        # STEP 6: Calculate costs
        costs = calculate_labor_costs(
            num_pickers,
            crew['supervisors'],
            crew['sorters'],
            crew['loaders'],
            completion_days
        )
        
        # Calculate cost efficiency
        cost_per_kg = costs['total_cost'] / input_data['expected_harvest_kg']
        
        # Calculate suggested start date
        buffer_days = int(np.ceil(completion_days * 0.2))  # 20% buffer
        days_before_target = int(np.ceil(completion_days)) + buffer_days
        
        print(f"✅ Labor prediction complete:")
        print(f"   Total hours: {labor_hours:.1f}")
        print(f"   Pickers: {num_pickers}")
        print(f"   Total workers: {total_workers}")
        print(f"   Completion: {completion_days:.1f} days")
        print(f"   Total cost: LKR {costs['total_cost']:,.2f}")
        
        # Save to Firestore
        prediction_data = {
            'userId': request.user_id,
            'inputData': {
                'expected_harvest_kg': input_data['expected_harvest_kg'],
                'plot_size_trees': input_data['plot_size_trees'],
                'coffee_variety': input_data['coffee_variety'],
                'terrain_type': input_data['terrain_type'],
                'target_completion_days': input_data['target_completion_days']
            },
            'total_labor_hours': float(labor_hours),
            'total_worker_days': float(worker_days),
            'num_pickers': int(num_pickers),
            'num_supervisors': int(crew['supervisors']),
            'num_sorters': int(crew['sorters']),
            'num_loaders': int(crew['loaders']),
            'total_workers': int(total_workers),
            'actual_completion_days': float(completion_days),
            'total_labor_cost_lkr': float(costs['total_cost']),
            'cost_per_kg_lkr': float(cost_per_kg),
            'timestamp': datetime.utcnow()
        }
        
        prediction_ref = db.collection('labor_predictions').add(prediction_data)
        
        return jsonify({
            'success': True,
            'data': {
                'id': prediction_ref[1].id,
                'total_labor_hours': float(labor_hours),
                'total_worker_days': float(worker_days),
                'num_pickers': int(num_pickers),
                'num_supervisors': int(crew['supervisors']),
                'num_sorters': int(crew['sorters']),
                'num_loaders': int(crew['loaders']),
                'total_workers': int(total_workers),
                'actual_completion_days': float(completion_days),
                'picker_cost_lkr': float(costs['picker_cost']),
                'supervisor_cost_lkr': float(costs['supervisor_cost']),
                'sorter_cost_lkr': float(costs['sorter_cost']),
                'loader_cost_lkr': float(costs['loader_cost']),
                'total_labor_cost_lkr': float(costs['total_cost']),
                'cost_per_kg_lkr': float(cost_per_kg),
                'days_before_target': int(days_before_target)
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

@labor_bp.route('/history', methods=['GET'])
@token_required
def get_labor_history():
    """Get labor prediction history"""
    try:
        predictions_ref = db.collection('labor_predictions')
        query = predictions_ref.where('userId', '==', request.user_id)
        docs = query.stream()
        
        predictions = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            if 'timestamp' in data:
                data['timestamp'] = data['timestamp'].isoformat()
            predictions.append(data)
        
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

@labor_bp.route('/health', methods=['GET'])
def health_check():
    """Check labor prediction service health"""
    return jsonify({
        'success': True,
        'models_loaded': {
            'labor_hours': 'labor_hours_xgb' in models,
            'pickers': 'num_pickers_xgb' in models,
            'workers': 'total_workers_xgb' in models,
            'completion': 'completion_days_xgb' in models,
            'encoders': 'label_encoders' in models
        }
    }), 200