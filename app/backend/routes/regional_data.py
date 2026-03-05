from flask import Blueprint, request, jsonify
from datetime import datetime

regional_data_bp = Blueprint('regional_data', __name__)

# ============================================================================
# REGIONAL AVERAGES (Based on Sri Lankan Agricultural Data)
# ============================================================================

REGIONAL_AVERAGES = {
    'Kandy': {
        'avg_yield_kg_per_hectare': 85,
        'avg_disease_severity': 0.35,
        'regional_factor': 1.1,
        'quality_factor': 1.15,
        'typical_quality': 'Grade A',
        'description': 'Central Highlands - Premium coffee region'
    },
    'Nuwara Eliya': {
        'avg_yield_kg_per_hectare': 75,
        'avg_disease_severity': 0.28,
        'regional_factor': 1.25,
        'quality_factor': 1.3,
        'typical_quality': 'Grade A',
        'description': 'High elevation - Specialty coffee with unique flavor'
    },
    'Badulla': {
        'avg_yield_kg_per_hectare': 90,
        'avg_disease_severity': 0.32,
        'regional_factor': 1.05,
        'quality_factor': 1.1,
        'typical_quality': 'Grade A',
        'description': 'Uva Province - Known for distinct character'
    },
    'Ratnapura': {
        'avg_yield_kg_per_hectare': 95,
        'avg_disease_severity': 0.42,
        'regional_factor': 0.95,
        'quality_factor': 1.0,
        'typical_quality': 'Grade B',
        'description': 'Sabaragamuwa - Good commercial coffee'
    },
    'Matale': {
        'avg_yield_kg_per_hectare': 88,
        'avg_disease_severity': 0.38,
        'regional_factor': 1.0,
        'quality_factor': 1.05,
        'typical_quality': 'Grade B',
        'description': 'Central Province - Balanced production'
    }
}

@regional_data_bp.route('/averages', methods=['GET'])
def get_regional_averages():
    """Get regional average data"""
    region = request.args.get('region')
    
    if region and region in REGIONAL_AVERAGES:
        return jsonify({
            'success': True,
            'data': REGIONAL_AVERAGES[region]
        }), 200
    
    return jsonify({
        'success': True,
        'data': REGIONAL_AVERAGES
    }), 200


@regional_data_bp.route('/market-advice', methods=['POST'])
def get_market_advice():
    """Generate market advice based on predictions"""
    try:
        data = request.json
        predicted_price = data['predicted_price']
        demand_index = data.get('demand_index', 1.0)
        quality = data['quality_grade']
        
        advice = generate_advice(predicted_price, demand_index, quality)
        
        return jsonify({
            'success': True,
            'data': advice
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def generate_advice(price, demand, quality):
    """Generate simple, actionable advice"""
    
    # Price assessment
    if price > 2500:
        price_outlook = "excellent"
        price_advice = "Prices are very favorable. This is a great time to sell."
    elif price > 2000:
        price_outlook = "good"
        price_advice = "Prices are good. Reasonable profit margins expected."
    elif price > 1500:
        price_outlook = "moderate"
        price_advice = "Prices are moderate. Check your costs carefully."
    else:
        price_outlook = "low"
        price_advice = "Prices are below average. Consider storing if possible."
    
    # Demand assessment
    if demand > 1.2:
        demand_outlook = "very high"
        demand_advice = "Strong buyer interest. Sell quickly before demand drops."
    elif demand > 1.0:
        demand_outlook = "good"
        demand_advice = "Good buyer interest. Negotiate for best price."
    elif demand > 0.8:
        demand_outlook = "moderate"
        demand_advice = "Some buyers looking. May need to wait for better demand."
    else:
        demand_outlook = "low"
        demand_advice = "Limited buyer interest. Consider storing or waiting."
    
    # Overall recommendation
    if price_outlook in ["excellent", "good"] and demand_outlook in ["very high", "good"]:
        recommendation = "SELL NOW - Excellent market conditions!"
        action = "Contact buyers immediately and negotiate premium prices."
    elif price_outlook == "moderate" and demand_outlook in ["good", "very high"]:
        recommendation = "GOOD TIME TO SELL - Take advantage of buyer interest"
        action = "Sell your harvest while demand is strong."
    elif price_outlook in ["low"] or demand_outlook in ["low"]:
        recommendation = "CONSIDER WAITING - Market conditions not ideal"
        action = "If you can store safely, wait for prices to improve. If you must sell, negotiate carefully."
    else:
        recommendation = "MODERATE CONDITIONS - Your decision"
        action = "Evaluate your costs and decide if profit margins are acceptable."
    
    return {
        'price_outlook': price_outlook,
        'demand_outlook': demand_outlook,
        'price_advice': price_advice,
        'demand_advice': demand_advice,
        'recommendation': recommendation,
        'action': action
    }