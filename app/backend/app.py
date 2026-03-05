from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv()

# Import routes
from routes.auth import auth_bp
from routes.disease import disease_bp
from routes.yield_prediction import yield_bp
from routes.price_demand import price_demand_bp 
from routes.regional_data import regional_data_bp
from routes.fertilizer import fertilizer_bp
from routes.labor import labor_bp
from routes.pest import pest_bp

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.register_blueprint(regional_data_bp, url_prefix='/api/regional')

# Enable CORS with specific configuration
CORS(app, resources={
    r"/*": {
        "origins": "*",  # Allow all origins in development
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(disease_bp, url_prefix='/api/disease')
app.register_blueprint(yield_bp, url_prefix='/api/yield')
app.register_blueprint(price_demand_bp, url_prefix='/api/price-demand')
app.register_blueprint(fertilizer_bp, url_prefix='/api/fertilizer')
app.register_blueprint(labor_bp, url_prefix='/api/labor')
app.register_blueprint(pest_bp, url_prefix='/api/pest')


# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Ceylon Coffee API is running',
        'timestamp': datetime.utcnow().isoformat()
    }), 200

# Root endpoint
@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'Welcome to Ceylon Coffee API',
        'version': '1.0.0',
        'endpoints': {
            'health': '/health',
            'auth': '/api/auth',
            'disease': '/api/disease',
            'yield': '/api/yield',
            'price_demand': '/api/price-demand',
            'fertilizer': '/api/fertilizer',
            'labor': '/api/labor',
            'pest': '/api/pest'
        }
    }), 200

# Test endpoint (useful for debugging)
@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({
        'success': True,
        'message': 'API is working!',
        'timestamp': datetime.utcnow().isoformat()
    }), 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'details': str(error)}), 500

# ✅ Add startup message with model status
def check_models():
    """Check if all required models are present"""
    models_status = {
        'leaf_disease': os.path.exists('models/disease/coffee_l_disease_classifier.keras'),
        'bean_disease': os.path.exists('models/disease/coffee_beans_classifier.keras'),
        'yield_prediction': os.path.exists('models/yield/yield_xgboost_model.pkl'),
    }
    return models_status

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 CEYLON COFFEE API - Starting Up")
    print("="*60)
    
    # Check models
    print("\n📦 Checking Models...")
    models_status = check_models()
    for model_name, exists in models_status.items():
        status = "✅" if exists else "❌"
        print(f"   {status} {model_name}")
    
    # Start server
    port = int(os.getenv('PORT', 5000))
    print(f"\n🌐 Server Information:")
    print(f"   Local: http://localhost:{port}")
    print(f"   Network: http://0.0.0.0:{port}")
    print(f"\n📚 API Documentation:")
    print(f"   Health: http://localhost:{port}/health")
    print(f"   Root: http://localhost:{port}/")
    print("\n" + "="*60 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=True)