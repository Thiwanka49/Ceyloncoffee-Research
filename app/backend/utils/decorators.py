from functools import wraps
from flask import request, jsonify
from config.firebase import verify_token

def token_required(f):
    """Decorator to verify Firebase token"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        
        # Get token from header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(' ')[1]  # Bearer <token>
            except IndexError:
                return jsonify({'error': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        # Verify token
        decoded_token = verify_token(token)
        if not decoded_token:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Add user info to request
        request.user_id = decoded_token['uid']
        request.user_email = decoded_token.get('email')
        
        return f(*args, **kwargs)
    
    return decorated_function