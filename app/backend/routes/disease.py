from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import json
from PIL import Image
import numpy as np
from utils.decorators import token_required
from config.firebase import db
from datetime import datetime
import cv2
import tempfile

disease_bp = Blueprint('disease', __name__)

# ============================================================================
# MODEL PATHS
# ============================================================================
MODEL_PATHS = {
    'leaf': 'models/disease/coffee_l_disease_classifier.keras',
    'bean': 'models/disease/coffee_beans_classifier.keras'
}

CLASS_INDICES_PATHS = {
    'leaf': 'models/disease/class_indices_coffee_l_disease.json',
    'bean': 'models/disease/class_indices_coffee_beans.json'
}

# ============================================================================
# LOAD MODELS WITH DETAILED LOGGING
# ============================================================================
leaf_model = None
bean_model = None
leaf_classes = None
bean_classes = None

print("\n" + "="*70)
print("📦 LOADING DISEASE DETECTION MODELS")
print("="*70)

# Check TensorFlow
try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} available")
except ImportError:
    print("❌ TensorFlow not installed!")

# Load Leaf Disease Model
print("\n🍃 Loading Leaf Disease Model...")
try:
    from tensorflow.keras.models import load_model
    
    if os.path.exists(MODEL_PATHS['leaf']):
        leaf_model = load_model(MODEL_PATHS['leaf'])
        print(f"✅ Leaf model loaded successfully")
        
        # Test prediction
        dummy_img = np.random.rand(1, 128, 128, 3).astype(np.float32)
        test_pred = leaf_model.predict(dummy_img, verbose=0)
        print(f"✅ Test prediction successful")
        
    if os.path.exists(CLASS_INDICES_PATHS['leaf']):
        with open(CLASS_INDICES_PATHS['leaf'], 'r') as f:
            leaf_classes = json.load(f)
        print(f"✅ Leaf classes loaded: {list(leaf_classes.keys())}")
        
except Exception as e:
    print(f"❌ Error loading leaf model: {e}")

# Load Bean Disease Model
print("\n☕ Loading Bean Disease Model...")
try:
    if os.path.exists(MODEL_PATHS['bean']):
        bean_model = load_model(MODEL_PATHS['bean'])
        print(f"✅ Bean model loaded successfully")
        
        # Test prediction
        dummy_img = np.random.rand(1, 128, 128, 3).astype(np.float32)
        test_pred = bean_model.predict(dummy_img, verbose=0)
        print(f"✅ Test prediction successful")
        
    if os.path.exists(CLASS_INDICES_PATHS['bean']):
        with open(CLASS_INDICES_PATHS['bean'], 'r') as f:
            bean_classes = json.load(f)
        print(f"✅ Bean classes loaded: {list(bean_classes.keys())}")
        
except Exception as e:
    print(f"❌ Error loading bean model: {e}")

# Summary
print("\n" + "="*70)
print("📊 MODEL LOADING SUMMARY")
print("="*70)
print(f"Leaf Model:   {'✅ Loaded' if leaf_model is not None else '❌ Failed'}")
print(f"Bean Model:   {'✅ Loaded' if bean_model is not None else '❌ Failed'}")
print(f"Leaf Classes: {'✅ Loaded' if leaf_classes is not None else '❌ Failed'}")
print(f"Bean Classes: {'✅ Loaded' if bean_classes is not None else '❌ Failed'}")
print("="*70 + "\n")

# ============================================================================
# DISEASE SEVERITY CONSTANTS
# ============================================================================
DISEASE_BASE_SEVERITY = {
    "nodisease": 0.0,
    "rust": 0.85,
    "miner": 0.55,
    "phoma": 0.65,
    "brown_eye_spot": 0.70,
    "red_spider_mite": 0.60
}

DISEASE_COLOR_RANGES = {
    "rust": {
        "lower": np.array([5, 100, 100]),
        "upper": np.array([20, 255, 255])
    },
    "brown_eye_spot": {
        "lower": np.array([10, 50, 20]),
        "upper": np.array([25, 200, 100])
    },
    "red_spider_mite": {
        "lower": np.array([0, 100, 100]),
        "upper": np.array([10, 255, 200])
    }
}

def compute_severity(disease, confidence, image_path):
    """Calculate disease severity"""
    base = DISEASE_BASE_SEVERITY.get(disease, 0.0)
    
    # CNN severity
    if confidence < 0.5:
        confidence_factor = confidence * 0.5
    else:
        confidence_factor = confidence
    
    cnn_score = base * confidence_factor
    
    # Area-based severity for applicable diseases
    if disease in DISEASE_COLOR_RANGES:
        try:
            img = cv2.imread(image_path)
            if img is not None:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                color_range = DISEASE_COLOR_RANGES[disease]
                mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])
                
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                area_ratio = np.count_nonzero(mask) / mask.size
                
                # Non-linear scaling
                if area_ratio < 0.1:
                    area_severity = area_ratio * 2
                elif area_ratio < 0.3:
                    area_severity = 0.2 + (area_ratio - 0.1) * 2.5
                else:
                    area_severity = 0.7 + (area_ratio - 0.3) * 0.43
                
                area_severity = min(area_severity, 1.0)
                
                # Combined severity: 60% CNN, 40% area
                final_score = 0.6 * cnn_score + 0.4 * area_severity
            else:
                final_score = cnn_score
        except Exception as e:
            print(f"⚠️ Area calculation error: {e}")
            final_score = cnn_score
    else:
        final_score = cnn_score
    
    # Severity level
    if final_score < 0.2:
        severity_level = "Minimal"
    elif final_score < 0.4:
        severity_level = "Low"
    elif final_score < 0.6:
        severity_level = "Moderate"
    elif final_score < 0.8:
        severity_level = "High"
    else:
        severity_level = "Severe"
    
    return {
        'severity_score': round(final_score, 3),
        'severity_level': severity_level
    }


# ============================================================================
# BEAN DISEASE DETECTION (Individual) - NO STORAGE
# ============================================================================

@disease_bp.route('/bean/detect', methods=['POST'])
@token_required
def detect_bean_disease():
    """Detect bean disease from uploaded image"""
    try:
        if bean_model is None:
            return jsonify({
                'error': 'Bean disease model is not available. Please contact administrator.'
            }), 503
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        print(f"📸 Received image: {file.filename}")
        
        # Save temporarily ONLY for processing
        temp_path = tempfile.mktemp(suffix='.jpg')
        file.save(temp_path)
        
        try:
            # Preprocess
            print("🔄 Preprocessing image...")
            img = Image.open(temp_path)
            img = img.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            print("🔮 Running prediction...")
            predictions = bean_model.predict(img_array, verbose=0)
            class_idx = np.argmax(predictions)
            confidence = float(np.max(predictions))
            
            idx_to_class = {v: k for k, v in bean_classes.items()}
            disease_name = idx_to_class[class_idx]
            
            print(f"✅ Detected: {disease_name} ({confidence:.2%})")
            
        finally:
            # ✅ Always cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print("🗑️ Temp file cleaned up")
        
        # ✅ Save only detection results to Firestore (no image)
        detection_data = {
            'userId': request.user_id,
            'disease': disease_name,
            'confidence': confidence,
            'timestamp': datetime.utcnow(),
            'type': 'bean'
        }
        
        detection_ref = db.collection('bean_detections').add(detection_data)
        
        print(f"✅ Bean detection complete!")
        
        return jsonify({
            'success': True,
            'data': {
                'id': detection_ref[1].id,
                'disease': disease_name,
                'confidence': confidence
            }
        }), 200
        
    except Exception as e:
        print(f"❌ Bean detection error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500


@disease_bp.route('/bean/history', methods=['GET'])
@token_required
def get_bean_history():
    """Get bean detection history"""
    try:
        detections_ref = db.collection('bean_detections')
        # ✅ Remove order_by
        query = detections_ref.where('userId', '==', request.user_id)
        docs = query.stream()
        
        detections = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            if 'timestamp' in data:
                data['timestamp'] = data['timestamp'].isoformat()
            detections.append(data)
        
        # ✅ Sort in Python
        detections.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify({
            'success': True,
            'data': detections
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# LEAF DISEASE DETECTION (Estate-wide Sessions) - NO STORAGE
# ============================================================================

@disease_bp.route('/leaf/session/start', methods=['POST'])
@token_required
def start_leaf_session():
    """Start a new leaf disease collection session"""
    try:
        data = request.json or {}
        
        session_data = {
            'userId': request.user_id,
            'sessionDate': datetime.utcnow(),
            'farmId': data.get('farmId', ''),
            'samples': [],
            'estateSeverity': None,
            'status': 'in_progress',
            'createdAt': datetime.utcnow()
        }
        
        session_ref = db.collection('leaf_sessions').add(session_data)
        
        print(f"✅ Started new session: {session_ref[1].id}")
        
        return jsonify({
            'success': True,
            'data': {
                'sessionId': session_ref[1].id
            }
        }), 201
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@disease_bp.route('/leaf/session/<session_id>/add-sample', methods=['POST'])
@token_required
def add_leaf_sample(session_id):
    """Add a leaf sample to ongoing session - NO IMAGE STORAGE"""
    try:
        if leaf_model is None:
            return jsonify({
                'error': 'Leaf disease model is not available.'
            }), 503
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        print(f"📸 Received leaf sample: {file.filename}")
        
        # Save temporarily ONLY for processing
        temp_path = tempfile.mktemp(suffix='.jpg')
        file.save(temp_path)
        
        try:
            # Preprocess
            print("🔄 Preprocessing image...")
            img = Image.open(temp_path)
            img = img.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            print("🔮 Running prediction...")
            predictions = leaf_model.predict(img_array, verbose=0)
            class_idx = np.argmax(predictions)
            confidence = float(np.max(predictions))
            
            idx_to_class = {v: k for k, v in leaf_classes.items()}
            disease_name = idx_to_class[class_idx]
            
            print(f"✅ Detected: {disease_name} ({confidence:.2%})")
            
            # Calculate severity
            severity = compute_severity(disease_name, confidence, temp_path)
            print(f"📊 Severity: {severity['severity_score']} ({severity['severity_level']})")
            
        finally:
            # ✅ Always cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print("🗑️ Temp file cleaned up")
        
        # ✅ Create sample data WITHOUT image
        sample_data = {
            'disease': disease_name,
            'confidence': confidence,
            'severity_score': severity['severity_score'],
            'severity_level': severity['severity_level'],
            'timestamp': datetime.utcnow()
        }
        
        # Update session
        session_ref = db.collection('leaf_sessions').document(session_id)
        session_doc = session_ref.get()
        
        if not session_doc.exists:
            return jsonify({'error': 'Session not found'}), 404
        
        session_data = session_doc.to_dict()
        
        # Check ownership
        if session_data['userId'] != request.user_id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Add sample
        samples = session_data.get('samples', [])
        samples.append(sample_data)
        
        session_ref.update({'samples': samples})
        
        print(f"✅ Added sample to session {session_id}: {disease_name}")
        
        return jsonify({
            'success': True,
            'data': {
                'sample': sample_data,
                'totalSamples': len(samples)
            }
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@disease_bp.route('/leaf/session/<session_id>/complete', methods=['POST'])
@token_required
def complete_leaf_session(session_id):
    """Complete session and calculate estate severity"""
    try:
        session_ref = db.collection('leaf_sessions').document(session_id)
        session_doc = session_ref.get()
        
        if not session_doc.exists:
            return jsonify({'error': 'Session not found'}), 404
        
        session_data = session_doc.to_dict()
        
        if session_data['userId'] != request.user_id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        samples = session_data.get('samples', [])
        
        if len(samples) == 0:
            return jsonify({'error': 'No samples in session'}), 400
        
        # Calculate estate severity
        severity_scores = [s['severity_score'] for s in samples]
        diseases = [s['disease'] for s in samples]
        
        avg_severity = float(np.mean(severity_scores))
        max_severity = float(np.max(severity_scores))
        
        # Disease distribution
        disease_counts = {}
        for disease in diseases:
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
        
        dominant_disease = max(disease_counts, key=disease_counts.get)
        
        estate_severity = {
            'average_severity': round(avg_severity, 3),
            'max_severity': round(max_severity, 3),
            'dominant_disease': dominant_disease,
            'disease_distribution': disease_counts,
            'total_samples': len(samples)
        }
        
        # Update session
        session_ref.update({
            'estateSeverity': estate_severity,
            'status': 'completed',
            'completedAt': datetime.utcnow()
        })
        
        print(f"✅ Completed session {session_id}: {len(samples)} samples")
        
        return jsonify({
            'success': True,
            'data': {
                'sessionId': session_id,
                'estateSeverity': estate_severity
            }
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@disease_bp.route('/leaf/sessions', methods=['GET'])
@token_required
def get_leaf_sessions():
    """Get all leaf disease sessions"""
    try:
        sessions_ref = db.collection('leaf_sessions')
        # ✅ Remove order_by to avoid needing index
        query = sessions_ref.where('userId', '==', request.user_id)
        docs = query.stream()
        
        sessions = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            
            # Convert timestamps
            for field in ['sessionDate', 'createdAt', 'completedAt']:
                if field in data and data[field]:
                    data[field] = data[field].isoformat()
            
            # Convert sample timestamps
            if 'samples' in data:
                for sample in data['samples']:
                    if 'timestamp' in sample:
                        sample['timestamp'] = sample['timestamp'].isoformat()
            
            sessions.append(data)
        
        # ✅ Sort in Python instead of Firestore
        sessions.sort(key=lambda x: x.get('sessionDate', ''), reverse=True)
        
        return jsonify({
            'success': True,
            'data': sessions
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@disease_bp.route('/leaf/session/<session_id>', methods=['GET'])
@token_required
def get_leaf_session(session_id):
    """Get specific session details"""
    try:
        session_ref = db.collection('leaf_sessions').document(session_id)
        session_doc = session_ref.get()
        
        if not session_doc.exists:
            return jsonify({'error': 'Session not found'}), 404
        
        data = session_doc.to_dict()
        data['id'] = session_doc.id
        
        # Convert timestamps
        for field in ['sessionDate', 'createdAt', 'completedAt']:
            if field in data and data[field]:
                data[field] = data[field].isoformat()
        
        if 'samples' in data:
            for sample in data['samples']:
                if 'timestamp' in sample:
                    sample['timestamp'] = sample['timestamp'].isoformat()
        
        return jsonify({
            'success': True,
            'data': data
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@disease_bp.route('/leaf/session/<session_id>', methods=['DELETE'])
@token_required
def delete_leaf_session(session_id):
    """Delete a leaf session"""
    try:
        session_ref = db.collection('leaf_sessions').document(session_id)
        session_doc = session_ref.get()
        
        if not session_doc.exists:
            return jsonify({'error': 'Session not found'}), 404
        
        session_data = session_doc.to_dict()
        
        if session_data['userId'] != request.user_id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        session_ref.delete()
        
        return jsonify({
            'success': True,
            'message': 'Session deleted successfully'
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ✅ Health check for models
@disease_bp.route('/health', methods=['GET'])
def disease_health():
    """Check disease detection service health"""
    return jsonify({
        'success': True,
        'models': {
            'leaf_model': leaf_model is not None,
            'bean_model': bean_model is not None,
            'leaf_classes': leaf_classes is not None,
            'bean_classes': bean_classes is not None,
        }
    }), 200