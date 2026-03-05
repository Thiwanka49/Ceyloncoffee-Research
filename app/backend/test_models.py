"""
Model Validation Script
Run this to verify all models are loaded correctly
"""

import os
import json
import numpy as np
from PIL import Image

print("\n" + "="*70)
print("🔍 CEYLON COFFEE - MODEL VALIDATION")
print("="*70)

# ============================================================================
# CHECK FILE EXISTENCE
# ============================================================================
print("\n📁 Step 1: Checking File Existence...")
print("-"*70)

files_to_check = {
    'Leaf Disease Model': 'models/disease/coffee_l_disease_classifier.keras',
    'Leaf Classes': 'models/disease/class_indices_coffee_l_disease.json',
    'Bean Disease Model': 'models/disease/coffee_beans_classifier.keras',
    'Bean Classes': 'models/disease/class_indices_coffee_beans.json',
    'Yield Model': 'models/yield/yield_xgboost_model.pkl',
    'Variety Encoder': 'models/yield/variety_encoder.pkl',
    'Disease Encoder': 'models/yield/disease_encoder.pkl',
    'Feature Names': 'models/yield/feature_names.txt',
}

files_exist = {}
for name, path in files_to_check.items():
    exists = os.path.exists(path)
    files_exist[name] = exists
    status = "✅" if exists else "❌"
    size = os.path.getsize(path) if exists else 0
    size_mb = size / (1024 * 1024)
    print(f"{status} {name:25s} | {path:50s} | {size_mb:.2f} MB")

# ============================================================================
# LOAD AND TEST TENSORFLOW MODELS
# ============================================================================
print("\n🧠 Step 2: Loading TensorFlow Models...")
print("-"*70)

try:
    import tensorflow as tf
    print(f"✅ TensorFlow version: {tf.__version__}")
    
    # Test Leaf Model
    if files_exist['Leaf Disease Model']:
        print("\n📦 Loading Leaf Disease Model...")
        try:
            from tensorflow.keras.models import load_model
            leaf_model = load_model('models/disease/coffee_l_disease_classifier.keras')
            print(f"✅ Leaf model loaded successfully")
            print(f"   Model type: {type(leaf_model)}")
            print(f"   Input shape: {leaf_model.input_shape}")
            print(f"   Output shape: {leaf_model.output_shape}")
            
            # Load classes
            with open('models/disease/class_indices_coffee_l_disease.json', 'r') as f:
                leaf_classes = json.load(f)
            print(f"   Classes: {list(leaf_classes.keys())}")
            
            # Test prediction with dummy data
            print("\n🧪 Testing leaf model prediction...")
            dummy_img = np.random.rand(1, 128, 128, 3).astype(np.float32)
            predictions = leaf_model.predict(dummy_img, verbose=0)
            print(f"✅ Prediction successful!")
            print(f"   Prediction shape: {predictions.shape}")
            print(f"   Predictions: {predictions}")
            print(f"   Sum of probabilities: {np.sum(predictions):.4f} (should be ~1.0)")
            
        except Exception as e:
            print(f"❌ Error loading leaf model: {e}")
            import traceback
            traceback.print_exc()
    
    # Test Bean Model
    if files_exist['Bean Disease Model']:
        print("\n📦 Loading Bean Disease Model...")
        try:
            bean_model = load_model('models/disease/coffee_beans_classifier.keras')
            print(f"✅ Bean model loaded successfully")
            print(f"   Model type: {type(bean_model)}")
            print(f"   Input shape: {bean_model.input_shape}")
            print(f"   Output shape: {bean_model.output_shape}")
            
            # Load classes
            with open('models/disease/class_indices_coffee_beans.json', 'r') as f:
                bean_classes = json.load(f)
            print(f"   Classes: {list(bean_classes.keys())}")
            
            # Test prediction
            print("\n🧪 Testing bean model prediction...")
            dummy_img = np.random.rand(1, 128, 128, 3).astype(np.float32)
            predictions = bean_model.predict(dummy_img, verbose=0)
            print(f"✅ Prediction successful!")
            print(f"   Prediction shape: {predictions.shape}")
            print(f"   Predictions: {predictions}")
            print(f"   Sum of probabilities: {np.sum(predictions):.4f} (should be ~1.0)")
            
        except Exception as e:
            print(f"❌ Error loading bean model: {e}")
            import traceback
            traceback.print_exc()
    
except ImportError:
    print("❌ TensorFlow not installed!")
    print("   Install with: pip install tensorflow")

# ============================================================================
# LOAD AND TEST SKLEARN MODELS
# ============================================================================
print("\n🎯 Step 3: Loading Scikit-learn Models...")
print("-"*70)

try:
    import joblib
    import xgboost
    print(f"✅ XGBoost version: {xgboost.__version__}")
    
    if files_exist['Yield Model']:
        print("\n📦 Loading Yield Prediction Model...")
        try:
            yield_model = joblib.load('models/yield/yield_xgboost_model.pkl')
            variety_encoder = joblib.load('models/yield/variety_encoder.pkl')
            disease_encoder = joblib.load('models/yield/disease_encoder.pkl')
            
            print(f"✅ Yield model loaded successfully")
            print(f"   Model type: {type(yield_model)}")
            
            # Load feature names
            with open('models/yield/feature_names.txt', 'r') as f:
                features = [line.strip() for line in f.readlines()]
            print(f"   Number of features: {len(features)}")
            print(f"   Features: {features[:5]}... (showing first 5)")
            
            # Encoders
            print(f"   Variety classes: {list(variety_encoder.classes_)}")
            print(f"   Disease classes: {list(disease_encoder.classes_)}")
            
            # Test prediction
            print("\n🧪 Testing yield model prediction...")
            dummy_features = np.random.rand(1, len(features))
            prediction = yield_model.predict(dummy_features)
            print(f"✅ Prediction successful!")
            print(f"   Predicted yield: {prediction[0]:.2f} kg")
            
        except Exception as e:
            print(f"❌ Error loading yield model: {e}")
            import traceback
            traceback.print_exc()
    
except ImportError as e:
    print(f"❌ Required package not installed: {e}")
    print("   Install with: pip install xgboost joblib scikit-learn")

# ============================================================================
# TEST IMAGE PROCESSING
# ============================================================================
print("\n🖼️  Step 4: Testing Image Processing...")
print("-"*70)

try:
    import cv2
    from PIL import Image
    
    print("✅ PIL (Pillow) available")
    print("✅ OpenCV available")
    
    # Create test image
    print("\n🧪 Creating test image...")
    test_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    pil_img = Image.fromarray(test_img)
    print(f"✅ PIL image created: {pil_img.size}")
    
    # Test preprocessing
    img_array = np.array(pil_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    print(f"✅ Preprocessing successful: {img_array.shape}")
    
except ImportError as e:
    print(f"❌ Image processing library not available: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("📊 VALIDATION SUMMARY")
print("="*70)

all_files_exist = all(files_exist.values())
print(f"\n📁 Files: {'✅ All present' if all_files_exist else '❌ Some missing'}")

try:
    models_loaded = (
        'leaf_model' in locals() and 
        'bean_model' in locals() and 
        'yield_model' in locals()
    )
    print(f"🧠 Models: {'✅ All loaded' if models_loaded else '❌ Some failed'}")
except:
    print(f"🧠 Models: ❌ Failed to load")

if all_files_exist and 'leaf_model' in locals() and 'bean_model' in locals():
    print(f"\n✅ ALL CHECKS PASSED - Ready for production!")
else:
    print(f"\n⚠️  SOME CHECKS FAILED - Review errors above")

print("="*70 + "\n")