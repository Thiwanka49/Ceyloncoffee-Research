#!/usr/bin/env python3
"""
Create compatible encoders locally
"""

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

print("\n" + "="*70)
print("🔧 Creating Compatible Encoders")
print("="*70)

print(f"\nNumpy version: {np.__version__}")

# Create variety encoder
print("\n📦 Creating variety encoder...")
variety_encoder = LabelEncoder()
variety_encoder.fit(['Arabica', 'Robusta'])
print(f"✅ Variety classes: {list(variety_encoder.classes_)}")

# Create disease encoder
print("\n📦 Creating disease encoder...")
disease_encoder = LabelEncoder()
disease_encoder.fit([
    'nodisease', 
    'rust', 
    'miner', 
    'phoma', 
    'brown_eye_spot', 
    'red_spider_mite'
])
print(f"✅ Disease classes: {list(disease_encoder.classes_)}")

# Save to models directory
os.makedirs('models/yield', exist_ok=True)

print("\n💾 Saving encoders...")
joblib.dump(variety_encoder, 'models/yield/variety_encoder.pkl')
joblib.dump(disease_encoder, 'models/yield/disease_encoder.pkl')

print("✅ Encoders saved successfully!")

# Verify
print("\n🧪 Verifying encoders...")
v_enc = joblib.load('models/yield/variety_encoder.pkl')
d_enc = joblib.load('models/yield/disease_encoder.pkl')

print(f"✅ Variety encoder: {list(v_enc.classes_)}")
print(f"✅ Disease encoder: {list(d_enc.classes_)}")

# Test encoding
print("\n🧪 Testing encoding...")
print(f"Arabica -> {v_enc.transform(['Arabica'])[0]}")
print(f"Robusta -> {v_enc.transform(['Robusta'])[0]}")
print(f"nodisease -> {d_enc.transform(['nodisease'])[0]}")
print(f"rust -> {d_enc.transform(['rust'])[0]}")

print("\n" + "="*70)
print("✅ ALL DONE - Encoders are ready!")
print("="*70 + "\n")