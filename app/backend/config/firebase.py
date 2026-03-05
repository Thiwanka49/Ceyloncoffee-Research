import firebase_admin
from firebase_admin import credentials, auth, firestore
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Firebase Admin (NO STORAGE)
cred = credentials.Certificate('firebase-credentials.json')
firebase_admin.initialize_app(cred)

# Get Firestore client
db = firestore.client()

def verify_token(id_token):
    """Verify Firebase ID token from React Native app"""
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        print(f"❌ Token verification error: {e}")
        return None

def get_user_by_uid(uid):
    """Get user from Firebase Auth by UID"""
    try:
        user_record = auth.get_user(uid)
        return user_record
    except Exception as e:
        print(f"❌ Error getting user: {e}")
        return None

print("✅ Firebase Admin SDK initialized successfully (Firestore only)")