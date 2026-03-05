from flask import Blueprint, request, jsonify
from config.firebase import db, auth as admin_auth
from utils.decorators import token_required
from datetime import datetime

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/verify', methods=['POST'])
@token_required
def verify_user():
    """Verify user token and return user info"""
    try:
        user_id = request.user_id
        
        # Get user data from Firestore
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        
        if user_doc.exists:
            user_data = user_doc.to_dict()
            return jsonify({
                'success': True,
                'user': user_data
            }), 200
        else:
            return jsonify({'error': 'User not found'}), 404
            
    except Exception as e:
        print(f"Error in verify_user: {str(e)}")
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/profile', methods=['GET'])
@token_required
def get_profile():
    """Get user profile from Firestore"""
    try:
        user_id = request.user_id
        user_email = request.user_email
        
        print(f"📋 Fetching profile for user: {user_id}")
        
        # Get user data from Firestore
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        
        if user_doc.exists:
            user_data = user_doc.to_dict()
            print(f"✅ Profile found: {user_data.get('name', 'Unknown')}")
            return jsonify({
                'success': True,
                'data': user_data
            }), 200
        else:
            print(f"⚠️ Profile not found for user: {user_id}")
            
            # Try to get user info from Firebase Auth
            try:
                user_record = admin_auth.get_user(user_id)
                
                # Create basic profile from Firebase Auth data
                basic_profile = {
                    'uid': user_id,
                    'email': user_email or user_record.email,
                    'name': user_record.display_name or 'User',
                    'phone': user_record.phone_number or '',
                    'createdAt': datetime.utcnow().isoformat(),
                    'role': 'farmer'
                }
                
                # Save to Firestore
                user_ref.set(basic_profile)
                print(f"✅ Created new profile for user: {user_id}")
                
                return jsonify({
                    'success': True,
                    'data': basic_profile
                }), 200
                
            except Exception as auth_error:
                print(f"❌ Error getting user from Firebase Auth: {str(auth_error)}")
                return jsonify({'error': 'Profile not found'}), 404
            
    except Exception as e:
        print(f"❌ Error in get_profile: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/profile', methods=['PUT'])
@token_required
def update_profile():
    """Update user profile"""
    try:
        user_id = request.user_id
        data = request.json
        
        print(f"📝 Updating profile for user: {user_id}")
        print(f"Data: {data}")
        
        # Remove sensitive fields that shouldn't be updated
        data.pop('uid', None)
        data.pop('createdAt', None)
        
        # Add updated timestamp
        data['updatedAt'] = datetime.utcnow().isoformat()
        
        # Update Firestore
        user_ref = db.collection('users').document(user_id)
        user_ref.update(data)
        
        print(f"✅ Profile updated successfully")
        
        return jsonify({
            'success': True,
            'message': 'Profile updated successfully'
        }), 200
        
    except Exception as e:
        print(f"❌ Error in update_profile: {str(e)}")
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/create-profile', methods=['POST'])
@token_required
def create_profile():
    """Create user profile in Firestore (called from signup)"""
    try:
        user_id = request.user_id
        user_email = request.user_email
        data = request.json
        
        print(f"👤 Creating profile for user: {user_id}")
        print(f"Data: {data}")
        
        # Check if profile already exists
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        
        if user_doc.exists:
            print(f"⚠️ Profile already exists for user: {user_id}")
            return jsonify({
                'success': True,
                'message': 'Profile already exists',
                'data': user_doc.to_dict()
            }), 200
        
        # Create new profile
        profile_data = {
            'uid': user_id,
            'email': user_email or data.get('email'),
            'name': data.get('name', ''),
            'phone': data.get('phone', ''),
            'createdAt': datetime.utcnow().isoformat(),
            'role': data.get('role', 'farmer'),
            'profileComplete': False
        }
        
        # Save to Firestore
        user_ref.set(profile_data)
        
        print(f"✅ Profile created successfully for: {profile_data.get('name')}")
        
        return jsonify({
            'success': True,
            'message': 'Profile created successfully',
            'data': profile_data
        }), 201
        
    except Exception as e:
        print(f"❌ Error in create_profile: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/delete-account', methods=['DELETE'])
@token_required
def delete_account():
    """Delete user account and all associated data"""
    try:
        user_id = request.user_id
        
        print(f"🗑️ Deleting account for user: {user_id}")
        
        # Delete user profile from Firestore
        user_ref = db.collection('users').document(user_id)
        user_ref.delete()
        
        # Delete user from Firebase Auth
        admin_auth.delete_user(user_id)
        
        print(f"✅ Account deleted successfully")
        
        return jsonify({
            'success': True,
            'message': 'Account deleted successfully'
        }), 200
        
    except Exception as e:
        print(f"❌ Error in delete_account: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Health check for auth routes
@auth_bp.route('/health', methods=['GET'])
def auth_health():
    """Health check for auth service"""
    return jsonify({
        'success': True,
        'service': 'auth',
        'status': 'healthy'
    }), 200