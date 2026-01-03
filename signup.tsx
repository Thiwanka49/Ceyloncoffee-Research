import React, { useState } from 'react';
import { 
  View, 
  Text, 
  TextInput, 
  TouchableOpacity, 
  Image, 
  ScrollView,
  StyleSheet 
} from 'react-native';
import { useRouter } from 'expo-router';
import { FontAwesome, Ionicons, MaterialIcons } from '@expo/vector-icons';

export default function SignupPage() {
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    phone: '',
    email: '',
    password: '',
    confirmPassword: '',
  });
  
  const router = useRouter();

  const handleSubmit = () => {
    console.log('Signup:', formData);
  };

  const handleChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <View style={styles.container}>
      <ScrollView 
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.mainContent}>
          {/* Header Section */}
          <View style={styles.header}>
            <View style={styles.logoContainer}>
              <Image
                source={require('@/assets/images/ceylon-coffee-logo.png')}
                style={styles.logo}
                accessibilityLabel="Ceylon Coffee Logo"
              />
            </View>
            <Text style={styles.title}>Sign up</Text>
            <Text style={styles.subtitle}>Create your account</Text>
          </View>

          {/* Form Section */}
          <View style={styles.formContainer}>
            <View style={styles.form}>
              {/* Name Input */}
              <View style={styles.inputWrapper}>
                <Text style={styles.inputLabel}>Name</Text>
                <View style={styles.inputContainer}>
                  <MaterialIcons name="person" size={20} color="#6B5B4F" style={styles.inputIcon} />
                  <TextInput
                    style={styles.input}
                    placeholder="Enter your name"
                    value={formData.name}
                    onChangeText={(value) => handleChange('name', value)}
                    placeholderTextColor="#9CA3AF"
                  />
                </View>
              </View>

              {/* Phone Input */}
              <View style={styles.inputWrapper}>
                <Text style={styles.inputLabel}>Phone</Text>
                <View style={styles.inputContainer}>
                  <MaterialIcons name="phone" size={20} color="#6B5B4F" style={styles.inputIcon} />
                  <TextInput
                    style={styles.input}
                    placeholder="Enter your phone number"
                    value={formData.phone}
                    onChangeText={(value) => handleChange('phone', value)}
                    keyboardType="phone-pad"
                    placeholderTextColor="#9CA3AF"
                  />
                </View>
              </View>

              {/* Email Input */}
              <View style={styles.inputWrapper}>
                <Text style={styles.inputLabel}>Email</Text>
                <View style={styles.inputContainer}>
                  <MaterialIcons name="email" size={20} color="#6B5B4F" style={styles.inputIcon} />
                  <TextInput
                    style={styles.input}
                    placeholder="Enter your email"
                    value={formData.email}
                    onChangeText={(value) => handleChange('email', value)}
                    keyboardType="email-address"
                    autoCapitalize="none"
                    placeholderTextColor="#9CA3AF"
                  />
                </View>
              </View>

              {/* Password Input */}
              <View style={styles.inputWrapper}>
                <Text style={styles.inputLabel}>Password</Text>
                <View style={styles.inputContainer}>
                  <MaterialIcons name="lock" size={20} color="#6B5B4F" style={styles.inputIcon} />
                  <TextInput
                    style={styles.input}
                    placeholder="Enter your password"
                    value={formData.password}
                    onChangeText={(value) => handleChange('password', value)}
                    secureTextEntry={!showPassword}
                    placeholderTextColor="#9CA3AF"
                  />
                  <TouchableOpacity 
                    style={styles.eyeButton}
                    onPress={() => setShowPassword(!showPassword)}
                  >
                    <Ionicons 
                      name={showPassword ? "eye-off" : "eye"} 
                      size={20} 
                      color="#6B5B4F" 
                    />
                  </TouchableOpacity>
                </View>
              </View>

              {/* Confirm Password Input */}
              <View style={styles.inputWrapper}>
                <Text style={styles.inputLabel}>Confirm Password</Text>
                <View style={styles.inputContainer}>
                  <MaterialIcons name="lock" size={20} color="#6B5B4F" style={styles.inputIcon} />
                  <TextInput
                    style={styles.input}
                    placeholder="Confirm your password"
                    value={formData.confirmPassword}
                    onChangeText={(value) => handleChange('confirmPassword', value)}
                    secureTextEntry={!showConfirmPassword}
                    placeholderTextColor="#9CA3AF"
                  />
                  <TouchableOpacity 
                    style={styles.eyeButton}
                    onPress={() => setShowConfirmPassword(!showConfirmPassword)}
                  >
                    <Ionicons 
                      name={showConfirmPassword ? "eye-off" : "eye"} 
                      size={20} 
                      color="#6B5B4F" 
                    />
                  </TouchableOpacity>
                </View>
              </View>

              {/* Sign up Button */}
              <TouchableOpacity style={styles.signupButton} onPress={handleSubmit}>
                <Text style={styles.signupButtonText}>Sign up</Text>
              </TouchableOpacity>

              {/* Divider */}
              <View style={styles.divider}>
                <View style={styles.dividerLine} />
                <Text style={styles.dividerText}>or</Text>
                <View style={styles.dividerLine} />
              </View>

              {/* Continue with */}
              <Text style={styles.continueText}>Continue with</Text>

              {/* Social Login Buttons */}
              <View style={styles.socialContainer}>
                <TouchableOpacity style={styles.socialButton}>
                  <FontAwesome name="google" size={24} color="#DB4437" />
                </TouchableOpacity>
                <TouchableOpacity style={styles.socialButton}>
                  <FontAwesome name="facebook" size={24} color="#1877F2" />
                </TouchableOpacity>
                <TouchableOpacity style={styles.socialButton}>
                  <FontAwesome name="apple" size={24} color="#000000" />
                </TouchableOpacity>
              </View>

              {/* Login Link */}
              <View style={styles.loginLinkContainer}>
                <Text style={styles.loginText}>
                  Already have an account?{' '}
                </Text>
                <TouchableOpacity onPress={() => router.push('/login')}>
                  <Text style={styles.loginLink}>Login</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F1ED',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
  },
  mainContent: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: 20,
    paddingVertical: 20,
  },
  header: {
    backgroundColor: '#8B5A3C',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    paddingVertical: 40,
    paddingHorizontal: 32,
    alignItems: 'center',
    marginBottom: -10, // Overlap with form container
    zIndex: 1,
  },
  logoContainer: {
    width: 80,
    height: 80,
    marginBottom: 16,
    backgroundColor: '#8B5A3C',
    borderRadius: 40,
    padding: 8,
    shadowColor: '#8B5A3C',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 5,
  },
  logo: {
    width: '100%',
    height: '100%',
    resizeMode: 'contain',
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: 'rgba(255,255,255,0.8)',
  },
  formContainer: {
    backgroundColor: 'white',
    borderBottomLeftRadius: 24,
    borderBottomRightRadius: 24,
    paddingHorizontal: 24,
    paddingVertical: 32,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.1,
    shadowRadius: 20,
    elevation: 10,
    zIndex: 0,
  },
  form: {
    gap: 20,
  },
  inputWrapper: {
    gap: 8,
  },
  inputLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginLeft: 4,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#E5E7EB',
    borderRadius: 12,
    height: 56,
    backgroundColor: '#F9FAFB',
  },
  inputIcon: {
    marginLeft: 16,
    marginRight: 12,
  },
  input: {
    flex: 1,
    paddingHorizontal: 0,
    fontSize: 16,
    color: '#374151',
    paddingVertical: 8,
  },
  eyeButton: {
    padding: 16,
  },
  signupButton: {
    backgroundColor: '#8B5A3C',
    height: 56,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 8,
    shadowColor: '#2D5F4F',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
  },
  signupButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
  },
  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    paddingVertical: 24,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: '#E5E7EB',
  },
  dividerText: {
    color: '#6B7280',
    fontSize: 14,
    fontWeight: '500',
  },
  continueText: {
    textAlign: 'center',
    color: '#6B7280',
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 20,
  },
  socialContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 20,
  },
  socialButton: {
    width: 56,
    height: 56,
    borderRadius: 28,
    borderWidth: 2,
    borderColor: '#E5E7EB',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'white',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  loginLinkContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 24,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#F3F4F6',
  },
  loginText: {
    color: '#6B7280',
    fontSize: 16,
  },
  loginLink: {
    color: '#2D5F4F',
    fontSize: 16,
    fontWeight: '600',
  },
});