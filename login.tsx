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

export default function LoginPage() {
  const [showPassword, setShowPassword] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
  });
  
  const router = useRouter();

  const handleSubmit = () => {
    console.log('Login:', formData);
    // Handle login logic here
  };

  const handleChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleForgotPassword = () => {
    // Handle forgot password logic here
    console.log('Forgot password pressed');
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
            <Text style={styles.title}>Welcome Back</Text>
            <Text style={styles.subtitle}>Sign in to your account</Text>
          </View>

          {/* Form Section */}
          <View style={styles.formContainer}>
            <View style={styles.form}>
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
                <View style={styles.passwordHeader}>
                  <Text style={styles.inputLabel}>Password</Text>
                  <TouchableOpacity onPress={handleForgotPassword}>
                    <Text style={styles.forgotPasswordText}>Forgot Password?</Text>
                  </TouchableOpacity>
                </View>
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

              {/* Remember Me Checkbox */}
              <View style={styles.rememberMeContainer}>
                <TouchableOpacity style={styles.checkbox}>
                  <MaterialIcons name="check-box-outline-blank" size={20} color="#6B5B4F" />
                </TouchableOpacity>
                <Text style={styles.rememberMeText}>Remember me</Text>
              </View>

              {/* Login Button */}
              <TouchableOpacity style={styles.loginButton} onPress={handleSubmit}>
                <Text style={styles.loginButtonText}>Sign in</Text>
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

              {/* Signup Link */}
              <View style={styles.signupLinkContainer}>
                <Text style={styles.signupText}>
                  Don't have an account?{' '}
                </Text>
                <TouchableOpacity onPress={() => router.push('/signup')}>
                  <Text style={styles.signupLink}>Sign up</Text>
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
    marginBottom: -10,
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
  passwordHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  inputLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginLeft: 4,
  },
  forgotPasswordText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#8B5A3C',
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
  rememberMeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  checkbox: {
    padding: 4,
  },
  rememberMeText: {
    fontSize: 14,
    color: '#374151',
    fontWeight: '500',
  },
  loginButton: {
    backgroundColor: '#8B5A3C',
    height: 56,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 8,
    shadowColor: '#8B5A3C',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
  },
  loginButtonText: {
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
  signupLinkContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 24,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#F3F4F6',
  },
  signupText: {
    color: '#6B7280',
    fontSize: 16,
  },
  signupLink: {
    color: '#8B5A3C',
    fontSize: 16,
    fontWeight: '600',
  },
});