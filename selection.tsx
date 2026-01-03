"use client"

import { useEffect, useRef, useState } from "react"
import { View, Text, TouchableOpacity, Animated, StyleSheet, Dimensions, StatusBar, ImageBackground } from "react-native"
import { LinearGradient } from "expo-linear-gradient"
import { useRouter } from "expo-router" // ✅ Import useRouter for navigation
import React from "react"

const { width, height } = Dimensions.get("window")

export default function SelectionScreen() {
  const fadeAnim = useRef(new Animated.Value(0)).current
  const slideAnim = useRef(new Animated.Value(50)).current
  const scaleAnim = useRef(new Animated.Value(0.9)).current

  const [selectedOption, setSelectedOption] = useState<"register" | "login" | null>(null)
  const router = useRouter() // ✅ Initialize router

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 1000,
        useNativeDriver: true,
      }),
      Animated.spring(slideAnim, {
        toValue: 0,
        friction: 8,
        tension: 40,
        useNativeDriver: true,
      }),
      Animated.spring(scaleAnim, {
        toValue: 1,
        friction: 8,
        tension: 40,
        useNativeDriver: true,
      }),
    ]).start()
  }, [])

  // ✅ Button Press Handler with Navigation
  const handleButtonPress = (option: "register" | "login") => {
    setSelectedOption(option)

    if (option === "register") {
      router.push("/signup") // Navigate to signup.tsx
    } else {
      router.push("/login") // Navigate to login.tsx
    }
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#F5F1ED" />

      <LinearGradient colors={["#F5F1ED", "#F5F1ED", "#F5F1ED"]} style={styles.background}>
        <View style={[styles.decorativeCircle, styles.circle1]} />
        <View style={[styles.decorativeCircle, styles.circle2]} />

        <Animated.View
          style={[
            styles.cardContainer,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnim }, { scale: scaleAnim }],
            },
          ]}
        >
          <LinearGradient colors={["#F5F1ED", "#FDFBF9"]} style={styles.card}>
            <View style={styles.imageContainer}>
              <ImageBackground
                source={require("@/assets/images/ceylon-coffee-logo.png")}
                style={styles.coffeeImage}
                resizeMode="contain"
              />
            </View>

            <View style={styles.contentContainer}>
              <Text style={styles.mainTitle}>Welcome to{"\n"}Ceylon Coffee</Text>

              <Text style={styles.subtitle}>
                Where quality meets every sip. Elevating coffee excellence, one cup at a time.
              </Text>

              <View style={styles.buttonsContainer}>
                {/* Register Button */}
                <TouchableOpacity onPress={() => handleButtonPress("register")} activeOpacity={0.85}>
                  <Animated.View
                    style={[
                      styles.buttonWrapper,
                      {
                        transform: [{ scale: selectedOption === "register" ? 0.98 : 1 }],
                      },
                    ]}
                  >
                    <LinearGradient
                      colors={["#4A3426", "#3D2817"]}
                      start={{ x: 0, y: 0 }}
                      end={{ x: 1, y: 1 }}
                      style={styles.registerButton}
                    >
                      <Text style={styles.registerButtonText}>Register</Text>
                    </LinearGradient>
                  </Animated.View>
                </TouchableOpacity>

                {/* Login Button */}
                <TouchableOpacity onPress={() => handleButtonPress("login")} activeOpacity={0.85}>
                  <Animated.View
                    style={[
                      styles.buttonWrapper,
                      {
                        transform: [{ scale: selectedOption === "login" ? 0.98 : 1 }],
                      },
                    ]}
                  >
                    <LinearGradient
                      colors={["#E8E8D0", "#F0F0E0"]}
                      start={{ x: 0, y: 0 }}
                      end={{ x: 1, y: 1 }}
                      style={styles.loginButton}
                    >
                      <Text style={styles.loginButtonText}>Log in</Text>
                    </LinearGradient>
                  </Animated.View>
                </TouchableOpacity>
              </View>
            </View>
          </LinearGradient>
        </Animated.View>
      </LinearGradient>
    </View>
  )
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F5F1ED" },
  background: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 24,
    position: "relative",
    overflow: "hidden",
  },
  decorativeCircle: { position: "absolute", borderRadius: 999, opacity: 0.15 },
  circle1: { width: 280, height: 280, top: -100, right: -80, backgroundColor: "#8B4513" },
  circle2: { width: 200, height: 200, bottom: -60, left: -40, backgroundColor: "#6F4E37" },
  cardContainer: {
    width: "100%",
    maxWidth: 320,
    borderRadius: 40,
    overflow: "hidden",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.25,
    shadowRadius: 16,
    elevation: 12,
  },
  card: { paddingBottom: 40, alignItems: "center" },
  imageContainer: {
    width: "100%",
    height: 280,
    backgroundColor: "#F9F7F4",
    borderBottomLeftRadius: 30,
    borderBottomRightRadius: 30,
    overflow: "hidden",
    marginBottom: 24,
  },
  coffeeImage: { flex: 1, justifyContent: "center", alignItems: "center" },
  contentContainer: { paddingHorizontal: 28, alignItems: "center" },
  mainTitle: {
    fontSize: 32,
    fontWeight: "800",
    color: "#3D3D3D",
    textAlign: "center",
    marginBottom: 16,
    lineHeight: 40,
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize: 15,
    color: "#666666",
    textAlign: "center",
    lineHeight: 22,
    marginBottom: 32,
    fontWeight: "400",
  },
  buttonsContainer: { width: "100%", gap: 12 },
  buttonWrapper: { borderRadius: 20, overflow: "hidden" },
  registerButton: {
    paddingVertical: 14,
    paddingHorizontal: 28,
    justifyContent: "center",
    alignItems: "center",
    borderRadius: 20,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 5,
  },
  registerButtonText: { color: "#FFFFFF", fontSize: 18, fontWeight: "700", letterSpacing: 0.5 },
  loginButton: {
    paddingVertical: 14,
    paddingHorizontal: 28,
    justifyContent: "center",
    alignItems: "center",
    borderRadius: 20,
    borderWidth: 1,
    borderColor: "rgba(0, 0, 0, 0.08)",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  loginButtonText: { color: "#4A4A4A", fontSize: 18, fontWeight: "700", letterSpacing: 0.5 },
})
