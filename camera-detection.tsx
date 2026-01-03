"use client";

import { Ionicons } from "@expo/vector-icons";
import { Camera, CameraView } from "expo-camera";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import React, { useEffect, useRef, useState } from "react";
import {
  Animated,
  Dimensions,
  Platform,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

const { width } = Dimensions.get("window");

export default function CameraDetectionScreen() {
  const router = useRouter();
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [detectedPlant, setDetectedPlant] = useState<string | null>(null);
  const [plantVariety, setPlantVariety] = useState<string | null>(null);
  const cameraRef = useRef<any>(null);

  // Animation values
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.8)).current;
  const scanLineAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    // Request camera permissions
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === "granted");
    })();

    // Entrance animation
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 600,
        useNativeDriver: true,
      }),
      Animated.spring(scaleAnim, {
        toValue: 1,
        friction: 8,
        tension: 40,
        useNativeDriver: true,
      }),
    ]).start();

    // Continuous scan line animation
    Animated.loop(
      Animated.sequence([
        Animated.timing(scanLineAnim, {
          toValue: 1,
          duration: 2000,
          useNativeDriver: true,
        }),
        Animated.timing(scanLineAnim, {
          toValue: 0,
          duration: 0,
          useNativeDriver: true,
        }),
      ])
    ).start();

    // Simulate plant detection after 2 seconds
    const timer = setTimeout(() => {
      setDetectedPlant("Plant Coffee");
      setPlantVariety("Regal Do Lobo");
    }, 2000);

    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const scanLineTranslateY = scanLineAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0, 450],
  });

  if (hasPermission === null) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.permissionText}>
          Requesting camera permission...
        </Text>
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.permissionContainer}>
        <Ionicons name="camera-outline" size={64} color="#666" />
        <Text style={styles.permissionText}>No access to camera</Text>
        <TouchableOpacity
          style={styles.permissionButton}
          onPress={() => router.back()}
        >
          <Text style={styles.permissionButtonText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar
        barStyle="light-content"
        backgroundColor="transparent"
        translucent
      />

      {/* Camera View */}
      <CameraView style={styles.camera} ref={cameraRef} facing="back">
        {/* Back Button */}
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => router.back()}
          activeOpacity={0.8}
        >
          <View style={styles.backButtonInner}>
            <Ionicons name="arrow-back" size={28} color="#333" />
          </View>
        </TouchableOpacity>

        {/* Instruction Text */}
        <Animated.View
          style={[
            styles.instructionContainer,
            {
              opacity: fadeAnim,
              transform: [{ scale: scaleAnim }],
            },
          ]}
        >
          <LinearGradient
            colors={["rgba(0,0,0,0.7)", "rgba(0,0,0,0.5)"]}
            style={styles.instructionGradient}
          >
            <Text style={styles.instructionText}>
              Point Your Camera At Plant
            </Text>
          </LinearGradient>
        </Animated.View>

        {/* Camera Frame with Corner Brackets */}
        <View style={styles.cameraFrame}>
          {/* Top Left Corner */}
          <View style={[styles.corner, styles.cornerTopLeft]}>
            <View style={styles.cornerHorizontalLeft} />
            <View style={styles.cornerVerticalTopLeft} />
          </View>

          {/* Top Right Corner */}
          <View style={[styles.corner, styles.cornerTopRight]}>
            <View style={styles.cornerHorizontalRight} />
            <View style={styles.cornerVerticalTopRight} />
          </View>

          {/* Bottom Left Corner */}
          <View style={[styles.corner, styles.cornerBottomLeft]}>
            <View style={styles.cornerHorizontalBottomLeft} />
            <View style={styles.cornerVerticalBottomLeft} />
          </View>

          {/* Bottom Right Corner */}
          <View style={[styles.corner, styles.cornerBottomRight]}>
            <View style={styles.cornerHorizontalBottomRight} />
            <View style={styles.cornerVerticalBottomRight} />
          </View>

          {/* Scanning Line Animation */}
          <Animated.View
            style={[
              styles.scanLine,
              {
                transform: [{ translateY: scanLineTranslateY }],
              },
            ]}
          />
        </View>

        {/* Analyze Button Below Scanning Box */}
        <Animated.View
          style={[
            styles.analyzeButtonContainer,
            {
              opacity: fadeAnim,
              transform: [{ scale: scaleAnim }],
            },
          ]}
        >
          <TouchableOpacity
            style={styles.analyzeButton}
            onPress={() => router.push("/disease-severity")}
            activeOpacity={0.8}
          >
            <LinearGradient
              colors={["#6F4E37", "#8B6341"]}
              style={styles.analyzeButtonGradient}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
            >
              <Ionicons name="analytics" size={24} color="#FFF" />
              <Text style={styles.analyzeButtonText}>Analyze Disease</Text>
              <Ionicons name="arrow-forward" size={20} color="#FFF" />
            </LinearGradient>
          </TouchableOpacity>
        </Animated.View>

        {/* Detection Result Card */}
        {/* {detectedPlant && (
          <Animated.View 
            style={[
              styles.resultCard,
              {
                opacity: fadeAnim,
              }
            ]}
          >
            <TouchableOpacity activeOpacity={0.9}>
              <LinearGradient
                colors={["#FFFFFF", "#F8F8F8"]}
                style={styles.resultGradient}
              >
                <View style={styles.resultContent}>
                  <View style={styles.plantIconContainer}>
                    <Ionicons name="leaf" size={40} color="#6F4E37" />
                  </View>
                  <View style={styles.plantInfo}>
                    <Text style={styles.plantName}>{detectedPlant}</Text>
                    <Text style={styles.plantVariety}>{plantVariety}</Text>
                  </View>
                  <View style={styles.arrowContainer}>
                    <Ionicons name="chevron-forward" size={32} color="#6F4E37" />
                  </View>
                </View>
              </LinearGradient>
            </TouchableOpacity>
          </Animated.View>
        )} */}
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#000",
  },
  camera: {
    flex: 1,
    justifyContent: "space-between",
  },
  permissionContainer: {
    flex: 1,
    backgroundColor: "#F5F1ED",
    justifyContent: "center",
    alignItems: "center",
    gap: 20,
    padding: 20,
  },
  permissionText: {
    fontSize: 18,
    color: "#666",
    textAlign: "center",
  },
  permissionButton: {
    backgroundColor: "#6F4E37",
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 12,
    marginTop: 10,
  },
  permissionButtonText: {
    color: "#FFF",
    fontSize: 16,
    fontWeight: "600",
  },
  backButton: {
    position: "absolute",
    top:
      Platform.OS === "ios"
        ? 60
        : StatusBar.currentHeight
        ? StatusBar.currentHeight + 20
        : 40,
    left: 20,
    zIndex: 10,
  },
  backButtonInner: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: "rgba(255, 255, 255, 0.95)",
    justifyContent: "center",
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  instructionContainer: {
    position: "absolute",
    top:
      Platform.OS === "ios"
        ? 130
        : StatusBar.currentHeight
        ? StatusBar.currentHeight + 90
        : 110,
    left: 20,
    right: 20,
    alignItems: "center",
    zIndex: 5,
  },
  instructionGradient: {
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 25,
  },
  instructionText: {
    color: "#FFF",
    fontSize: 18,
    fontWeight: "600",
    textAlign: "center",
  },
  cameraFrame: {
    position: "absolute",
    top: "60%",
    left: "50%",
    width: width * 0.8,
    height: width * 0.8,
    marginLeft: -width * 0.4,
    marginTop: -225,
    zIndex: 1,
  },
  corner: {
    position: "absolute",
    width: 60,
    height: 60,
  },
  cornerTopLeft: {
    top: 0,
    left: 0,
  },
  cornerTopRight: {
    top: 0,
    right: 0,
  },
  cornerBottomLeft: {
    bottom: 0,
    left: 0,
  },
  cornerBottomRight: {
    bottom: 0,
    right: 0,
  },
  cornerHorizontalLeft: {
    position: "absolute",
    width: 60,
    height: 4,
    backgroundColor: "#FFF",
    borderRadius: 10,
    top: 0,
    left: 0,
  },
  cornerHorizontalRight: {
    position: "absolute",
    width: 60,
    height: 4,
    backgroundColor: "#FFF",
    borderRadius: 10,
    top: 0,
    right: 0,
  },
  cornerHorizontalBottomLeft: {
    position: "absolute",
    width: 60,
    height: 4,
    backgroundColor: "#FFF",
    borderRadius: 2,
    bottom: 0,
    left: 0,
  },
  cornerHorizontalBottomRight: {
    position: "absolute",
    width: 60,
    height: 4,
    backgroundColor: "#FFF",
    borderRadius: 2,
    bottom: 0,
    right: 0,
  },
  cornerVerticalTopLeft: {
    position: "absolute",
    width: 4,
    height: 60,
    backgroundColor: "#FFF",
    borderRadius: 2,
    top: 0,
    left: 0,
  },
  cornerVerticalTopRight: {
    position: "absolute",
    width: 4,
    height: 60,
    backgroundColor: "#FFF",
    borderRadius: 2,
    top: 0,
    right: 0,
  },
  cornerVerticalBottomLeft: {
    position: "absolute",
    width: 4,
    height: 60,
    backgroundColor: "#FFF",
    borderRadius: 2,
    bottom: 0,
    left: 0,
  },
  cornerVerticalBottomRight: {
    position: "absolute",
    width: 4,
    height: 60,
    backgroundColor: "#FFF",
    borderRadius: 2,
    bottom: 0,
    right: 0,
  },
  scanLine: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    height: 3,
    backgroundColor: "#4ECDC4",
    shadowColor: "#4ECDC4",
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 10,
    elevation: 5,
  },
  resultCard: {
    position: "absolute",
    bottom: "10%",
    left: 20,
    right: 20,
    zIndex: 10,
  },
  resultGradient: {
    borderRadius: 20,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  resultContent: {
    flexDirection: "row",
    alignItems: "center",
    padding: 20,
    gap: 15,
  },
  plantIconContainer: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: "#F5F1ED",
    justifyContent: "center",
    alignItems: "center",
  },
  plantInfo: {
    flex: 1,
    gap: 5,
  },
  plantName: {
    fontSize: 22,
    fontWeight: "700",
    color: "#2C2C2C",
    letterSpacing: 0.5,
  },
  plantVariety: {
    fontSize: 16,
    color: "#888",
    fontWeight: "500",
  },
  arrowContainer: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: "#D4A574",
    justifyContent: "center",
    alignItems: "center",
  },
  analyzeButtonContainer: {
    position: "absolute",
    bottom: "15%",
    left: 20,
    right: 20,
    zIndex: 10,
  },
  analyzeButton: {
    borderRadius: 16,
    overflow: "hidden",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  analyzeButtonGradient: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 18,
    paddingHorizontal: 24,
    gap: 12,
  },
  analyzeButtonText: {
    fontSize: 18,
    fontWeight: "700",
    color: "#FFF",
    letterSpacing: 0.5,
  },
});
