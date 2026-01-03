import { Image } from "expo-image"
import React from "react"
import { StyleSheet, View } from "react-native"

export default function ExploreScreen() {
  return (
    <View style={styles.container}>
      <Image source={require("@/assets/images/ceylon-coffee-logo.png")} style={styles.logo} contentFit="contain" />
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F5F1ED",
    alignItems: "center",
    justifyContent: "center",
  },
  logo: {
    width: 320,
    height: 320,
  },
})
