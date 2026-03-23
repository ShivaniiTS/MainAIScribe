import React from "react";
import { View, StyleSheet } from "react-native";
import { colors, radius } from "../lib/theme";

interface ProgressBarProps {
  progress: number; // 0–100
  color?: string;
}

export default function ProgressBar({ progress, color = colors.brand }: ProgressBarProps) {
  return (
    <View style={styles.track}>
      <View style={[styles.fill, { width: `${Math.min(100, Math.max(0, progress))}%`, backgroundColor: color }]} />
    </View>
  );
}

const styles = StyleSheet.create({
  track: {
    height: 6,
    backgroundColor: "#E5E7EB",
    borderRadius: radius.full,
    overflow: "hidden",
  },
  fill: {
    height: 6,
    borderRadius: radius.full,
  },
});
