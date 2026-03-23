import React from "react";
import { View, Text, StyleSheet } from "react-native";
import { colors, fontSize, radius, spacing } from "../lib/theme";

type Variant = "success" | "error" | "warning" | "info" | "neutral";

const BG: Record<Variant, string> = {
  success: "#D1FAE5",
  error: "#FEE2E2",
  warning: "#FEF3C7",
  info: "#DBEAFE",
  neutral: "#F3F4F6",
};
const FG: Record<Variant, string> = {
  success: "#065F46",
  error: "#991B1B",
  warning: "#92400E",
  info: "#1E40AF",
  neutral: "#374151",
};

interface BadgeProps {
  label: string;
  variant?: Variant;
}

export default function Badge({ label, variant = "neutral" }: BadgeProps) {
  return (
    <View style={[styles.badge, { backgroundColor: BG[variant] }]}>
      <Text style={[styles.text, { color: FG[variant] }]}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: radius.full,
    alignSelf: "flex-start",
  },
  text: {
    fontSize: fontSize.xs,
    fontWeight: "600",
  },
});
