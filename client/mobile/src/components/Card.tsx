import React from "react";
import { View, StyleSheet, useWindowDimensions, ViewStyle } from "react-native";
import { colors, radius, spacing } from "../lib/theme";

interface CardProps {
  children: React.ReactNode;
  style?: ViewStyle;
}

export default function Card({ children, style }: CardProps) {
  const { width } = useWindowDimensions();
  const isTablet = width >= 768;
  return (
    <View style={[styles.card, isTablet && styles.tabletCard, style]}>
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    padding: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border,
  },
  tabletCard: {
    padding: spacing.xl,
  },
});
