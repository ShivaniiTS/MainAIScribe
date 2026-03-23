/** Talisman Solutions design tokens — matches the web app */
export const colors = {
  brand: "#00B27A",
  brandDark: "#009966",
  indigo: "#6366F1",
  sidebar: "#1E1B4B",

  bg: "#F9FAFB",
  card: "#FFFFFF",
  border: "#E5E7EB",
  borderLight: "#F3F4F6",

  text: "#111827",
  textSecondary: "#6B7280",
  textTertiary: "#9CA3AF",
  textInverse: "#FFFFFF",

  success: "#10B981",
  error: "#EF4444",
  warning: "#F59E0B",
  info: "#3B82F6",
} as const;

export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  xxl: 24,
  xxxl: 32,
} as const;

export const fontSize = {
  xs: 11,
  sm: 13,
  md: 15,
  lg: 17,
  xl: 20,
  xxl: 24,
  title: 28,
} as const;

export const radius = {
  sm: 6,
  md: 10,
  lg: 14,
  xl: 20,
  full: 999,
} as const;
