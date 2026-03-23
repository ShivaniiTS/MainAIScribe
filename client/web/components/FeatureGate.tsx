"use client";

import { useFeatures } from "@/lib/useFeatures";
import type { FeatureFlags } from "@/lib/api";

interface FeatureGateProps {
  feature: keyof FeatureFlags;
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

/**
 * Conditionally render children based on a feature flag.
 * Useful in server-component pages that can't use hooks directly.
 */
export default function FeatureGate({ feature, children, fallback }: FeatureGateProps) {
  const features = useFeatures();
  if (!features[feature]) return fallback ?? null;
  return <>{children}</>;
}
