"use client";

import { useEffect, useState } from "react";
import { fetchFeatures, type FeatureFlags } from "@/lib/api";

// Default features: everything enabled (fallback before server responds)
const ALL_ENABLED: FeatureFlags = {
  dashboard: true,
  view_encounters: true,
  view_providers: true,
  view_specialties: true,
  view_templates: true,
  view_quality: true,
  record_audio: true,
  trigger_pipeline: true,
  run_pipeline: true,
  batch_processing: true,
  ehr_access: true,
  patient_search: true,
  create_providers: true,
  edit_providers: true,
  create_templates: true,
  edit_templates: true,
  create_specialties: true,
  edit_specialties: true,
};

let _cached: FeatureFlags | null = null;

export function useFeatures(): FeatureFlags {
  const [features, setFeatures] = useState<FeatureFlags>(_cached ?? ALL_ENABLED);

  useEffect(() => {
    if (_cached) {
      setFeatures(_cached);
      return;
    }
    fetchFeatures()
      .then((f) => {
        _cached = f;
        setFeatures(f);
      })
      .catch(() => {
        // On error, default to all enabled
      });
  }, []);

  return features;
}
