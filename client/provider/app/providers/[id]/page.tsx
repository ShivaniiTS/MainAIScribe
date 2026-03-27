"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import {
  fetchProvider,
  fetchProviderTrend,
  fetchSamples,
} from "@/lib/api";
import type { SampleSummary } from "@/lib/api";
import ProviderTrendChart from "@/components/ProviderTrendChart";
import SamplesTable from "@/components/SamplesTable";

interface ProviderProfile {
  id: string;
  name?: string;
  credentials?: string;
  specialty?: string;
  practice_id?: string;
  style_directives?: string[];
  custom_vocabulary?: string[];
  template_routing?: Record<string, string>;
  quality_scores?: Record<string, number>;
  latest_score?: number;
}

export default function ProviderDetailPage() {
  const params = useParams();
  const id = params.id as string;

  const [provider, setProvider] = useState<ProviderProfile | null>(null);
  const [trendData, setTrendData] = useState<{
    trend: Array<{ version: string; score: number; date: string | null; samples: number | null }>;
  }>({ trend: [] });
  const [providerSamples, setProviderSamples] = useState<SampleSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetchProvider(id).catch(() => null),
      fetchProviderTrend(id).catch(() => ({ trend: [] })),
      fetchSamples().catch(() => []),
    ]).then(([rawProvider, trend, allSamples]) => {
      if (rawProvider) {
        setProvider(rawProvider as unknown as ProviderProfile);
      }
      setTrendData(trend);
      setProviderSamples(allSamples.filter((s) => s.physician === id));
      setLoading(false);
    });
  }, [id]);

  if (loading) return <div className="p-8 text-gray-400">Loading...</div>;
  if (!provider) return <div className="p-8 text-gray-400">Provider not found.</div>;

  const p = provider;

  return (
    <div className="p-8 space-y-6">
      <div className="flex items-center gap-3 mb-1">
        <a href="/providers" className="text-sm text-gray-400 hover:text-gray-600">
          &larr; Providers
        </a>
      </div>

      {/* Header */}
      <div className="flex items-start gap-5">
        <div
          className="w-14 h-14 rounded-full flex items-center justify-center text-white font-bold text-lg flex-shrink-0"
          style={{ background: "var(--brand-accent)" }}
        >
          {(p.name ?? p.id ?? "?").charAt(0).toUpperCase()}
        </div>
        <div>
          <h1 className="text-2xl font-bold text-gray-900">
            {p.name ?? p.id}{" "}
            {p.credentials && (
              <span className="text-gray-400 font-normal text-lg">{p.credentials}</span>
            )}
          </h1>
          <p className="text-gray-500 text-sm mt-0.5 capitalize">{p.specialty ?? "General"}</p>
          <div className="flex items-center gap-3 mt-2">
            <span className="text-xs text-gray-400">{providerSamples.length} encounters</span>
            <span className="text-xs text-gray-400">
              {providerSamples.filter((s) => s.mode === "dictation").length} dictation
            </span>
            <span className="text-xs text-gray-400">
              {providerSamples.filter((s) => s.mode === "ambient").length} ambient
            </span>
          </div>
        </div>
      </div>

      {/* Quality trend */}
      {trendData.trend.length > 0 && <ProviderTrendChart trend={trendData.trend} />}

      {/* Encounters */}
      {providerSamples.length > 0 && (
        <div>
          <h2 className="text-sm font-semibold text-gray-800 mb-3">
            Encounters ({providerSamples.length})
          </h2>
          <SamplesTable samples={providerSamples} />
        </div>
      )}
    </div>
  );
}
