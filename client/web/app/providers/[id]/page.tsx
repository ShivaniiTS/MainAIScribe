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
import ProviderEditForm from "@/components/ProviderEditForm";
import { useFeatures } from "@/lib/useFeatures";

interface ProviderProfile {
  id: string;
  name?: string;
  credentials?: string;
  specialty?: string;
  practice_id?: string;
  note_format?: string;
  noise_suppression_level?: string;
  postprocessor_mode?: string;
  style_directives?: string[];
  custom_vocabulary?: string[];
  template_routing?: Record<string, string>;
  quality_scores?: Record<string, number>;
  latest_score?: number;
  quality_history?: Array<{ date?: string; version?: string; score?: number; samples?: number }>;
}

export default function ProviderDetailPage() {
  const params = useParams();
  const id = params.id as string;

  const [provider, setProvider] = useState<ProviderProfile | null>(null);
  const [trendData, setTrendData] = useState<{
    trend: Array<{ version: string; score: number; date: string | null; samples: number | null }>;
  }>({ trend: [] });
  const [providerSamples, setProviderSamples] = useState<SampleSummary[]>([]);
  const [editing, setEditing] = useState(false);
  const [loading, setLoading] = useState(true);
  const features = useFeatures();

  const loadData = () => {
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
  };

  useEffect(() => {
    loadData();
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
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-5">
          <div
            className="w-14 h-14 rounded-full flex items-center justify-center text-white font-bold text-lg flex-shrink-0"
            style={{ background: "var(--brand-indigo)" }}
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
        {features.edit_providers && (
          <button
            onClick={() => setEditing(!editing)}
            className="px-4 py-2 text-sm font-medium rounded-lg"
            style={{
              background: editing ? "#F3F4F6" : "var(--brand-green)",
              color: editing ? "#374151" : "white",
            }}
          >
            {editing ? "Cancel" : "Edit Provider"}
          </button>
        )}
      </div>

      {editing ? (
        <ProviderEditForm
          provider={p}
          onSaved={() => {
            setEditing(false);
            loadData();
          }}
        />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-6">
            {/* Style directives */}
            {p.style_directives && p.style_directives.length > 0 && (
              <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100">
                <h2 className="text-sm font-semibold text-gray-800 mb-3">Style Directives</h2>
                <ul className="space-y-2">
                  {p.style_directives.map((d, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
                      <span
                        className="w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0"
                        style={{ background: "var(--brand-green)" }}
                      />
                      {d}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Custom vocabulary */}
            {p.custom_vocabulary && p.custom_vocabulary.length > 0 && (
              <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100">
                <h2 className="text-sm font-semibold text-gray-800 mb-3">
                  Custom Vocabulary ({p.custom_vocabulary.length} terms)
                </h2>
                <div className="flex flex-wrap gap-2">
                  {p.custom_vocabulary.map((t) => (
                    <span
                      key={t}
                      className="text-xs px-2.5 py-1 rounded-full font-mono"
                      style={{ background: "#EEF2FF", color: "var(--brand-indigo)" }}
                    >
                      {t}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Quality trend chart */}
          {trendData.trend.length > 0 && <ProviderTrendChart trend={trendData.trend} />}

          {/* Template routing */}
          {p.template_routing && (
            <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100">
              <h2 className="text-sm font-semibold text-gray-800 mb-3">Template Routing</h2>
              <div className="grid grid-cols-3 gap-3">
                {Object.entries(p.template_routing).map(([visit, tpl]) => (
                  <div key={visit} className="text-sm">
                    <div className="text-xs text-gray-400 capitalize">
                      {visit.replace(/_/g, " ")}
                    </div>
                    <div className="font-mono text-gray-700 text-xs mt-0.5">{tpl as string}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Provider's samples */}
          {providerSamples.length > 0 && (
            <div>
              <h2 className="text-sm font-semibold text-gray-800 mb-3">
                Encounters ({providerSamples.length})
              </h2>
              <SamplesTable samples={providerSamples} />
            </div>
          )}
        </>
      )}
    </div>
  );
}
