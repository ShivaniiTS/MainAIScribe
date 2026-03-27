import {
  fetchAggregate,
  fetchTrend,
  fetchDimensions,
  fetchSamples,
  fetchQualityByMode,
  fetchQualityByProvider,
} from "@/lib/api";
import DashboardCharts from "@/components/DashboardCharts";
import ScoreBadge from "@/components/ScoreBadge";
import Link from "next/link";

export const dynamic = "force-dynamic";

export default async function Dashboard() {
  const [rawAgg, trendData, dims, samples, byMode, byProvider] = await Promise.all([
    fetchAggregate("latest").catch(() => null),
    fetchTrend().catch(() => ({ trend: [] })),
    fetchDimensions("latest").catch(() => []),
    fetchSamples().catch(() => []),
    fetchQualityByMode("latest").catch(() => ({})),
    fetchQualityByProvider("latest").catch(() => []),
  ]);

  // Guard against empty aggregate ({} from API when no quality data exists)
  const agg = rawAgg && typeof rawAgg.average === "number" ? rawAgg : null;

  const modeData = byMode as Record<string, typeof agg>;
  const dictationAgg = modeData?.["dictation"] ?? null;
  const ambientAgg = modeData?.["ambient"] ?? null;

  const kpis = [
    {
      label: "Total Samples",
      value: samples.length,
      sub: `${samples.filter((s) => s.mode === "dictation").length} dictation · ${samples.filter((s) => s.mode === "ambient").length} ambient`,
      color: "var(--brand-green)",
    },
    {
      label: "Avg Quality Score",
      value: agg?.average != null ? `${agg.average.toFixed(2)} / 5.0` : "—",
      sub: `${agg?.sample_count ?? 0} samples evaluated`,
      color: "var(--brand-indigo)",
    },
    {
      label: "Dictation Avg",
      value: dictationAgg?.average != null ? `${dictationAgg.average.toFixed(2)} / 5.0` : "—",
      sub: dictationAgg?.sample_count ? `${dictationAgg.sample_count} samples` : "no data",
      color: "#6366F1",
    },
    {
      label: "Ambient Avg",
      value: ambientAgg?.average != null ? `${ambientAgg.average.toFixed(2)} / 5.0` : "—",
      sub: ambientAgg?.sample_count ? `${ambientAgg.sample_count} samples` : "no data",
      color: "#10B981",
    },
  ];

  // Top samples by quality
  const topSamples = [...samples]
    .filter((s) => s.quality?.overall != null)
    .sort((a, b) => (b.quality!.overall ?? 0) - (a.quality!.overall ?? 0))
    .slice(0, 8);

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-500 text-sm mt-1">
          AI Scribe pipeline quality overview · {agg?.version ?? "—"} · {byProvider.length} providers
        </p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-4 gap-5">
        {kpis.map((k) => (
          <div
            key={k.label}
            className="bg-white rounded-xl p-5 shadow-sm border border-gray-100"
          >
            <div
              className="w-2 h-2 rounded-full mb-3"
              style={{ background: k.color }}
            />
            <div className="text-2xl font-bold text-gray-900">{k.value}</div>
            <div className="text-xs font-medium text-gray-600 mt-1">{k.label}</div>
            <div className="text-xs text-gray-400 mt-0.5">{k.sub}</div>
          </div>
        ))}
      </div>

      {/* Charts row */}
      <DashboardCharts trend={trendData.trend} dimensions={dims} />

      {/* Provider quality breakdown */}
      {byProvider.length > 0 && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-100">
          <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between">
            <h2 className="font-semibold text-gray-800 text-sm">Quality by Provider</h2>
            <Link
              href="/providers"
              className="text-xs font-medium"
              style={{ color: "var(--brand-green)" }}
            >
              View all &rarr;
            </Link>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-gray-400 border-b border-gray-50">
                  <th className="px-6 py-3 font-medium">Provider</th>
                  <th className="px-4 py-3 font-medium">Samples</th>
                  <th className="px-4 py-3 font-medium">Avg Score</th>
                  <th className="px-4 py-3 font-medium">Min</th>
                  <th className="px-4 py-3 font-medium">Max</th>
                  <th className="px-4 py-3 font-medium"></th>
                </tr>
              </thead>
              <tbody>
                {byProvider.map((pq) => (
                  <tr
                    key={pq.provider_id}
                    className="border-b border-gray-50 hover:bg-gray-50 transition-colors"
                  >
                    <td className="px-6 py-3 font-medium text-gray-800 text-xs">
                      {pq.provider_name}
                    </td>
                    <td className="px-4 py-3 text-gray-600 text-xs">{pq.sample_count}</td>
                    <td className="px-4 py-3">
                      <ScoreBadge score={pq.average} />
                    </td>
                    <td className="px-4 py-3 text-gray-600 text-xs">{pq.min?.toFixed(2) ?? "—"}</td>
                    <td className="px-4 py-3 text-gray-600 text-xs">{pq.max?.toFixed(2) ?? "—"}</td>
                    <td className="px-4 py-3">
                      <Link
                        href={`/providers/${pq.provider_id}`}
                        className="text-xs font-medium hover:underline"
                        style={{ color: "var(--brand-green)" }}
                      >
                        View &rarr;
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Mode breakdown — dimension comparison */}
      {dictationAgg && ambientAgg && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
          <h2 className="font-semibold text-gray-800 text-sm mb-4">
            Quality by Mode — Dimension Breakdown
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-gray-400 border-b border-gray-50">
                  <th className="py-2 font-medium">Dimension</th>
                  <th className="py-2 font-medium px-4">Dictation ({dictationAgg.sample_count})</th>
                  <th className="py-2 font-medium px-4">Ambient ({ambientAgg.sample_count})</th>
                  <th className="py-2 font-medium px-4">Delta</th>
                </tr>
              </thead>
              <tbody>
                {["accuracy", "completeness", "no_hallucination", "structure", "language"].map(
                  (dim) => {
                    const dVal = dictationAgg.dimensions?.[dim] ?? 0;
                    const aVal = ambientAgg.dimensions?.[dim] ?? 0;
                    const delta = dVal && aVal ? dVal - aVal : null;
                    const dimLabel = dim
                      .replace("no_hallucination", "No Hallucination")
                      .replace(/_/g, " ")
                      .replace(/\b\w/g, (c) => c.toUpperCase());
                    return (
                      <tr key={dim} className="border-b border-gray-50">
                        <td className="py-2 text-gray-700 text-xs">{dimLabel}</td>
                        <td className="py-2 px-4 text-gray-800 text-xs font-medium">
                          {dVal ? dVal.toFixed(2) : "—"}
                        </td>
                        <td className="py-2 px-4 text-gray-800 text-xs font-medium">
                          {aVal ? aVal.toFixed(2) : "—"}
                        </td>
                        <td className="py-2 px-4 text-xs font-medium">
                          {delta !== null ? (
                            <span
                              style={{
                                color: delta >= 0 ? "#10B981" : "#EF4444",
                              }}
                            >
                              {delta >= 0 ? "+" : ""}
                              {delta.toFixed(2)}
                            </span>
                          ) : (
                            "—"
                          )}
                        </td>
                      </tr>
                    );
                  }
                )}
                <tr className="border-t border-gray-200">
                  <td className="py-2 text-gray-900 text-xs font-semibold">Overall</td>
                  <td className="py-2 px-4 text-xs font-bold">
                    <ScoreBadge score={dictationAgg.average} />
                  </td>
                  <td className="py-2 px-4 text-xs font-bold">
                    <ScoreBadge score={ambientAgg.average} />
                  </td>
                  <td className="py-2 px-4 text-xs font-bold">
                    {(() => {
                      const d = dictationAgg.average - ambientAgg.average;
                      return (
                        <span style={{ color: d >= 0 ? "#10B981" : "#EF4444" }}>
                          {d >= 0 ? "+" : ""}
                          {d.toFixed(2)}
                        </span>
                      );
                    })()}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Top samples table */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100">
        <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between">
          <h2 className="font-semibold text-gray-800 text-sm">Top Samples by Quality</h2>
          <Link
            href="/samples"
            className="text-xs font-medium"
            style={{ color: "var(--brand-green)" }}
          >
            View all &rarr;
          </Link>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-gray-400 border-b border-gray-50">
                <th className="px-6 py-3 font-medium">Sample</th>
                <th className="px-4 py-3 font-medium">Provider</th>
                <th className="px-4 py-3 font-medium">Mode</th>
                <th className="px-4 py-3 font-medium">Overall</th>
                <th className="px-4 py-3 font-medium">Accuracy</th>
                <th className="px-4 py-3 font-medium">Completeness</th>
                <th className="px-4 py-3 font-medium">Structure</th>
                <th className="px-4 py-3 font-medium"></th>
              </tr>
            </thead>
            <tbody>
              {topSamples.map((s) => (
                <tr
                  key={s.sample_id}
                  className="border-b border-gray-50 hover:bg-gray-50 transition-colors"
                >
                  <td className="px-6 py-3 font-mono text-xs text-gray-700 max-w-[200px] truncate">
                    {s.sample_id}
                  </td>
                  <td className="px-4 py-3 text-xs text-gray-600">
                    {s.physician
                      ?.replace(/^dr_/, "Dr. ")
                      .replace(/_/g, " ")
                      .replace(/\b\w/g, (c) => c.toUpperCase())
                      .replace(/^Dr\. /, "Dr. ")}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className="text-xs px-2 py-0.5 rounded-full font-medium"
                      style={{
                        background:
                          s.mode === "dictation" ? "#EEF2FF" : "#E6F7F2",
                        color:
                          s.mode === "dictation"
                            ? "var(--brand-indigo)"
                            : "var(--brand-green-dark)",
                      }}
                    >
                      {s.mode}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <ScoreBadge score={s.quality?.overall} />
                  </td>
                  <td className="px-4 py-3 text-gray-600">
                    {s.quality?.accuracy?.toFixed(1) ?? "—"}
                  </td>
                  <td className="px-4 py-3 text-gray-600">
                    {s.quality?.completeness?.toFixed(1) ?? "—"}
                  </td>
                  <td className="px-4 py-3 text-gray-600">
                    {s.quality?.structure?.toFixed(1) ?? "—"}
                  </td>
                  <td className="px-4 py-3">
                    <Link
                      href={`/samples/${s.sample_id}`}
                      className="text-xs font-medium hover:underline"
                      style={{ color: "var(--brand-green)" }}
                    >
                      View &rarr;
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
