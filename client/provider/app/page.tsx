import {
  fetchAggregate,
  fetchTrend,
  fetchDimensions,
  fetchSamples,
  fetchQualityByProvider,
} from "@/lib/api";
import DashboardCharts from "@/components/DashboardCharts";
import ScoreBadge from "@/components/ScoreBadge";
import Link from "next/link";

export const dynamic = "force-dynamic";

export default async function Dashboard() {
  const [rawAgg, trendData, dims, samples, byProvider] = await Promise.all([
    fetchAggregate("latest").catch(() => null),
    fetchTrend().catch(() => ({ trend: [] })),
    fetchDimensions("latest").catch(() => []),
    fetchSamples().catch(() => []),
    fetchQualityByProvider("latest").catch(() => []),
  ]);

  // Guard against empty aggregate ({} from API when no quality data exists)
  const agg = rawAgg && typeof rawAgg.average === "number" ? rawAgg : null;

  const kpis = [
    {
      label: "Total Encounters",
      value: samples.length,
      sub: `${samples.filter((s) => s.mode === "dictation").length} dictation \u00B7 ${samples.filter((s) => s.mode === "ambient").length} ambient`,
      color: "var(--brand-primary)",
    },
    {
      label: "Avg Quality Score",
      value: agg?.average != null ? `${agg.average.toFixed(2)} / 5.0` : "\u2014",
      sub: `${agg?.sample_count ?? 0} samples evaluated`,
      color: "var(--brand-accent)",
    },
    {
      label: "Providers",
      value: byProvider.length,
      sub: "active providers",
      color: "#6366F1",
    },
    {
      label: "Latest Version",
      value: trendData.trend.length > 0 ? trendData.trend[trendData.trend.length - 1].version : "\u2014",
      sub: trendData.trend.length > 0 ? `${trendData.trend[trendData.trend.length - 1].average?.toFixed(2) ?? "—"} avg` : "",
      color: "#10B981",
    },
  ];

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-500 text-sm mt-1">
          Documentation quality overview
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

      {/* Charts */}
      <DashboardCharts trend={trendData.trend} dimensions={dims} />

      {/* Provider quality table */}
      {byProvider.length > 0 && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-100">
          <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between">
            <h2 className="font-semibold text-gray-800 text-sm">Quality by Provider</h2>
            <Link
              href="/providers"
              className="text-xs font-medium"
              style={{ color: "var(--brand-primary)" }}
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
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
