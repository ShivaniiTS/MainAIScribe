import { fetchProviders, fetchQualityByProvider } from "@/lib/api";
import Link from "next/link";
import ScoreBadge from "@/components/ScoreBadge";
import FeatureGate from "@/components/FeatureGate";

export const dynamic = "force-dynamic";

export default async function ProvidersPage() {
  const [providers, providerQuality] = await Promise.all([
    fetchProviders().catch(() => []),
    fetchQualityByProvider("latest").catch(() => []),
  ]);

  // Merge quality data into providers
  const qualityMap = Object.fromEntries(
    providerQuality.map((pq) => [pq.provider_id, pq])
  );

  return (
    <div className="p-8 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Providers</h1>
          <p className="text-gray-500 text-sm mt-1">
            {providers.length} providers across all encounters
          </p>
        </div>
        <FeatureGate feature="create_providers">
          <Link
            href="/providers/new"
            className="px-4 py-2 text-sm font-medium text-white rounded-lg"
            style={{ background: "var(--brand-green)" }}
          >
            + New Provider
          </Link>
        </FeatureGate>
      </div>

      <div className="grid grid-cols-3 gap-5">
        {providers.map((p) => {
          const pq = qualityMap[p.id];
          return (
            <Link
              key={p.id}
              href={`/providers/${p.id}`}
              className="bg-white rounded-xl p-5 shadow-sm border border-gray-100 hover:border-green-200 hover:shadow-md transition-all block"
            >
              <div className="flex items-start justify-between mb-3">
                <div
                  className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold text-sm"
                  style={{ background: "var(--brand-indigo)" }}
                >
                  {(p.name ?? p.id).charAt(0).toUpperCase()}
                </div>
                <ScoreBadge score={pq?.average ?? p.latest_score} />
              </div>
              <div className="font-semibold text-gray-900 text-sm">
                {(p.credentials === "TEST" || p.id.includes("test")) && (
                  <span className="inline-block mr-1.5 px-1.5 py-0.5 rounded text-[10px] font-bold bg-amber-100 text-amber-700 align-middle">
                    TEST
                  </span>
                )}
                {p.name ?? p.id}
                {p.credentials && (
                  <span className="text-gray-400 font-normal ml-1 text-xs">
                    {p.credentials}
                  </span>
                )}
              </div>
              <div className="text-xs text-gray-500 mt-0.5 capitalize">
                {p.specialty ?? "General"}
              </div>
              {pq && (
                <div className="mt-3 flex flex-wrap gap-2 text-xs text-gray-500">
                  <span className="px-2 py-0.5 rounded-full bg-gray-50">
                    {pq.sample_count} samples
                  </span>
                  <span className="px-2 py-0.5 rounded-full bg-gray-50">
                    min {pq.min?.toFixed(2) ?? "—"}
                  </span>
                  <span className="px-2 py-0.5 rounded-full bg-gray-50">
                    max {pq.max?.toFixed(2) ?? "—"}
                  </span>
                </div>
              )}
              {Object.keys(p.quality_scores).length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {Object.entries(p.quality_scores).map(([v, s]) => (
                    <span
                      key={v}
                      className="text-xs px-2 py-0.5 rounded-full bg-gray-50 text-gray-500"
                    >
                      {v}: {(s as number).toFixed(2)}
                    </span>
                  ))}
                </div>
              )}
            </Link>
          );
        })}
        {providers.length === 0 && (
          <div className="col-span-3 text-center py-12 text-gray-400 text-sm">
            No providers found.
          </div>
        )}
      </div>
    </div>
  );
}
