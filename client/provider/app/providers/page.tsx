import { fetchProviders, fetchQualityByProvider } from "@/lib/api";
import Link from "next/link";
import ScoreBadge from "@/components/ScoreBadge";

export const dynamic = "force-dynamic";

export default async function ProvidersPage() {
  const [providers, providerQuality] = await Promise.all([
    fetchProviders().catch(() => []),
    fetchQualityByProvider("latest").catch(() => []),
  ]);

  const qualityMap = Object.fromEntries(
    providerQuality.map((pq) => [pq.provider_id, pq])
  );

  return (
    <div className="p-8 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Providers</h1>
        <p className="text-gray-500 text-sm mt-1">
          {providers.length} providers
        </p>
      </div>

      <div className="grid grid-cols-3 gap-5">
        {providers.map((p) => {
          const pq = qualityMap[p.id];
          return (
            <Link
              key={p.id}
              href={`/providers/${p.id}`}
              className="bg-white rounded-xl p-5 shadow-sm border border-gray-100 hover:shadow-md transition-all block"
              style={{ borderColor: undefined }}
            >
              <div className="flex items-start justify-between mb-3">
                <div
                  className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold text-sm"
                  style={{ background: "var(--brand-accent)" }}
                >
                  {(p.name ?? p.id).charAt(0).toUpperCase()}
                </div>
                <ScoreBadge score={pq?.average ?? p.latest_score} />
              </div>
              <div className="font-semibold text-gray-900 text-sm">
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
