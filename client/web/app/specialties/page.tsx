import { fetchSpecialties } from "@/lib/api";
import Link from "next/link";
import FeatureGate from "@/components/FeatureGate";

export const dynamic = "force-dynamic";

export default async function SpecialtiesPage() {
  const specialties = await fetchSpecialties().catch(() => []);

  return (
    <div className="p-8 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Specialties</h1>
          <p className="text-gray-500 text-sm mt-1">
            {specialties.length} specialties with keyword dictionaries
          </p>
        </div>
        <FeatureGate feature="create_specialties">
          <Link
            href="/specialties/new"
            className="px-4 py-2 text-sm font-medium text-white rounded-lg"
            style={{ background: "var(--brand-green)" }}
          >
            + New Specialty
          </Link>
        </FeatureGate>
      </div>

      <div className="grid grid-cols-3 gap-5">
        {specialties.map((s) => (
          <Link
            key={s.id}
            href={`/specialties/${s.id}`}
            className="bg-white rounded-xl p-5 shadow-sm border border-gray-100 hover:border-green-200 hover:shadow-md transition-all block"
          >
            <div className="flex items-start justify-between mb-3">
              <div
                className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold text-sm"
                style={{ background: "var(--brand-indigo)" }}
              >
                {s.name.charAt(0).toUpperCase()}
              </div>
              <span className="text-xs px-2.5 py-1 rounded-full bg-green-50 text-green-700 font-medium">
                {s.term_count} terms
              </span>
            </div>
            <div className="font-semibold text-gray-900 text-sm capitalize">
              {s.name}
            </div>
            <div className="text-xs text-gray-400 mt-1 font-mono">{s.id}.txt</div>
          </Link>
        ))}
        {specialties.length === 0 && (
          <div className="col-span-3 text-center py-12 text-gray-400 text-sm">
            No specialties found. Create one to get started.
          </div>
        )}
      </div>
    </div>
  );
}
