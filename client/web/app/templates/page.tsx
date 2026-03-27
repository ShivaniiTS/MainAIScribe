import { fetchTemplates } from "@/lib/api";
import Link from "next/link";
import FeatureGate from "@/components/FeatureGate";

export const dynamic = "force-dynamic";

const SPECIALTY_COLORS: Record<string, string> = {
  orthopedic: "#3B82F6",
  chiropractic: "#8B5CF6",
  neurology: "#EC4899",
  cardiology: "#EF4444",
  gastroenterology: "#F59E0B",
};

export default async function TemplatesPage() {
  const templates = await fetchTemplates().catch(() => []);

  return (
    <div className="p-8 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Templates</h1>
          <p className="text-gray-500 text-sm mt-1">
            {templates.length} note templates across all specialties
          </p>
        </div>
        <FeatureGate feature="create_templates">
          <Link
            href="/templates/new"
            className="px-4 py-2 text-sm font-medium text-white rounded-lg"
            style={{ background: "var(--brand-green)" }}
          >
            + New Template
          </Link>
        </FeatureGate>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-100">
              <th className="text-left px-5 py-3 text-gray-500 font-medium">Template</th>
              <th className="text-left px-5 py-3 text-gray-500 font-medium">Specialty</th>
              <th className="text-left px-5 py-3 text-gray-500 font-medium">Visit Type</th>
              <th className="text-center px-5 py-3 text-gray-500 font-medium">Sections</th>
              <th className="text-left px-5 py-3 text-gray-500 font-medium">Used By</th>
            </tr>
          </thead>
          <tbody>
            {templates.map((t) => (
              <tr key={t.id} className="border-b border-gray-50 hover:bg-gray-50/50">
                <td className="px-5 py-3">
                  <Link
                    href={`/templates/${t.id}`}
                    className="font-medium text-gray-900 hover:text-green-600"
                  >
                    {t.name}
                  </Link>
                  <div className="text-xs text-gray-400 font-mono mt-0.5">{t.id}</div>
                </td>
                <td className="px-5 py-3">
                  <span
                    className="text-xs px-2 py-0.5 rounded-full font-medium capitalize"
                    style={{
                      background: `${SPECIALTY_COLORS[t.specialty] || "#6B7280"}15`,
                      color: SPECIALTY_COLORS[t.specialty] || "#6B7280",
                    }}
                  >
                    {t.specialty || "general"}
                  </span>
                </td>
                <td className="px-5 py-3 text-gray-600 capitalize">
                  {t.visit_type.replace(/_/g, " ")}
                </td>
                <td className="px-5 py-3 text-center text-gray-600">{t.section_count}</td>
                <td className="px-5 py-3">
                  {t.providers.length > 0 ? (
                    <div className="flex flex-wrap gap-1">
                      {t.providers.map((pid) => (
                        <span
                          key={pid}
                          className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-500"
                        >
                          {pid.replace("dr_", "Dr. ").replace(/_/g, " ")}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <span className="text-xs text-gray-300">none</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {templates.length === 0 && (
          <div className="text-center py-12 text-gray-400 text-sm">
            No templates found.
          </div>
        )}
      </div>
    </div>
  );
}
