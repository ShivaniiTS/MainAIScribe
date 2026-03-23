"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  createProvider,
  fetchSpecialties,
  fetchTemplates,
} from "@/lib/api";
import type { SpecialtySummary, TemplateSummary } from "@/lib/api";
import { useFeatures } from "@/lib/useFeatures";

export default function NewProviderPage() {
  const router = useRouter();
  const features = useFeatures();

  if (!features.create_providers) {
    return (
      <div className="p-8 text-gray-500">
        Creating providers is not available on this server.
      </div>
    );
  }
  const [specialties, setSpecialties] = useState<SpecialtySummary[]>([]);
  const [templates, setTemplates] = useState<TemplateSummary[]>([]);

  const [providerId, setProviderId] = useState("");
  const [name, setName] = useState("");
  const [credentials, setCredentials] = useState("");
  const [specialty, setSpecialty] = useState("");
  const [practiceId, setPracticeId] = useState("");
  const [defaultTemplate, setDefaultTemplate] = useState("");
  const [initialTemplate, setInitialTemplate] = useState("");
  const [followUpTemplate, setFollowUpTemplate] = useState("");

  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    Promise.all([fetchSpecialties(), fetchTemplates()])
      .then(([specs, tpls]) => {
        setSpecialties(specs);
        setTemplates(tpls);
      })
      .catch(() => {});
  }, []);

  const matchingTemplates = templates.filter(
    (t) => !specialty || t.specialty === specialty
  );

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    const cleanId = providerId.trim().toLowerCase().replace(/\s+/g, "_");
    if (!cleanId || !name.trim()) {
      setError("Provider ID and name are required");
      return;
    }
    if (!specialty) {
      setError("Specialty is required");
      return;
    }

    const templateRouting: Record<string, string> = {};
    if (defaultTemplate) templateRouting.default = defaultTemplate;
    if (initialTemplate) {
      templateRouting.initial = initialTemplate;
      templateRouting.initial_evaluation = initialTemplate;
    }
    if (followUpTemplate) templateRouting.follow_up = followUpTemplate;

    setSaving(true);
    try {
      await createProvider({
        id: cleanId,
        name: name.trim(),
        credentials,
        specialty,
        practice_id: practiceId,
        template_routing: templateRouting,
      });
      router.push(`/providers/${cleanId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="p-8 max-w-2xl space-y-6">
      <div>
        <a href="/providers" className="text-sm text-gray-400 hover:text-gray-600">
          &larr; Providers
        </a>
        <h1 className="text-2xl font-bold text-gray-900 mt-2">New Provider</h1>
      </div>

      <form onSubmit={handleSubmit} className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 space-y-5">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Provider ID</label>
            <input
              type="text"
              value={providerId}
              onChange={(e) => setProviderId(e.target.value)}
              placeholder="e.g. dr_jane_smith"
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-200"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Full Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Dr. Jane Smith"
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-200"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Credentials</label>
            <input
              type="text"
              value={credentials}
              onChange={(e) => setCredentials(e.target.value)}
              placeholder="MD, DO, DC, NP..."
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-200"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Specialty</label>
            <select
              value={specialty}
              onChange={(e) => {
                setSpecialty(e.target.value);
                setDefaultTemplate("");
                setInitialTemplate("");
                setFollowUpTemplate("");
              }}
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-200"
            >
              <option value="">Select specialty...</option>
              {specialties.map((s) => (
                <option key={s.id} value={s.id}>{s.name}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Practice ID</label>
            <input
              type="text"
              value={practiceId}
              onChange={(e) => setPracticeId(e.target.value)}
              placeholder="e.g. excelsia_injury_care"
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-200"
            />
          </div>
        </div>

        {/* Template routing */}
        {specialty && (
          <div className="pt-4 border-t border-gray-100 space-y-3">
            <h3 className="text-sm font-semibold text-gray-800">Template Routing</h3>
            <div className="space-y-2">
              {[
                { key: "default", label: "Default", val: defaultTemplate, set: setDefaultTemplate },
                { key: "initial", label: "Initial Evaluation", val: initialTemplate, set: setInitialTemplate },
                { key: "follow_up", label: "Follow-Up", val: followUpTemplate, set: setFollowUpTemplate },
              ].map(({ key, label, val, set }) => (
                <div key={key} className="flex items-center gap-3">
                  <span className="text-xs text-gray-500 w-32">{label}</span>
                  <select
                    value={val}
                    onChange={(e) => set(e.target.value)}
                    className="flex-1 border border-gray-200 rounded px-2 py-1 text-sm"
                  >
                    <option value="">-- not set --</option>
                    {matchingTemplates.map((t) => (
                      <option key={t.id} value={t.id}>{t.name} ({t.id})</option>
                    ))}
                  </select>
                </div>
              ))}
            </div>
            <p className="text-xs text-gray-400">
              Templates are filtered by selected specialty. Style directives and custom vocabulary can be added after creation.
            </p>
          </div>
        )}

        {error && (
          <div className="text-sm text-red-600 bg-red-50 px-3 py-2 rounded-lg">
            {error}
          </div>
        )}

        <div className="flex gap-3">
          <button
            type="submit"
            disabled={saving}
            className="px-4 py-2 text-sm font-medium text-white rounded-lg disabled:opacity-50"
            style={{ background: "var(--brand-green)" }}
          >
            {saving ? "Creating..." : "Create Provider"}
          </button>
          <a
            href="/providers"
            className="px-4 py-2 text-sm font-medium text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200"
          >
            Cancel
          </a>
        </div>
      </form>
    </div>
  );
}
