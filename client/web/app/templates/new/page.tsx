"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { createTemplate, fetchSpecialties } from "@/lib/api";
import type { SpecialtySummary } from "@/lib/api";
import { useFeatures } from "@/lib/useFeatures";

const COMMON_SECTIONS = [
  { id: "chief_complaint", label: "Chief Complaint" },
  { id: "history_of_present_illness", label: "History of Present Illness" },
  { id: "past_medical_history", label: "Past Medical History" },
  { id: "past_surgical_history", label: "Past Surgical History" },
  { id: "medications", label: "Current Medications" },
  { id: "allergies", label: "Allergies" },
  { id: "social_history", label: "Social History" },
  { id: "review_of_systems", label: "Review of Systems" },
  { id: "physical_examination", label: "Physical Examination" },
  { id: "imaging", label: "Imaging / Diagnostics" },
  { id: "assessment", label: "Assessment" },
  { id: "plan", label: "Plan" },
];

const HEADER_FIELDS = [
  "patient_name", "date_of_birth", "date_of_service", "date_of_injury",
  "mechanism_of_injury", "provider_name", "referring_provider", "location",
];

interface SectionDraft {
  id: string;
  label: string;
  required: boolean;
  prompt_hint: string;
}

export default function NewTemplatePage() {
  const router = useRouter();
  const features = useFeatures();

  if (!features.create_templates) {
    return (
      <div className="p-8 text-gray-500">
        Creating templates is not available on this server.
      </div>
    );
  }
  const [specialties, setSpecialties] = useState<SpecialtySummary[]>([]);
  const [templateId, setTemplateId] = useState("");
  const [name, setName] = useState("");
  const [specialty, setSpecialty] = useState("");
  const [visitType, setVisitType] = useState("follow_up");
  const [headerFields, setHeaderFields] = useState<string[]>(["patient_name", "date_of_service", "provider_name"]);
  const [sections, setSections] = useState<SectionDraft[]>([
    { id: "chief_complaint", label: "Chief Complaint", required: true, prompt_hint: "" },
    { id: "history_of_present_illness", label: "History of Present Illness", required: true, prompt_hint: "" },
    { id: "assessment", label: "Assessment", required: true, prompt_hint: "Numbered list of diagnoses" },
    { id: "plan", label: "Plan", required: true, prompt_hint: "Numbered plan items matching assessment" },
  ]);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetchSpecialties().then(setSpecialties).catch(() => []);
  }, []);

  const addSection = (preset?: { id: string; label: string }) => {
    setSections([
      ...sections,
      {
        id: preset?.id || "",
        label: preset?.label || "",
        required: true,
        prompt_hint: "",
      },
    ]);
  };

  const removeSection = (idx: number) => {
    setSections(sections.filter((_, i) => i !== idx));
  };

  const updateSection = (idx: number, field: keyof SectionDraft, val: string | boolean) => {
    const next = [...sections];
    next[idx] = { ...next[idx], [field]: val };
    setSections(next);
  };

  const toggleHeader = (field: string) => {
    setHeaderFields((prev) =>
      prev.includes(field) ? prev.filter((f) => f !== field) : [...prev, field]
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (!templateId.trim() || !name.trim() || !specialty) {
      setError("Template ID, name, and specialty are required");
      return;
    }
    if (sections.length === 0) {
      setError("At least one section is required");
      return;
    }
    setSaving(true);
    try {
      await createTemplate({
        id: templateId.trim().toLowerCase().replace(/\s+/g, "_"),
        name,
        specialty,
        visit_type: visitType,
        header_fields: headerFields,
        sections,
      });
      router.push("/templates");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create");
    } finally {
      setSaving(false);
    }
  };

  // Available sections not yet added
  const availablePresets = COMMON_SECTIONS.filter(
    (cs) => !sections.some((s) => s.id === cs.id)
  );

  return (
    <div className="p-8 max-w-3xl space-y-6">
      <div>
        <a href="/templates" className="text-sm text-gray-400 hover:text-gray-600">
          &larr; Templates
        </a>
        <h1 className="text-2xl font-bold text-gray-900 mt-2">New Template</h1>
      </div>

      <form onSubmit={handleSubmit} className="space-y-5">
        {/* Basic info */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 space-y-4">
          <h2 className="text-sm font-semibold text-gray-800">Basic Information</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-gray-500 mb-1">Template ID</label>
              <input
                type="text"
                value={templateId}
                onChange={(e) => setTemplateId(e.target.value)}
                placeholder="e.g. derm_follow_up"
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-200"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 mb-1">Display Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Dermatology Follow-Up"
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-200"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 mb-1">Specialty</label>
              <select
                value={specialty}
                onChange={(e) => setSpecialty(e.target.value)}
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-200"
              >
                <option value="">Select specialty...</option>
                {specialties.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 mb-1">Visit Type</label>
              <select
                value={visitType}
                onChange={(e) => setVisitType(e.target.value)}
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-200"
              >
                <option value="initial_evaluation">Initial Evaluation</option>
                <option value="follow_up">Follow-Up</option>
                <option value="discharge">Discharge</option>
                <option value="consultation">Consultation</option>
                <option value="procedure_note">Procedure Note</option>
              </select>
            </div>
          </div>
        </div>

        {/* Header fields */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 space-y-3">
          <h2 className="text-sm font-semibold text-gray-800">Header Fields</h2>
          <div className="flex flex-wrap gap-2">
            {HEADER_FIELDS.map((f) => (
              <button
                type="button"
                key={f}
                onClick={() => toggleHeader(f)}
                className="text-xs px-3 py-1.5 rounded-full border transition-colors capitalize"
                style={{
                  background: headerFields.includes(f) ? "var(--brand-green)" : "white",
                  color: headerFields.includes(f) ? "white" : "#6B7280",
                  borderColor: headerFields.includes(f) ? "var(--brand-green)" : "#E5E7EB",
                }}
              >
                {f.replace(/_/g, " ")}
              </button>
            ))}
          </div>
        </div>

        {/* Sections */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-gray-800">Sections ({sections.length})</h2>
          </div>

          {sections.map((sec, idx) => (
            <div key={idx} className="border border-gray-100 rounded-lg p-3 space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-300 w-5">#{idx + 1}</span>
                <input
                  type="text"
                  value={sec.id}
                  onChange={(e) => updateSection(idx, "id", e.target.value)}
                  placeholder="section_id"
                  className="w-40 border border-gray-200 rounded px-2 py-1 text-xs font-mono"
                />
                <input
                  type="text"
                  value={sec.label}
                  onChange={(e) => updateSection(idx, "label", e.target.value)}
                  placeholder="Section Label"
                  className="flex-1 border border-gray-200 rounded px-2 py-1 text-sm"
                />
                <label className="flex items-center gap-1 text-xs text-gray-500">
                  <input
                    type="checkbox"
                    checked={sec.required}
                    onChange={(e) => updateSection(idx, "required", e.target.checked)}
                    className="rounded"
                  />
                  Req
                </label>
                <button
                  type="button"
                  onClick={() => removeSection(idx)}
                  className="text-red-400 hover:text-red-600 text-sm px-1"
                >
                  &times;
                </button>
              </div>
              <input
                type="text"
                value={sec.prompt_hint}
                onChange={(e) => updateSection(idx, "prompt_hint", e.target.value)}
                placeholder="Prompt hint..."
                className="w-full border border-gray-200 rounded px-2 py-1 text-xs text-gray-500 ml-7"
              />
            </div>
          ))}

          {/* Quick-add presets */}
          {availablePresets.length > 0 && (
            <div className="pt-2 border-t border-gray-100">
              <p className="text-xs text-gray-400 mb-2">Quick add:</p>
              <div className="flex flex-wrap gap-1">
                {availablePresets.map((cs) => (
                  <button
                    type="button"
                    key={cs.id}
                    onClick={() => addSection(cs)}
                    className="text-xs px-2 py-1 rounded-full border border-dashed border-gray-300 text-gray-500 hover:border-green-400 hover:text-green-600"
                  >
                    + {cs.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          <button
            type="button"
            onClick={() => addSection()}
            className="text-xs text-indigo-500 hover:text-indigo-700"
          >
            + Add custom section
          </button>
        </div>

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
            {saving ? "Creating..." : "Create Template"}
          </button>
          <a
            href="/templates"
            className="px-4 py-2 text-sm font-medium text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200"
          >
            Cancel
          </a>
        </div>
      </form>
    </div>
  );
}
