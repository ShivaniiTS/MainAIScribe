"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { fetchTemplate, updateTemplate } from "@/lib/api";
import type { TemplateDetail, TemplateSection } from "@/lib/api";
import { useFeatures } from "@/lib/useFeatures";

export default function TemplateDetailPage() {
  const params = useParams();
  const id = params.id as string;
  const features = useFeatures();

  const [template, setTemplate] = useState<TemplateDetail | null>(null);
  const [editing, setEditing] = useState(false);
  const [sections, setSections] = useState<TemplateSection[]>([]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  useEffect(() => {
    fetchTemplate(id)
      .then((t) => {
        setTemplate(t);
        setSections(t.sections);
      })
      .catch(() => setTemplate(null));
  }, [id]);

  const handleSave = async () => {
    setError("");
    setSuccess("");
    setSaving(true);
    try {
      const updated = await updateTemplate(id, { sections });
      setTemplate(updated);
      setSections(updated.sections);
      setEditing(false);
      setSuccess("Template saved");
      setTimeout(() => setSuccess(""), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save");
    } finally {
      setSaving(false);
    }
  };

  const moveSection = (idx: number, dir: -1 | 1) => {
    const next = [...sections];
    const target = idx + dir;
    if (target < 0 || target >= next.length) return;
    [next[idx], next[target]] = [next[target], next[idx]];
    setSections(next);
  };

  const removeSection = (idx: number) => {
    setSections(sections.filter((_, i) => i !== idx));
  };

  const addSection = () => {
    setSections([
      ...sections,
      { id: "", label: "", required: true, prompt_hint: "" },
    ]);
  };

  const updateSectionField = (idx: number, field: keyof TemplateSection, value: string | boolean) => {
    const next = [...sections];
    next[idx] = { ...next[idx], [field]: value };
    setSections(next);
  };

  if (template === null) {
    return <div className="p-8 text-gray-400">Loading...</div>;
  }

  return (
    <div className="p-8 space-y-6">
      <div className="flex items-center gap-3 mb-1">
        <a href="/templates" className="text-sm text-gray-400 hover:text-gray-600">
          &larr; Templates
        </a>
      </div>

      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{template.name}</h1>
          <div className="flex items-center gap-3 mt-1">
            <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-600 font-medium capitalize">
              {template.specialty}
            </span>
            <span className="text-xs text-gray-400 capitalize">
              {template.visit_type.replace(/_/g, " ")}
            </span>
            <span className="text-xs text-gray-400 font-mono">{template.id}.yaml</span>
          </div>
          {template.providers.length > 0 && (
            <div className="flex items-center gap-2 mt-2">
              <span className="text-xs text-gray-400">Used by:</span>
              {template.providers.map((pid) => (
                <a
                  key={pid}
                  href={`/providers/${pid}`}
                  className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-600 hover:bg-green-50 hover:text-green-700"
                >
                  {pid.replace("dr_", "Dr. ").replace(/_/g, " ")}
                </a>
              ))}
            </div>
          )}
        </div>
        {features.edit_templates && (
          <button
            onClick={() => setEditing(!editing)}
            className="px-4 py-2 text-sm font-medium rounded-lg"
            style={{
              background: editing ? "#F3F4F6" : "var(--brand-green)",
              color: editing ? "#374151" : "white",
            }}
          >
            {editing ? "Cancel" : "Edit Sections"}
          </button>
        )}
      </div>

      {success && (
        <div className="text-sm text-green-700 bg-green-50 px-3 py-2 rounded-lg">
          {success}
        </div>
      )}
      {error && (
        <div className="text-sm text-red-600 bg-red-50 px-3 py-2 rounded-lg">
          {error}
        </div>
      )}

      {/* Header fields */}
      {template.header_fields.length > 0 && (
        <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100">
          <h2 className="text-sm font-semibold text-gray-800 mb-3">Header Fields</h2>
          <div className="flex flex-wrap gap-2">
            {template.header_fields.map((f) => (
              <span
                key={f}
                className="text-xs px-2.5 py-1 rounded-full bg-gray-100 text-gray-600 capitalize"
              >
                {f.replace(/_/g, " ")}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Formatting */}
      {Object.keys(template.formatting).length > 0 && (
        <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100">
          <h2 className="text-sm font-semibold text-gray-800 mb-3">Formatting Rules</h2>
          <div className="grid grid-cols-3 gap-3">
            {Object.entries(template.formatting).map(([key, val]) => (
              <div key={key} className="text-sm">
                <span className="text-xs text-gray-400 capitalize">{key.replace(/_/g, " ")}</span>
                <div className="font-mono text-gray-700 text-xs mt-0.5">{val}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Sections */}
      <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-gray-800">
            Sections ({sections.length})
          </h2>
          {editing && (
            <button
              onClick={addSection}
              className="text-xs px-3 py-1 rounded-lg text-white"
              style={{ background: "var(--brand-indigo)" }}
            >
              + Add Section
            </button>
          )}
        </div>

        <div className="space-y-3">
          {sections.map((sec, idx) => (
            <div
              key={idx}
              className="border border-gray-100 rounded-lg p-4"
              style={{ background: sec.required ? "white" : "#FAFAFA" }}
            >
              {editing ? (
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <input
                      type="text"
                      value={sec.id}
                      onChange={(e) => updateSectionField(idx, "id", e.target.value)}
                      placeholder="section_id"
                      className="flex-1 border border-gray-200 rounded px-2 py-1 text-sm font-mono"
                    />
                    <input
                      type="text"
                      value={sec.label}
                      onChange={(e) => updateSectionField(idx, "label", e.target.value)}
                      placeholder="Section Label"
                      className="flex-1 border border-gray-200 rounded px-2 py-1 text-sm"
                    />
                    <label className="flex items-center gap-1 text-xs text-gray-500">
                      <input
                        type="checkbox"
                        checked={sec.required}
                        onChange={(e) => updateSectionField(idx, "required", e.target.checked)}
                        className="rounded"
                      />
                      Req
                    </label>
                    <button
                      onClick={() => moveSection(idx, -1)}
                      className="text-gray-400 hover:text-gray-600 text-xs px-1"
                      disabled={idx === 0}
                    >
                      &uarr;
                    </button>
                    <button
                      onClick={() => moveSection(idx, 1)}
                      className="text-gray-400 hover:text-gray-600 text-xs px-1"
                      disabled={idx === sections.length - 1}
                    >
                      &darr;
                    </button>
                    <button
                      onClick={() => removeSection(idx)}
                      className="text-red-400 hover:text-red-600 text-xs px-1"
                    >
                      &times;
                    </button>
                  </div>
                  <input
                    type="text"
                    value={sec.prompt_hint}
                    onChange={(e) => updateSectionField(idx, "prompt_hint", e.target.value)}
                    placeholder="Prompt hint for this section..."
                    className="w-full border border-gray-200 rounded px-2 py-1 text-xs text-gray-500"
                  />
                </div>
              ) : (
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-gray-800">{sec.label}</span>
                      <span className="text-xs font-mono text-gray-400">{sec.id}</span>
                      {sec.required && (
                        <span className="text-xs px-1.5 py-0.5 rounded bg-green-50 text-green-600">
                          required
                        </span>
                      )}
                    </div>
                    {sec.prompt_hint && (
                      <p className="text-xs text-gray-400 mt-1">{sec.prompt_hint}</p>
                    )}
                  </div>
                  <span className="text-xs text-gray-300">#{idx + 1}</span>
                </div>
              )}
            </div>
          ))}
        </div>

        {editing && (
          <div className="mt-4 flex justify-end">
            <button
              onClick={handleSave}
              disabled={saving}
              className="px-4 py-2 text-sm font-medium text-white rounded-lg disabled:opacity-50"
              style={{ background: "var(--brand-green)" }}
            >
              {saving ? "Saving..." : "Save Template"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
