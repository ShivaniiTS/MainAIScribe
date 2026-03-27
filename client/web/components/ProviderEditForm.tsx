"use client";

import { useEffect, useState } from "react";
import {
  fetchSpecialties,
  fetchTemplates,
  updateProvider,
} from "@/lib/api";
import type { SpecialtySummary, TemplateSummary } from "@/lib/api";

interface ProviderData {
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
}

interface Props {
  provider: ProviderData;
  onSaved?: () => void;
}

export default function ProviderEditForm({ provider, onSaved }: Props) {
  const [specialties, setSpecialties] = useState<SpecialtySummary[]>([]);
  const [templates, setTemplates] = useState<TemplateSummary[]>([]);

  const [name, setName] = useState(provider.name || "");
  const [credentials, setCredentials] = useState(provider.credentials || "");
  const [specialty, setSpecialty] = useState(provider.specialty || "");
  const [practiceId, setPracticeId] = useState(provider.practice_id || "");
  const [noiseSuppression, setNoiseSuppression] = useState(provider.noise_suppression_level || "moderate");
  const [directives, setDirectives] = useState<string[]>(provider.style_directives || []);
  const [vocab, setVocab] = useState<string[]>(provider.custom_vocabulary || []);
  const [routing, setRouting] = useState<Record<string, string>>(provider.template_routing || {});

  const [newDirective, setNewDirective] = useState("");
  const [newTerm, setNewTerm] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  useEffect(() => {
    Promise.all([fetchSpecialties(), fetchTemplates()])
      .then(([specs, tpls]) => {
        setSpecialties(specs);
        setTemplates(tpls);
      })
      .catch(() => {});
  }, []);

  // Filter templates by selected specialty
  const matchingTemplates = templates.filter(
    (t) => !specialty || t.specialty === specialty
  );

  const handleSave = async () => {
    setError("");
    setSuccess("");
    setSaving(true);
    try {
      await updateProvider(provider.id, {
        name,
        credentials,
        specialty,
        practice_id: practiceId,
        noise_suppression_level: noiseSuppression,
        style_directives: directives,
        custom_vocabulary: vocab,
        template_routing: routing,
      });
      setSuccess("Provider updated successfully");
      setTimeout(() => setSuccess(""), 3000);
      onSaved?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save");
    } finally {
      setSaving(false);
    }
  };

  const addDirective = () => {
    if (newDirective.trim()) {
      setDirectives([...directives, newDirective.trim()]);
      setNewDirective("");
    }
  };

  const addTerm = () => {
    if (newTerm.trim()) {
      setVocab([...vocab, newTerm.trim()]);
      setNewTerm("");
    }
  };

  const VISIT_TYPES = ["default", "initial", "initial_evaluation", "follow_up", "assume_care", "discharge"];

  return (
    <div className="space-y-5">
      {success && (
        <div className="text-sm text-green-700 bg-green-50 px-3 py-2 rounded-lg">{success}</div>
      )}
      {error && (
        <div className="text-sm text-red-600 bg-red-50 px-3 py-2 rounded-lg">{error}</div>
      )}

      {/* Basic info */}
      <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100 space-y-4">
        <h2 className="text-sm font-semibold text-gray-800">Profile</h2>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Credentials</label>
            <input
              type="text"
              value={credentials}
              onChange={(e) => setCredentials(e.target.value)}
              placeholder="MD, DO, DC, NP..."
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Specialty</label>
            <select
              value={specialty}
              onChange={(e) => setSpecialty(e.target.value)}
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
            >
              <option value="">Select...</option>
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
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Noise Suppression</label>
            <select
              value={noiseSuppression}
              onChange={(e) => setNoiseSuppression(e.target.value)}
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
            >
              <option value="none">None</option>
              <option value="low">Low</option>
              <option value="moderate">Moderate</option>
              <option value="aggressive">Aggressive</option>
            </select>
          </div>
        </div>
      </div>

      {/* Template routing */}
      <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100 space-y-3">
        <h2 className="text-sm font-semibold text-gray-800">Template Routing</h2>
        <div className="space-y-2">
          {VISIT_TYPES.map((vt) => (
            <div key={vt} className="flex items-center gap-3">
              <span className="text-xs text-gray-500 w-32 capitalize">{vt.replace(/_/g, " ")}</span>
              <select
                value={routing[vt] || ""}
                onChange={(e) => {
                  const next = { ...routing };
                  if (e.target.value) {
                    next[vt] = e.target.value;
                  } else {
                    delete next[vt];
                  }
                  setRouting(next);
                }}
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
      </div>

      {/* Style directives */}
      <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100 space-y-3">
        <h2 className="text-sm font-semibold text-gray-800">
          Style Directives ({directives.length})
        </h2>
        <ul className="space-y-2">
          {directives.map((d, i) => (
            <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
              <span
                className="w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0"
                style={{ background: "var(--brand-green)" }}
              />
              <span className="flex-1">{d}</span>
              <button
                onClick={() => setDirectives(directives.filter((_, j) => j !== i))}
                className="text-red-400 hover:text-red-600 text-xs flex-shrink-0"
              >
                &times;
              </button>
            </li>
          ))}
        </ul>
        <div className="flex gap-2">
          <input
            type="text"
            value={newDirective}
            onChange={(e) => setNewDirective(e.target.value)}
            placeholder="Add a style directive..."
            className="flex-1 border border-gray-200 rounded px-2 py-1 text-sm"
            onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), addDirective())}
          />
          <button
            onClick={addDirective}
            className="text-xs px-3 py-1 rounded text-white"
            style={{ background: "var(--brand-indigo)" }}
          >
            Add
          </button>
        </div>
      </div>

      {/* Custom vocabulary */}
      <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100 space-y-3">
        <h2 className="text-sm font-semibold text-gray-800">
          Custom Vocabulary ({vocab.length} terms)
        </h2>
        <div className="flex flex-wrap gap-2">
          {vocab.map((t, i) => (
            <span
              key={i}
              className="text-xs px-2.5 py-1 rounded-full font-mono inline-flex items-center gap-1"
              style={{ background: "#EEF2FF", color: "var(--brand-indigo)" }}
            >
              {t}
              <button
                onClick={() => setVocab(vocab.filter((_, j) => j !== i))}
                className="text-indigo-300 hover:text-red-500 ml-0.5"
              >
                &times;
              </button>
            </span>
          ))}
        </div>
        <div className="flex gap-2">
          <input
            type="text"
            value={newTerm}
            onChange={(e) => setNewTerm(e.target.value)}
            placeholder="Add a term..."
            className="flex-1 border border-gray-200 rounded px-2 py-1 text-sm font-mono"
            onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), addTerm())}
          />
          <button
            onClick={addTerm}
            className="text-xs px-3 py-1 rounded text-white"
            style={{ background: "var(--brand-indigo)" }}
          >
            Add
          </button>
        </div>
      </div>

      {/* Save */}
      <div className="flex justify-end">
        <button
          onClick={handleSave}
          disabled={saving}
          className="px-5 py-2 text-sm font-medium text-white rounded-lg disabled:opacity-50"
          style={{ background: "var(--brand-green)" }}
        >
          {saving ? "Saving..." : "Save Changes"}
        </button>
      </div>
    </div>
  );
}
