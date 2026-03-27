"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { fetchSpecialty, updateSpecialtyDictionary } from "@/lib/api";
import type { SpecialtyDetail } from "@/lib/api";
import { useFeatures } from "@/lib/useFeatures";

export default function SpecialtyDetailPage() {
  const params = useParams();
  const router = useRouter();
  const features = useFeatures();
  const id = params.id as string;

  const [specialty, setSpecialty] = useState<SpecialtyDetail | null>(null);
  const [editing, setEditing] = useState(false);
  const [termsText, setTermsText] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  useEffect(() => {
    fetchSpecialty(id)
      .then((s) => {
        setSpecialty(s);
        setTermsText(s.terms.join("\n"));
      })
      .catch(() => setSpecialty(null));
  }, [id]);

  const handleSave = async () => {
    setError("");
    setSuccess("");
    const terms = termsText
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l && !l.startsWith("#"));
    setSaving(true);
    try {
      const updated = await updateSpecialtyDictionary(id, terms);
      setSpecialty(updated);
      setEditing(false);
      setSuccess("Dictionary saved successfully");
      setTimeout(() => setSuccess(""), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save");
    } finally {
      setSaving(false);
    }
  };

  if (specialty === null) {
    return <div className="p-8 text-gray-400">Loading...</div>;
  }

  return (
    <div className="p-8 space-y-6">
      <div className="flex items-center gap-3 mb-1">
        <a href="/specialties" className="text-sm text-gray-400 hover:text-gray-600">
          &larr; Specialties
        </a>
      </div>

      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-4">
          <div
            className="w-14 h-14 rounded-full flex items-center justify-center text-white font-bold text-lg flex-shrink-0"
            style={{ background: "var(--brand-indigo)" }}
          >
            {specialty.name.charAt(0).toUpperCase()}
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900 capitalize">{specialty.name}</h1>
            <p className="text-gray-500 text-sm mt-0.5">
              <span className="font-mono">config/dictionaries/{specialty.id}.txt</span>
              {" "}&middot;{" "}
              {specialty.term_count} terms
            </p>
          </div>
        </div>
        {features.edit_specialties && (
          <button
            onClick={() => setEditing(!editing)}
            className="px-4 py-2 text-sm font-medium rounded-lg"
            style={{
              background: editing ? "#F3F4F6" : "var(--brand-green)",
              color: editing ? "#374151" : "white",
            }}
          >
            {editing ? "Cancel" : "Edit Dictionary"}
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

      {/* Dictionary */}
      <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100">
        <h2 className="text-sm font-semibold text-gray-800 mb-3">
          Keyword Dictionary
        </h2>
        {editing ? (
          <div className="space-y-3">
            <textarea
              value={termsText}
              onChange={(e) => setTermsText(e.target.value)}
              rows={24}
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-green-200"
            />
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-400">
                {termsText.split("\n").filter((l) => l.trim() && !l.trim().startsWith("#")).length} terms
              </span>
              <button
                onClick={handleSave}
                disabled={saving}
                className="px-4 py-2 text-sm font-medium text-white rounded-lg disabled:opacity-50"
                style={{ background: "var(--brand-green)" }}
              >
                {saving ? "Saving..." : "Save Dictionary"}
              </button>
            </div>
          </div>
        ) : (
          <div className="flex flex-wrap gap-2 max-h-96 overflow-y-auto">
            {specialty.terms.map((term, i) => (
              <span
                key={i}
                className="text-xs px-2.5 py-1 rounded-full font-mono"
                style={{ background: "#EEF2FF", color: "var(--brand-indigo)" }}
              >
                {term}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
