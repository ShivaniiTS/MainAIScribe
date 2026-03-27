"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createSpecialty } from "@/lib/api";
import { useFeatures } from "@/lib/useFeatures";

export default function NewSpecialtyPage() {
  const router = useRouter();
  const features = useFeatures();

  if (!features.create_specialties) {
    return (
      <div className="p-8 text-gray-500">
        Creating specialties is not available on this server.
      </div>
    );
  }
  const [id, setId] = useState("");
  const [termsText, setTermsText] = useState("");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    const cleanId = id.trim().toLowerCase().replace(/\s+/g, "_");
    if (!cleanId) {
      setError("Specialty ID is required");
      return;
    }
    const terms = termsText
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l && !l.startsWith("#"));

    setSaving(true);
    try {
      await createSpecialty({ id: cleanId, terms });
      router.push(`/specialties/${cleanId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="p-8 max-w-2xl space-y-6">
      <div>
        <a href="/specialties" className="text-sm text-gray-400 hover:text-gray-600">
          &larr; Specialties
        </a>
        <h1 className="text-2xl font-bold text-gray-900 mt-2">New Specialty</h1>
      </div>

      <form onSubmit={handleSubmit} className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 space-y-5">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Specialty ID
          </label>
          <input
            type="text"
            value={id}
            onChange={(e) => setId(e.target.value)}
            placeholder="e.g. dermatology"
            className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-200"
          />
          <p className="text-xs text-gray-400 mt-1">
            Used as the dictionary filename: config/dictionaries/{id || "..."}.txt
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Dictionary Terms
          </label>
          <textarea
            value={termsText}
            onChange={(e) => setTermsText(e.target.value)}
            placeholder={"Enter one term per line:\ndermatitis\neczema\npsoriasis\nmelanoma\n..."}
            rows={16}
            className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-green-200"
          />
          <p className="text-xs text-gray-400 mt-1">
            {termsText.split("\n").filter((l) => l.trim() && !l.trim().startsWith("#")).length} terms
          </p>
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
            {saving ? "Creating..." : "Create Specialty"}
          </button>
          <a
            href="/specialties"
            className="px-4 py-2 text-sm font-medium text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200"
          >
            Cancel
          </a>
        </div>
      </form>
    </div>
  );
}
