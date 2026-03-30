"use client";

import { useState, useRef } from "react";
import { Upload, FileAudio, Loader2, CheckCircle2 } from "lucide-react";
import type { ProviderSummary } from "@/lib/api";
import { BASE } from "@/lib/api";

const VISIT_TYPES = [
  { value: "initial_evaluation", label: "Initial Evaluation" },
  { value: "follow_up", label: "Follow-up" },
  { value: "assume_care", label: "Assume Care" },
  { value: "discharge", label: "Discharge" },
];

export default function UploadForm({ providers }: { providers: ProviderSummary[] }) {
  const [file, setFile] = useState<File | null>(null);
  const [providerId, setProviderId] = useState(providers[0]?.id ?? "");
  const [visitType, setVisitType] = useState("follow_up");
  const [mode, setMode] = useState<"dictation" | "ambient">("dictation");
  const [status, setStatus] = useState<
    "idle" | "creating" | "uploading" | "processing" | "done" | "error"
  >("idle");
  const [message, setMessage] = useState("");
  const [encounterId, setEncounterId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !providerId) return;

    try {
      setStatus("creating");
      setMessage("Creating encounter...");

      // 1. Create encounter
      const createRes = await fetch(`${BASE}/encounters`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider_id: providerId,
          patient_id: `upload-${Date.now()}`,
          visit_type: visitType,
          mode,
        }),
      });
      if (!createRes.ok) throw new Error("Failed to create encounter");
      const enc = await createRes.json();
      setEncounterId(enc.encounter_id);

      setStatus("uploading");
      setMessage("Uploading audio file...");

      // 2. Upload audio
      const form = new FormData();
      form.append("audio", file);
      const uploadRes = await fetch(`${BASE}/encounters/${enc.encounter_id}/upload`, {
        method: "POST",
        body: form,
      });
      if (!uploadRes.ok) throw new Error("Upload failed");
      const uploadData = await uploadRes.json();

      setStatus("processing");
      setMessage(uploadData.message ?? "Pipeline queued — requires local GPU + Ollama");
      // WebSocket progress would connect here in production
      setTimeout(() => setStatus("done"), 1500);
    } catch (err) {
      setStatus("error");
      setMessage(err instanceof Error ? err.message : "Unknown error");
    }
  };

  return (
    <div className="max-w-xl">
      <form onSubmit={handleSubmit} className="space-y-5">
        {/* Drop zone */}
        <div
          className="border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors"
          style={{
            borderColor: file ? "var(--brand-green)" : "#CBD5E1",
            background: file ? "var(--brand-green-light, #E6F7F2)" : "white",
          }}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".mp3,.wav,.m4a,.ogg,.flac"
            className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          {file ? (
            <div className="flex items-center justify-center gap-3">
              <FileAudio size={24} style={{ color: "var(--brand-green)" }} />
              <div>
                <div className="font-medium text-gray-800 text-sm">{file.name}</div>
                <div className="text-xs text-gray-400">
                  {(file.size / 1024 / 1024).toFixed(1)} MB
                </div>
              </div>
            </div>
          ) : (
            <div>
              <Upload size={32} className="mx-auto text-gray-300 mb-2" />
              <div className="text-sm text-gray-500">
                Drag & drop audio file here, or{" "}
                <span style={{ color: "var(--brand-green)" }}>browse</span>
              </div>
              <div className="text-xs text-gray-400 mt-1">
                MP3, WAV, M4A, OGG, FLAC
              </div>
            </div>
          )}
        </div>

        {/* Provider */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1.5">
            Provider
          </label>
          <select
            value={providerId}
            onChange={(e) => setProviderId(e.target.value)}
            className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 bg-white"
            style={{ "--tw-ring-color": "var(--brand-green)" } as React.CSSProperties}
          >
            {providers.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name ?? p.id} ({p.specialty})
              </option>
            ))}
            {providers.length === 0 && (
              <option value="">No providers configured</option>
            )}
          </select>
        </div>

        {/* Visit type + mode */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5">
              Visit Type
            </label>
            <select
              value={visitType}
              onChange={(e) => setVisitType(e.target.value)}
              className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm focus:outline-none bg-white"
            >
              {VISIT_TYPES.map((v) => (
                <option key={v.value} value={v.value}>
                  {v.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5">
              Recording Mode
            </label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as "dictation" | "ambient")}
              className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm focus:outline-none bg-white"
            >
              <option value="dictation">Dictation (single speaker)</option>
              <option value="ambient">Ambient (multi-speaker)</option>
            </select>
          </div>
        </div>

        {/* Submit */}
        <button
          type="submit"
          disabled={!file || !providerId || status !== "idle"}
          className="w-full py-3 rounded-xl text-sm font-semibold text-white transition-opacity disabled:opacity-50"
          style={{ background: "var(--brand-green)" }}
        >
          {status === "idle" && "Run Pipeline"}
          {status === "creating" && (
            <span className="flex items-center justify-center gap-2">
              <Loader2 size={14} className="animate-spin" /> Creating encounter...
            </span>
          )}
          {status === "uploading" && (
            <span className="flex items-center justify-center gap-2">
              <Loader2 size={14} className="animate-spin" /> Uploading...
            </span>
          )}
          {status === "processing" && "Processing (requires local GPU + Ollama)..."}
          {status === "done" && (
            <span className="flex items-center justify-center gap-2">
              <CheckCircle2 size={14} /> Submitted
            </span>
          )}
          {status === "error" && "Error — try again"}
        </button>

        {/* Status message */}
        {message && (
          <div
            className="text-sm rounded-lg px-4 py-3"
            style={{
              background: status === "error" ? "#FEE2E2" : "#E6F7F2",
              color: status === "error" ? "#991B1B" : "#065F46",
            }}
          >
            {message}
            {encounterId && (
              <div className="text-xs mt-1 opacity-70">
                Encounter ID: {encounterId}
              </div>
            )}
          </div>
        )}

        <p className="text-xs text-gray-400">
          Note: Pipeline execution requires WhisperX (GPU) and Ollama running locally.
          Audio data is processed on-device — zero PHI egress.
        </p>
      </form>
    </div>
  );
}
