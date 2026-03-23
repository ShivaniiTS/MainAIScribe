"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  Mic,
  MicOff,
  Upload,
  FileAudio,
  Loader2,
  CheckCircle2,
  Search,
  X,
} from "lucide-react";
import {
  fetchProviders,
  searchPatients,
  createEncounter,
  uploadEncounterAudio,
  WS_BASE,
} from "@/lib/api";
import type { ProviderSummary, PatientSearchResult } from "@/lib/api";
import { useFeatures } from "@/lib/useFeatures";

const VISIT_TYPES = [
  { value: "initial_evaluation", label: "Initial Evaluation" },
  { value: "follow_up", label: "Follow-up" },
  { value: "assume_care", label: "Assume Care" },
  { value: "discharge", label: "Discharge" },
];

type PipelineStage = "idle" | "creating" | "uploading" | "processing" | "complete" | "error";

interface WsEvent {
  type: "connected" | "progress" | "complete" | "error" | "ping";
  stage?: string;
  pct?: number;
  message?: string;
  sample_id?: string;
  error?: string;
}

export default function CapturePage() {
  const router = useRouter();
  const features = useFeatures();

  if (!features.record_audio && !features.trigger_pipeline) {
    return (
      <div className="p-8 text-gray-500">
        Audio capture is not available on this server.
      </div>
    );
  }

  // Providers
  const [providers, setProviders] = useState<ProviderSummary[]>([]);
  const [providerId, setProviderId] = useState("");

  // Patient search
  const [patientQuery, setPatientQuery] = useState("");
  const [patientResults, setPatientResults] = useState<PatientSearchResult[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<PatientSearchResult | null>(null);
  const [showPatientDropdown, setShowPatientDropdown] = useState(false);
  const patientSearchRef = useRef<HTMLDivElement>(null);

  // Visit config
  const [visitType, setVisitType] = useState("follow_up");
  const [mode, setMode] = useState<"dictation" | "ambient">("dictation");

  // Audio
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Pipeline status
  const [status, setStatus] = useState<PipelineStage>("idle");
  const [statusMessage, setStatusMessage] = useState("");
  const [progress, setProgress] = useState(0);
  const [encounterId, setEncounterId] = useState<string | null>(null);
  const [sampleId, setSampleId] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Load providers on mount
  useEffect(() => {
    fetchProviders()
      .then((ps) => {
        setProviders(ps);
        if (ps.length > 0) setProviderId(ps[0].id);
      })
      .catch(() => {});
  }, []);

  // Load default patients on mount (empty query returns first 10)
  const [allPatients, setAllPatients] = useState<PatientSearchResult[]>([]);
  useEffect(() => {
    searchPatients("")
      .then(setAllPatients)
      .catch(() => {});
  }, []);

  // Patient search debounce — works for any query length including empty
  useEffect(() => {
    if (patientQuery.trim().length === 0) {
      // Show default patients (already loaded)
      setPatientResults(allPatients);
      return;
    }
    const timer = setTimeout(() => {
      searchPatients(patientQuery)
        .then((results) => {
          setPatientResults(results);
          setShowPatientDropdown(true);
        })
        .catch(() => setPatientResults([]));
    }, 250);
    return () => clearTimeout(timer);
  }, [patientQuery, allPatients]);

  // Close patient dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (patientSearchRef.current && !patientSearchRef.current.contains(e.target as Node)) {
        setShowPatientDropdown(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  // Cleanup recording on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (mediaRecorderRef.current?.state === "recording") {
        mediaRecorderRef.current.stop();
      }
      wsRef.current?.close();
    };
  }, []);

  const selectPatient = (p: PatientSearchResult) => {
    setSelectedPatient(p);
    setPatientQuery(`${p.first_name} ${p.last_name}`);
    setShowPatientDropdown(false);
  };

  const clearPatient = () => {
    setSelectedPatient(null);
    setPatientQuery("");
    setPatientResults([]);
  };

  // Recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setRecordedBlob(blob);
        setAudioFile(null); // Clear file upload if recording
        stream.getTracks().forEach((t) => t.stop());
      };

      mediaRecorder.start(1000); // Collect data every second
      setIsRecording(true);
      setRecordingTime(0);
      setRecordedBlob(null);

      timerRef.current = setInterval(() => {
        setRecordingTime((t) => t + 1);
      }, 1000);
    } catch {
      setStatusMessage("Microphone access denied");
      setStatus("error");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  // WebSocket connection for pipeline progress
  const connectWs = useCallback((eid: string) => {
    const ws = new WebSocket(`${WS_BASE}/ws/encounters/${eid}`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data: WsEvent = JSON.parse(event.data);
      if (data.type === "progress") {
        setProgress(data.pct ?? 0);
        setStatusMessage(data.message ?? `Stage: ${data.stage}`);
      } else if (data.type === "complete") {
        setStatus("complete");
        setProgress(100);
        setStatusMessage("Pipeline complete — note generated");
        setSampleId(data.sample_id ?? null);
        ws.close();
      } else if (data.type === "error") {
        setStatus("error");
        setStatusMessage(data.error ?? "Pipeline error");
        ws.close();
      }
    };

    ws.onerror = () => {
      // WS not available — fall back to polling
    };
  }, []);

  // Submit
  const hasAudio = audioFile !== null || recordedBlob !== null;
  const canSubmit = providerId && selectedPatient && hasAudio && status === "idle";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;

    try {
      setStatus("creating");
      setStatusMessage("Creating encounter...");
      setProgress(0);

      const enc = await createEncounter({
        provider_id: providerId,
        patient_id: selectedPatient!.id,
        visit_type: visitType,
        mode,
      });
      setEncounterId(enc.encounter_id);

      // Connect WebSocket before uploading
      connectWs(enc.encounter_id);

      setStatus("uploading");
      setStatusMessage("Uploading audio...");
      setProgress(5);

      const audioBlob = audioFile ?? recordedBlob!;
      const filename = audioFile?.name ?? "recording.webm";
      const result = await uploadEncounterAudio(enc.encounter_id, audioBlob, filename);

      setStatus("processing");
      setStatusMessage(result.message ?? "Pipeline running...");
      setSampleId(result.sample_id);
      setProgress(10);
    } catch (err) {
      setStatus("error");
      setStatusMessage(err instanceof Error ? err.message : "Unknown error");
    }
  };

  const resetForm = () => {
    setStatus("idle");
    setStatusMessage("");
    setProgress(0);
    setEncounterId(null);
    setSampleId(null);
    setAudioFile(null);
    setRecordedBlob(null);
    setRecordingTime(0);
    wsRef.current?.close();
  };

  return (
    <div className="p-8 max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Capture Encounter</h1>
        <p className="text-gray-500 text-sm mt-1">
          Record or upload audio to generate a clinical note
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-5">
        {/* Step 1: Provider */}
        <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100 space-y-4">
          <h2 className="text-sm font-semibold text-gray-800">1. Provider</h2>
          <select
            value={providerId}
            onChange={(e) => setProviderId(e.target.value)}
            className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-green-200 bg-white"
          >
            {providers.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name ?? p.id} ({p.specialty})
              </option>
            ))}
            {providers.length === 0 && <option value="">No providers configured</option>}
          </select>
        </div>

        {/* Step 2: Patient search */}
        <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100 space-y-4">
          <h2 className="text-sm font-semibold text-gray-800">2. Patient</h2>

          {selectedPatient ? (
            <div className="flex items-center justify-between bg-green-50 rounded-lg px-4 py-3">
              <div>
                <div className="font-medium text-sm text-gray-800">
                  {selectedPatient.first_name} {selectedPatient.last_name}
                </div>
                <div className="text-xs text-gray-500">
                  MRN: {selectedPatient.mrn} &middot; DOB: {selectedPatient.date_of_birth} &middot; {selectedPatient.sex}
                </div>
              </div>
              <button type="button" onClick={clearPatient} className="text-gray-400 hover:text-gray-600">
                <X size={16} />
              </button>
            </div>
          ) : (
            <div ref={patientSearchRef} className="relative">
              <div className="relative">
                <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                <input
                  type="text"
                  value={patientQuery}
                  onChange={(e) => setPatientQuery(e.target.value)}
                  onFocus={() => {
                    // Show dropdown on focus — display default patients even when empty
                    if (patientResults.length > 0) {
                      setShowPatientDropdown(true);
                    } else if (allPatients.length > 0) {
                      setPatientResults(allPatients);
                      setShowPatientDropdown(true);
                    }
                  }}
                  placeholder="Search by name, MRN, or DOB..."
                  className="w-full border border-gray-200 rounded-lg pl-9 pr-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-green-200"
                />
              </div>
              {showPatientDropdown && patientResults.length > 0 && (
                <div className="absolute z-10 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-48 overflow-y-auto">
                  {patientResults.map((p) => (
                    <button
                      type="button"
                      key={p.id}
                      onClick={() => selectPatient(p)}
                      className="w-full text-left px-4 py-2.5 hover:bg-gray-50 border-b border-gray-100 last:border-b-0"
                    >
                      <div className="text-sm font-medium text-gray-800">
                        {p.first_name} {p.last_name}
                      </div>
                      <div className="text-xs text-gray-500">
                        MRN: {p.mrn} &middot; DOB: {p.date_of_birth} &middot; {p.sex}
                      </div>
                    </button>
                  ))}
                </div>
              )}
              {showPatientDropdown && patientQuery.length >= 1 && patientResults.length === 0 && (
                <div className="absolute z-10 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg px-4 py-3 text-sm text-gray-500">
                  No patients found
                </div>
              )}
            </div>
          )}
        </div>

        {/* Step 3: Visit config */}
        <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100 space-y-4">
          <h2 className="text-sm font-semibold text-gray-800">3. Visit Details</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-gray-500 mb-1">Visit Type</label>
              <select
                value={visitType}
                onChange={(e) => setVisitType(e.target.value)}
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-200 bg-white"
              >
                {VISIT_TYPES.map((v) => (
                  <option key={v.value} value={v.value}>{v.label}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 mb-1">Recording Mode</label>
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value as "dictation" | "ambient")}
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-200 bg-white"
              >
                <option value="dictation">Dictation (single speaker)</option>
                <option value="ambient">Ambient (multi-speaker)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Step 4: Audio */}
        <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100 space-y-4">
          <h2 className="text-sm font-semibold text-gray-800">4. Audio</h2>

          {/* Tab: Record vs Upload */}
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => { setAudioFile(null); }}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border transition-colors"
              style={{
                background: !audioFile && !recordedBlob ? "var(--brand-green)" : "white",
                color: !audioFile && !recordedBlob ? "white" : "#6B7280",
                borderColor: !audioFile && !recordedBlob ? "var(--brand-green)" : "#E5E7EB",
              }}
            >
              <Mic size={12} /> Record
            </button>
            <button
              type="button"
              onClick={() => { setRecordedBlob(null); setRecordingTime(0); }}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border transition-colors"
              style={{
                background: audioFile ? "var(--brand-green)" : "white",
                color: audioFile ? "white" : "#6B7280",
                borderColor: audioFile ? "var(--brand-green)" : "#E5E7EB",
              }}
            >
              <Upload size={12} /> Upload File
            </button>
          </div>

          {/* Record UI */}
          {!audioFile && (
            <div className="space-y-3">
              {!recordedBlob ? (
                <div className="flex items-center gap-4">
                  <button
                    type="button"
                    onClick={isRecording ? stopRecording : startRecording}
                    className="w-16 h-16 rounded-full flex items-center justify-center text-white transition-all"
                    style={{
                      background: isRecording ? "#EF4444" : "var(--brand-green)",
                    }}
                  >
                    {isRecording ? <MicOff size={24} /> : <Mic size={24} />}
                  </button>
                  <div>
                    {isRecording ? (
                      <>
                        <div className="text-sm font-medium text-red-600 flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                          Recording...
                        </div>
                        <div className="text-2xl font-mono text-gray-800">{formatTime(recordingTime)}</div>
                      </>
                    ) : (
                      <div className="text-sm text-gray-500">
                        Click to start recording
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="flex items-center gap-3 bg-green-50 rounded-lg px-4 py-3">
                  <FileAudio size={20} style={{ color: "var(--brand-green)" }} />
                  <div className="flex-1">
                    <div className="text-sm font-medium text-gray-800">
                      Recording ({formatTime(recordingTime)})
                    </div>
                    <div className="text-xs text-gray-500">
                      {(recordedBlob.size / 1024).toFixed(0)} KB
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => { setRecordedBlob(null); setRecordingTime(0); }}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    <X size={16} />
                  </button>
                </div>
              )}
            </div>
          )}

          {/* File upload UI */}
          {!recordedBlob && !isRecording && (
            <div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".mp3,.wav,.m4a,.ogg,.flac,.webm"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) { setAudioFile(f); setRecordedBlob(null); }
                }}
              />
              {audioFile ? (
                <div className="flex items-center gap-3 bg-green-50 rounded-lg px-4 py-3">
                  <FileAudio size={20} style={{ color: "var(--brand-green)" }} />
                  <div className="flex-1">
                    <div className="text-sm font-medium text-gray-800">{audioFile.name}</div>
                    <div className="text-xs text-gray-500">
                      {(audioFile.size / 1024 / 1024).toFixed(1)} MB
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setAudioFile(null)}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    <X size={16} />
                  </button>
                </div>
              ) : (
                <div
                  className="border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors hover:border-green-300"
                  style={{ borderColor: "#CBD5E1" }}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    const f = e.dataTransfer.files[0];
                    if (f) { setAudioFile(f); setRecordedBlob(null); }
                  }}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload size={20} className="mx-auto text-gray-300 mb-1" />
                  <div className="text-xs text-gray-500">
                    Drop audio file or <span style={{ color: "var(--brand-green)" }}>browse</span>
                  </div>
                  <div className="text-xs text-gray-400 mt-0.5">MP3, WAV, M4A, OGG, FLAC</div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Submit + progress */}
        {status === "idle" && (
          <button
            type="submit"
            disabled={!canSubmit}
            className="w-full py-3 rounded-xl text-sm font-semibold text-white transition-opacity disabled:opacity-50"
            style={{ background: "var(--brand-green)" }}
          >
            Run Pipeline
          </button>
        )}

        {status !== "idle" && (
          <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100 space-y-3">
            {/* Progress bar */}
            <div className="w-full bg-gray-100 rounded-full h-2">
              <div
                className="h-2 rounded-full transition-all duration-500"
                style={{
                  width: `${progress}%`,
                  background: status === "error" ? "#EF4444" : status === "complete" ? "var(--brand-green)" : "var(--brand-indigo)",
                }}
              />
            </div>

            <div className="flex items-center gap-2 text-sm">
              {status === "processing" && <Loader2 size={14} className="animate-spin text-indigo-500" />}
              {status === "creating" && <Loader2 size={14} className="animate-spin text-gray-500" />}
              {status === "uploading" && <Loader2 size={14} className="animate-spin text-gray-500" />}
              {status === "complete" && <CheckCircle2 size={14} style={{ color: "var(--brand-green)" }} />}
              {status === "error" && <X size={14} className="text-red-500" />}
              <span className={status === "error" ? "text-red-600" : "text-gray-700"}>
                {statusMessage}
              </span>
            </div>

            {encounterId && (
              <div className="text-xs text-gray-400">
                Encounter: {encounterId}
              </div>
            )}

            {status === "complete" && sampleId && (
              <div className="flex gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => router.push(`/samples/${sampleId}`)}
                  className="px-4 py-2 text-sm font-medium text-white rounded-lg"
                  style={{ background: "var(--brand-green)" }}
                >
                  View Note
                </button>
                <button
                  type="button"
                  onClick={resetForm}
                  className="px-4 py-2 text-sm font-medium text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200"
                >
                  New Encounter
                </button>
              </div>
            )}

            {status === "error" && (
              <button
                type="button"
                onClick={resetForm}
                className="px-4 py-2 text-sm font-medium text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200"
              >
                Try Again
              </button>
            )}
          </div>
        )}

        <p className="text-xs text-gray-400">
          Pipeline execution requires WhisperX (GPU) and Ollama running locally.
          Audio data is processed on-device — zero PHI egress.
        </p>
      </form>
    </div>
  );
}
