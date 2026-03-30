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

  // Audio mode: "live" | "offline" | "upload"
  const [audioMode, setAudioMode] = useState<"live" | "offline" | "upload">("offline");
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Live transcription state
  const [liveTranscript, setLiveTranscript] = useState<string>("");
  const [livePartial, setLivePartial] = useState<string>("");
  const [liveSegments, setLiveSegments] = useState<Array<{text: string; speaker?: string; is_final: boolean}>>([]);
  const asrWsRef = useRef<WebSocket | null>(null);
  const liveTranscriptRef = useRef<HTMLDivElement>(null);

  // Pipeline status
  const [status, setStatus] = useState<PipelineStage>("idle");
  const [statusMessage, setStatusMessage] = useState("");
  const [progress, setProgress] = useState(0);
  const [encounterId, setEncounterId] = useState<string | null>(null);
  const [sampleId, setSampleId] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // ASR model preload status
  const [asrReady, setAsrReady] = useState(false);

  // Load providers on mount + preload streaming ASR model
  useEffect(() => {
    fetchProviders()
      .then((ps) => {
        setProviders(ps);
        if (ps.length > 0) setProviderId(ps[0].id);
      })
      .catch(() => {});

    // Preload NeMo streaming model so it's warm when recording starts
    // Default to port 8100 (pipeline server); override via NEXT_PUBLIC_API_URL
    const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8100";
    fetch(`${apiBase}/asr/preload`, { method: "POST" })
      .then((r) => r.json())
      .then((data) => {
        if (data.status === "ready") {
          setAsrReady(true);
        } else if (data.status === "failed" || data.status === "unavailable") {
          // Engine failed to load — let user try anyway (will fail at record time)
          setAsrReady(true);
        } else {
          // Poll until ready — NeMo model can take 2-10 min on first download
          const poll = setInterval(() => {
            fetch(`${apiBase}/asr/status`)
              .then((r) => r.json())
              .then((s) => {
                if (s.status === "ready" || s.status === "failed") {
                  setAsrReady(true);
                  clearInterval(poll);
                }
              })
              .catch(() => {});
          }, 3000);
          // After 10 min, stop polling and allow user to try regardless
          setTimeout(() => {
            clearInterval(poll);
            setAsrReady(true);
          }, 600000);
        }
      })
      .catch(() => {
        // API unreachable — allow user to try anyway
        setAsrReady(true);
      });
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

  // Recording (offline mode — record full audio, then upload)
  const startRecordingOffline = async () => {
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
        setAudioFile(null);
        stream.getTracks().forEach((t) => t.stop());
      };

      mediaRecorder.start(1000);
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

  // Recording (live mode — stream audio chunks via WebSocket for real-time transcription)
  // Status stays "idle" during live recording so the form + mic + transcript panel stay visible.
  // The encounter is created on record start; the offline pipeline runs after recording stops.
  const [liveError, setLiveError] = useState<string>("");

  const startRecordingLive = async () => {
    if (!providerId || !selectedPatient) return;
    setLiveError("");

    try {
      // Create encounter (needed for WebSocket session ID)
      const enc = await createEncounter({
        provider_id: providerId,
        patient_id: selectedPatient.id,
        visit_type: visitType,
        mode,
      });
      setEncounterId(enc.encounter_id);

      // Get mic stream
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true },
      });

      // Also record WebM for offline pipeline (runs after recording stops)
      const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setRecordedBlob(blob);
        stream.getTracks().forEach((t) => t.stop());
        try {
          if (asrWsRef.current && asrWsRef.current.readyState === WebSocket.OPEN) {
            asrWsRef.current.close();
          }
        } catch { /* ignore */ }
      };

      // Connect ASR WebSocket with format=pcm (browser sends raw PCM, no FFmpeg needed)
      const wsUrl = `${WS_BASE}/ws/asr/${enc.encounter_id}?mode=${mode}&format=pcm`;
      const asrWs = new WebSocket(wsUrl);
      asrWsRef.current = asrWs;

      asrWs.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "partial") {
            setLivePartial(data.text);
          } else if (data.type === "final") {
            setLiveSegments((prev) => [...prev, {
              text: data.text,
              speaker: data.speaker,
              is_final: true,
            }]);
            setLiveTranscript((prev) => prev + data.text + " ");
            setLivePartial("");
            if (liveTranscriptRef.current) {
              liveTranscriptRef.current.scrollTop = liveTranscriptRef.current.scrollHeight;
            }
          } else if (data.type === "complete") {
            setLivePartial("");
          } else if (data.type === "error") {
            setLiveError(data.message ?? "ASR error");
          }
        } catch { /* ignore parse errors */ }
      };

      asrWs.onerror = () => {
        setLiveError("ASR connection failed — transcript will be generated after recording");
      };

      // Use AudioContext to capture raw PCM at 16kHz and send via WebSocket
      const audioCtx = new AudioContext({ sampleRate: 16000 });
      const source = audioCtx.createMediaStreamSource(stream);

      // ScriptProcessor sends PCM chunks (4096 samples = 256ms at 16kHz)
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (e) => {
        const float32 = e.inputBuffer.getChannelData(0);
        // Convert float32 [-1,1] to int16 PCM
        const int16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
          const s = Math.max(-1, Math.min(1, float32[i]));
          int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        // Send raw PCM bytes if WebSocket is open
        if (asrWsRef.current && asrWsRef.current.readyState === WebSocket.OPEN) {
          asrWsRef.current.send(int16.buffer);
        }
      };
      source.connect(processor);
      processor.connect(audioCtx.destination);

      // Store refs for cleanup
      (mediaRecorderRef.current as any)._audioCtx = audioCtx;
      (mediaRecorderRef.current as any)._processor = processor;
      (mediaRecorderRef.current as any)._source = source;

      mediaRecorder.start(1000); // WebM recording for offline pipeline
      setIsRecording(true);
      setRecordingTime(0);
      setRecordedBlob(null);
      setLiveTranscript("");
      setLivePartial("");
      setLiveSegments([]);

      timerRef.current = setInterval(() => {
        setRecordingTime((t) => t + 1);
      }, 1000);
    } catch (err) {
      setLiveError(err instanceof Error ? err.message : "Failed to start recording");
    }
  };

  const startRecording = () => {
    if (audioMode === "live") {
      startRecordingLive();
    } else {
      startRecordingOffline();
    }
  };

  const stopRecording = () => {
    // Clean up AudioContext (live mode PCM streaming)
    if (mediaRecorderRef.current) {
      const mr = mediaRecorderRef.current as any;
      if (mr._processor) { mr._processor.disconnect(); mr._processor = null; }
      if (mr._source) { mr._source.disconnect(); mr._source = null; }
      if (mr._audioCtx) { mr._audioCtx.close().catch(() => {}); mr._audioCtx = null; }
    }
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
  const canSubmitOffline = providerId && selectedPatient && hasAudio && status === "idle";
  // Live mode: encounter already created during recording, just need to upload audio for pipeline
  const canSubmitLive = audioMode === "live" && recordedBlob && encounterId && !isRecording;
  const canSubmit = canSubmitOffline || canSubmitLive;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;

    try {
      if (audioMode === "live" && encounterId) {
        // Live mode: encounter already exists, upload audio for pipeline processing
        setStatus("uploading");
        setStatusMessage("Uploading audio for note generation...");
        setProgress(5);

        connectWs(encounterId);

        const audioBlob = recordedBlob!;
        const result = await uploadEncounterAudio(encounterId, audioBlob, "recording.webm");

        setStatus("processing");
        setStatusMessage(result.message ?? "Generating clinical note...");
        setSampleId(result.sample_id);
        setProgress(10);
      } else {
        // Offline / Upload mode: create encounter + upload + trigger pipeline
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
      }
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
    setLiveTranscript("");
    setLivePartial("");
    setLiveSegments([]);
    wsRef.current?.close();
    asrWsRef.current?.close();
  };

  if (!features.record_audio && !features.trigger_pipeline) {
    return (
      <div className="p-8 text-gray-500">
        Audio capture is not available on this server.
      </div>
    );
  }

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

          {/* Three-mode selector */}
          <div className="grid grid-cols-3 gap-2">
            {([
              {
                key: "live" as const, icon: <Mic size={14} />, label: "Record",
                sub: asrReady ? "Live Transcription" : "Loading ASR...",
              },
              { key: "offline" as const, icon: <MicOff size={14} />, label: "Record", sub: "Offline Transcription" },
              { key: "upload" as const, icon: <Upload size={14} />, label: "Upload", sub: "Audio File" },
            ]).map((opt) => (
              <button
                key={opt.key}
                type="button"
                onClick={() => {
                  setAudioMode(opt.key);
                  if (opt.key === "upload") { setRecordedBlob(null); setRecordingTime(0); }
                  if (opt.key !== "upload") { setAudioFile(null); }
                }}
                className="flex flex-col items-center gap-1 px-3 py-3 rounded-xl text-xs font-medium border-2 transition-all"
                style={{
                  background: audioMode === opt.key ? "var(--brand-green)" : "white",
                  color: audioMode === opt.key ? "white" : "#6B7280",
                  borderColor: audioMode === opt.key ? "var(--brand-green)" : "#E5E7EB",
                }}
              >
                {opt.icon}
                <span className="font-semibold">{opt.label}</span>
                <span className="text-[10px] opacity-80">{opt.sub}</span>
              </button>
            ))}
          </div>

          {/* Live / Offline record UI */}
          {audioMode !== "upload" && (
            <div className="space-y-3">
              {!recordedBlob ? (
                <div className="flex items-center gap-4">
                  <button
                    type="button"
                    onClick={isRecording ? stopRecording : startRecording}
                    disabled={(audioMode === "live" && (!providerId || !selectedPatient || !asrReady))}
                    className="w-16 h-16 rounded-full flex items-center justify-center text-white transition-all disabled:opacity-50"
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
                          {audioMode === "live" ? "Live Transcription..." : "Recording..."}
                        </div>
                        <div className="text-2xl font-mono text-gray-800">{formatTime(recordingTime)}</div>
                      </>
                    ) : (
                      <div className="text-sm text-gray-500">
                        {audioMode === "live"
                          ? "Click to start live transcription"
                          : "Click to start recording"}
                        {audioMode === "live" && !asrReady && (
                          <div className="text-xs text-blue-600 mt-1 flex items-center gap-1">
                            <Loader2 size={10} className="animate-spin" /> Loading ASR model...
                          </div>
                        )}
                        {audioMode === "live" && asrReady && (!providerId || !selectedPatient) && (
                          <div className="text-xs text-amber-600 mt-1">Select provider and patient first</div>
                        )}
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
                    onClick={() => { setRecordedBlob(null); setRecordingTime(0); setLiveTranscript(""); setLiveSegments([]); }}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    <X size={16} />
                  </button>
                </div>
              )}

              {/* Live transcript panel */}
              {audioMode === "live" && isRecording && (
                <div
                  ref={liveTranscriptRef}
                  className="bg-gray-50 rounded-lg p-4 max-h-48 overflow-y-auto border border-gray-200"
                >
                  <div className="text-xs font-semibold text-gray-400 mb-2">Live Transcript</div>
                  {liveSegments.map((seg, i) => (
                    <span key={i} className="text-sm text-gray-800">
                      {seg.speaker && (
                        <span className="text-xs font-bold mr-1" style={{
                          color: seg.speaker === "SPEAKER_00" ? "var(--brand-green)" : "var(--brand-indigo)",
                        }}>
                          {seg.speaker === "SPEAKER_00" ? "DR" : "PT"}:
                        </span>
                      )}
                      {seg.text}{" "}
                    </span>
                  ))}
                  {livePartial && (
                    <span className="text-sm text-gray-400 italic">{livePartial}</span>
                  )}
                  {liveSegments.length === 0 && !livePartial && (
                    <span className="text-sm text-gray-400 italic">Waiting for speech...</span>
                  )}
                </div>
              )}

              {/* Show final transcript after live recording stops */}
              {audioMode === "live" && !isRecording && liveTranscript && (
                <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                  <div className="text-xs font-semibold text-green-700 mb-2">
                    Live Transcript ({liveSegments.length} segments)
                  </div>
                  <p className="text-sm text-gray-800 whitespace-pre-wrap">{liveTranscript.trim()}</p>
                </div>
              )}

              {/* Live error (non-fatal — recording continues, offline pipeline will handle it) */}
              {liveError && (
                <div className="text-xs text-amber-600 bg-amber-50 rounded-lg px-3 py-2">
                  {liveError}
                </div>
              )}
            </div>
          )}

          {/* File upload UI */}
          {audioMode === "upload" && (
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
