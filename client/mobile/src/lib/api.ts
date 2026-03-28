/**
 * API client for AI Scribe FastAPI backend.
 * Typed SDK matching the web app's lib/api.ts interface.
 */

import { getApiUrl } from "../store/settings";

// ---------------------------------------------------------------------------
// Types (mirroring web app)
// ---------------------------------------------------------------------------

export interface SampleSummary {
  sample_id: string;
  mode: "dictation" | "ambient";
  physician: string;
  versions: string[];
  latest_version: string | null;
  has_gold: boolean;
  quality: QualityScore | null;
}

export interface SampleDetail extends SampleSummary {
  patient_context: PatientContext | null;
}

export interface QualityScore {
  overall: number | null;
  accuracy: number | null;
  completeness: number | null;
  no_hallucination: number | null;
  structure: number | null;
  language: number | null;
  overlap: string | null;
}

export interface ProviderSummary {
  id: string;
  name: string | null;
  credentials: string | null;
  specialty: string | null;
  latest_score: number | null;
  quality_scores: Record<string, number>;
}

export interface PatientContext {
  patient?: {
    name?: string;
    date_of_birth?: string;
    age?: number;
    sex?: string;
    mrn?: string;
  };
  encounter?: {
    date_of_service?: string;
    visit_type?: string;
    date_of_injury?: string;
    mechanism_of_injury?: string;
  };
  provider?: {
    name?: string;
    credentials?: string;
    specialty?: string;
  };
  facility?: { name?: string; location?: string };
}

export interface PatientSearchResult {
  id: string;
  first_name: string;
  last_name: string;
  date_of_birth: string;
  sex: string;
  mrn: string;
  practice_id: string;
}

export interface EncounterCreateResponse {
  encounter_id: string;
  status: string;
  provider_id: string;
  patient_id: string;
  visit_type: string;
  mode: string;
  message: string | null;
}

export interface UploadResponse {
  encounter_id: string;
  sample_id: string;
  status: string;
  message: string;
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

/** Common headers sent with every request (bypass tunnel interstitials, etc.) */
const COMMON_HEADERS: Record<string, string> = {
  "Bypass-Tunnel-Reminder": "true",
};

async function get<T>(path: string, params?: Record<string, string>): Promise<T> {
  const base = getApiUrl();
  let fullUrl = `${base}${path}`;
  if (params) {
    const qs = Object.entries(params)
      .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
      .join("&");
    fullUrl += `?${qs}`;
  }
  const res = await fetch(fullUrl, { headers: COMMON_HEADERS });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${path} — ${body}`);
  }
  return res.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const base = getApiUrl();
  const res = await fetch(`${base}${path}`, {
    method: "POST",
    headers: { ...COMMON_HEADERS, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `API ${res.status}` }));
    throw new Error(err.detail || `API ${res.status}: ${path}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Encounters
// ---------------------------------------------------------------------------

export const fetchSamples = (mode?: string) =>
  get<SampleSummary[]>("/encounters", mode ? { mode } : undefined);

export const fetchSample = (id: string) =>
  get<SampleDetail>(`/encounters/${id}`);

export const fetchNote = (id: string, version?: string) =>
  get<{ content: string }>(`/encounters/${id}/note`, version ? { version } : undefined);

export const fetchTranscript = (id: string, version?: string) =>
  get<{ content: string; versions: string[] }>(
    `/encounters/${id}/transcript`,
    version ? { version } : undefined,
  );

export const fetchAudioUrl = (id: string): string => `${getApiUrl()}/encounters/${id}/audio`;

export const fetchGoldNote = (id: string) =>
  get<{ content: string }>(`/encounters/${id}/gold`);

export const fetchSampleQuality = (id: string, version?: string) =>
  get<QualityScore & { sample_id: string }>(
    `/encounters/${id}/quality`,
    version ? { version } : undefined,
  );

// ---------------------------------------------------------------------------
// Providers
// ---------------------------------------------------------------------------

export const fetchProviders = () => get<ProviderSummary[]>("/providers");

export const fetchProvider = (id: string) =>
  get<Record<string, unknown>>(`/providers/${id}`);

// ---------------------------------------------------------------------------
// Patients
// ---------------------------------------------------------------------------

export const searchPatients = (q: string) =>
  get<PatientSearchResult[]>("/patients/search", { q });

// ---------------------------------------------------------------------------
// Encounter creation + upload
// ---------------------------------------------------------------------------

export const createEncounter = (data: {
  provider_id: string;
  patient_id: string;
  visit_type: string;
  mode: string;
}) => post<EncounterCreateResponse>("/encounters", data);

export async function uploadEncounterAudio(
  encounterId: string,
  fileUri: string,
  filename = "recording.m4a",
): Promise<UploadResponse> {
  const base = getApiUrl();
  const form = new FormData();
  form.append("audio", {
    uri: fileUri,
    name: filename,
    type: "audio/m4a",
  } as unknown as Blob);

  // Don't set Content-Type manually — fetch auto-adds the multipart boundary
  const { "Content-Type": _, ...headersWithoutCT } = COMMON_HEADERS;
  const res = await fetch(`${base}/encounters/${encounterId}/upload`, {
    method: "POST",
    body: form,
    headers: headersWithoutCT,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `Upload failed (${res.status})` }));
    throw new Error(err.detail || `Upload failed (${res.status})`);
  }
  return res.json();
}

export const fetchEncounterStatus = (id: string) =>
  get<{ encounter_id: string; status: string; message: string; sample_id?: string }>(
    `/encounters/${id}/status`,
  );

// ---------------------------------------------------------------------------
// WebSocket URL
// ---------------------------------------------------------------------------

export function getWsUrl(): string {
  return getApiUrl().replace(/^http/, "ws");
}
