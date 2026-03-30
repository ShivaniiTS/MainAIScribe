const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8100";

export interface SampleSummary {
  sample_id: string;
  mode: "dictation" | "ambient";
  physician: string;
  versions: string[];
  latest_version: string | null;
  has_gold: boolean;
  quality: QualityScore | null;
  is_test: boolean;
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

export interface AggregateQuality {
  version: string;
  sample_count: number;
  average: number;
  min: number;
  max: number;
  dimensions: Record<string, number | null>;
}

export interface DimensionScore {
  dimension: string;
  score: number | null;
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

async function get<T>(path: string, params?: Record<string, string>): Promise<T> {
  const url = new URL(`${BASE}${path}`);
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  }
  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json();
}

// Encounters
export const fetchSamples = (mode?: string) =>
  get<SampleSummary[]>("/encounters", mode ? { mode } : undefined);

export const fetchSample = (id: string) =>
  get<SampleDetail>(`/encounters/${id}`);

export const fetchNote = (id: string, version = "latest") =>
  get<{ content: string }>(`/encounters/${id}/note`, { version });

export const fetchComparison = (id: string, version = "latest") =>
  get<{ content: string }>(`/encounters/${id}/comparison`, { version });

export const fetchGoldNote = (id: string) =>
  get<{ content: string }>(`/encounters/${id}/gold`);

export const fetchSampleQuality = (id: string, version = "latest") =>
  get<QualityScore & { sample_id: string }>(`/encounters/${id}/quality`, { version });

export const fetchTranscript = (id: string, version = "latest") =>
  get<{ content: string; versions: string[] }>(`/encounters/${id}/transcript`, { version });

export const fetchAudioUrl = (id: string): string =>
  `${BASE}/encounters/${id}/audio`;

// Quality
export const fetchAggregate = (version = "latest") =>
  get<AggregateQuality>("/quality/aggregate", { version });

export const fetchTrend = () =>
  get<{ trend: AggregateQuality[] }>("/quality/trend");

export const fetchDimensions = (version = "latest") =>
  get<DimensionScore[]>("/quality/dimensions", { version });

export const fetchSampleScores = (version = "latest", mode?: string) =>
  get<Array<QualityScore & { sample_id: string; mode: string; version: string }>>(
    "/quality/samples",
    mode ? { version, mode } : { version }
  );

// Quality — breakdowns
export interface ProviderQuality {
  provider_id: string;
  provider_name: string;
  sample_count: number;
  average: number;
  min: number;
  max: number;
}

export const fetchQualityByMode = (version = "latest") =>
  get<Record<string, AggregateQuality>>("/quality/by-mode", { version });

export const fetchQualityByProvider = (version = "latest") =>
  get<ProviderQuality[]>("/quality/by-provider", { version });

// Providers
export const fetchProviders = () => get<ProviderSummary[]>("/providers");
export const fetchProvider = (id: string) => get<Record<string, unknown>>(`/providers/${id}`);
export const fetchProviderTrend = (id: string) =>
  get<{ trend: Array<{ version: string; score: number; date: string | null; samples: number | null }> }>(
    `/providers/${id}/quality-trend`
  );
export const createProvider = (data: Record<string, unknown>) =>
  post<ProviderSummary>("/providers", data);
export const updateProvider = (id: string, data: Record<string, unknown>) =>
  put<ProviderSummary>(`/providers/${id}`, data);

// Specialties
export interface SpecialtySummary {
  id: string;
  name: string;
  term_count: number;
  has_dictionary: boolean;
}

export interface SpecialtyDetail {
  id: string;
  name: string;
  term_count: number;
  terms: string[];
}

export const fetchSpecialties = () => get<SpecialtySummary[]>("/specialties");
export const fetchSpecialty = (id: string) => get<SpecialtyDetail>(`/specialties/${id}`);
export const createSpecialty = (data: { id: string; terms: string[] }) =>
  post<SpecialtySummary>("/specialties", data);
export const updateSpecialtyDictionary = (id: string, terms: string[]) =>
  put<SpecialtyDetail>(`/specialties/${id}/dictionary`, { terms });

// Templates
export interface TemplateSummary {
  id: string;
  name: string;
  specialty: string;
  visit_type: string;
  section_count: number;
  providers: string[];
}

export interface TemplateSection {
  id: string;
  label: string;
  required: boolean;
  prompt_hint: string;
}

export interface TemplateDetail {
  id: string;
  name: string;
  specialty: string;
  visit_type: string;
  header_fields: string[];
  sections: TemplateSection[];
  formatting: Record<string, string>;
  providers: string[];
}

export const fetchTemplates = () => get<TemplateSummary[]>("/templates");
export const fetchTemplate = (id: string) => get<TemplateDetail>(`/templates/${id}`);
export const createTemplate = (data: Record<string, unknown>) =>
  post<TemplateSummary>("/templates", data);
export const updateTemplate = (id: string, data: Record<string, unknown>) =>
  put<TemplateDetail>(`/templates/${id}`, data);
export const deleteTemplate = async (id: string) => {
  const res = await fetch(`${BASE}/templates/${id}`, { method: "DELETE" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Delete failed" }));
    throw new Error(err.detail || "Delete failed");
  }
};

// Patients (EHR stub)
export interface PatientSearchResult {
  id: string;
  first_name: string;
  last_name: string;
  date_of_birth: string;
  sex: string;
  mrn: string;
  practice_id: string;
}

export const searchPatients = (q: string) =>
  get<PatientSearchResult[]>("/patients/search", { q });

// Encounter creation + upload
export interface EncounterCreateResponse {
  encounter_id: string;
  status: string;
  provider_id: string;
  patient_id: string;
  visit_type: string;
  mode: string;
  message: string | null;
}

export const createEncounter = (data: {
  provider_id: string;
  patient_id: string;
  visit_type: string;
  mode: string;
}) => post<EncounterCreateResponse>("/encounters", data);

export async function uploadEncounterAudio(
  encounterId: string,
  audioFile: File | Blob,
  filename = "audio.mp3",
): Promise<{ encounter_id: string; sample_id: string; status: string; message: string }> {
  const form = new FormData();
  form.append("audio", audioFile, filename);
  const res = await fetch(`${BASE}/encounters/${encounterId}/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `Upload failed (${res.status})` }));
    throw new Error(err.detail || `Upload failed (${res.status})`);
  }
  return res.json();
}

export const fetchEncounterStatus = (id: string) =>
  get<{ encounter_id: string; status: string; message: string; sample_id?: string }>(
    `/encounters/${id}/status`
  );

export const rerunPipeline = (sampleId: string) =>
  post<{ encounter_id: string; sample_id: string; version: string; status: string; message: string }>(
    `/encounters/${sampleId}/rerun`,
    {}
  );

export const WS_BASE = BASE.replace(/^http/, "ws");

// ---------------------------------------------------------------------------
// Server role / feature flags
// ---------------------------------------------------------------------------
export interface FeatureFlags {
  dashboard: boolean;
  view_encounters: boolean;
  view_providers: boolean;
  view_specialties: boolean;
  view_templates: boolean;
  view_quality: boolean;
  record_audio: boolean;
  trigger_pipeline: boolean;
  run_pipeline: boolean;
  batch_processing: boolean;
  ehr_access: boolean;
  patient_search: boolean;
  create_providers: boolean;
  edit_providers: boolean;
  create_templates: boolean;
  edit_templates: boolean;
  create_specialties: boolean;
  edit_specialties: boolean;
}

export interface ServerRoleInfo {
  role: string;
  instance_id: string;
  is_provider_facing: boolean;
  is_processing_pipeline: boolean;
}

export const fetchFeatures = () => get<FeatureFlags>("/config/features");
export const fetchServerRole = () => get<ServerRoleInfo>("/config/role");

// Mutation helpers
async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `API ${res.status}` }));
    throw new Error(err.detail || `API ${res.status}: ${path}`);
  }
  return res.json();
}

async function put<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `API ${res.status}` }));
    throw new Error(err.detail || `API ${res.status}: ${path}`);
  }
  return res.json();
}
