import {
  fetchSample,
  fetchNote,
  fetchComparison,
  fetchGoldNote,
  fetchSampleQuality,
  fetchTranscript,
} from "@/lib/api";
import SampleDetailTabs from "@/components/SampleDetailTabs";
import RerunButton from "@/components/RerunButton";

export const dynamic = "force-dynamic";

interface Props {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ version?: string }>;
}

export default async function SampleDetailPage({ params, searchParams }: Props) {
  const { id } = await params;
  const { version: versionParam } = await searchParams;

  // Fetch sample detail first to determine latest version
  const sampleDetail = await fetchSample(id).catch(() => null);
  const version = versionParam ?? sampleDetail?.latest_version ?? sampleDetail?.versions?.[sampleDetail.versions.length - 1] ?? "latest";

  const [, note, comparison, gold, quality, transcript] = await Promise.allSettled([
    Promise.resolve(null),
    fetchNote(id, version),
    fetchComparison(id, version),
    fetchGoldNote(id),
    fetchSampleQuality(id, version),
    fetchTranscript(id, version),
  ]);
  const noteContent = note.status === "fulfilled" ? note.value.content : null;
  const compContent = comparison.status === "fulfilled" ? comparison.value.content : null;
  const goldContent = gold.status === "fulfilled" ? gold.value.content : null;
  const qualityData = quality.status === "fulfilled" ? quality.value : null;
  const transcriptContent = transcript.status === "fulfilled" ? transcript.value.content : null;
  const txVersions = transcript.status === "fulfilled" ? transcript.value.versions : [];

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <a href="/samples" className="text-sm text-gray-400 hover:text-gray-600">
              ← Samples
            </a>
          </div>
          <h1 className="text-xl font-bold text-gray-900 font-mono">{id}</h1>
          {sampleDetail && (
            <div className="flex items-center gap-3 mt-2">
              <span
                className="text-xs px-2 py-0.5 rounded-full font-medium"
                style={{
                  background: sampleDetail.mode === "dictation" ? "#EEF2FF" : "#E6F7F2",
                  color: sampleDetail.mode === "dictation" ? "#6366F1" : "#008F62",
                }}
              >
                {sampleDetail.mode}
              </span>
              {sampleDetail.patient_context?.provider?.name && (
                <span className="text-xs text-gray-500">
                  {sampleDetail.patient_context.provider.name}
                </span>
              )}
              {sampleDetail.patient_context?.encounter?.visit_type && (
                <span className="text-xs text-gray-400">
                  {sampleDetail.patient_context.encounter.visit_type.replace(/_/g, " ")}
                </span>
              )}
            </div>
          )}
        </div>

        {/* Version selector + Re-run */}
        <div className="flex flex-col items-end gap-3">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-400">Version:</span>
            {(sampleDetail?.versions ?? [version]).map((v) => (
              <a
                key={v}
                href={`/samples/${id}?version=${v}`}
                className="px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
                style={{
                  background: v === version ? "var(--brand-indigo)" : "#F1F5F9",
                  color: v === version ? "white" : "#64748B",
                }}
              >
                {v}
              </a>
            ))}
          </div>
          <RerunButton sampleId={id} />
        </div>
      </div>

      {/* Patient context banner */}
      {sampleDetail?.patient_context?.patient && (
        <div
          className="rounded-xl p-4 text-sm"
          style={{ background: "var(--brand-green-light, #E6F7F2)" }}
        >
          <div className="flex flex-wrap gap-6 text-gray-700">
            {sampleDetail.patient_context.patient.name && (
              <span><span className="text-gray-400 text-xs">Patient</span><br />
                <span className="font-medium">{sampleDetail.patient_context.patient.name}</span>
              </span>
            )}
            {sampleDetail.patient_context.patient.sex && (
              <span><span className="text-gray-400 text-xs">Sex</span><br />
                <span className="font-medium">{sampleDetail.patient_context.patient.sex}</span>
              </span>
            )}
            {sampleDetail.patient_context.patient.age && (
              <span><span className="text-gray-400 text-xs">Age</span><br />
                <span className="font-medium">{sampleDetail.patient_context.patient.age}</span>
              </span>
            )}
            {sampleDetail.patient_context.encounter?.date_of_service && (
              <span><span className="text-gray-400 text-xs">DOS</span><br />
                <span className="font-medium">{sampleDetail.patient_context.encounter.date_of_service}</span>
              </span>
            )}
            {sampleDetail.patient_context.encounter?.date_of_injury && (
              <span><span className="text-gray-400 text-xs">Date of Injury</span><br />
                <span className="font-medium">{sampleDetail.patient_context.encounter.date_of_injury}</span>
              </span>
            )}
          </div>
        </div>
      )}

      {/* Tabbed content */}
      <SampleDetailTabs
        sampleId={id}
        version={version}
        availableVersions={sampleDetail?.versions ?? [version]}
        note={noteContent}
        comparison={compContent}
        gold={goldContent}
        quality={qualityData}
        transcript={transcriptContent}
        transcriptVersions={txVersions}
      />
    </div>
  );
}
