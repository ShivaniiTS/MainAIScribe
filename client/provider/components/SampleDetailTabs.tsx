"use client";

import { useState, useEffect, useRef } from "react";
import MarkdownViewer from "./MarkdownViewer";
import QualityPanel from "./QualityPanel";
import type { QualityScore } from "@/lib/api";
import { fetchNote, fetchTranscript } from "@/lib/api";

interface Props {
  sampleId: string;
  version: string;
  availableVersions: string[];
  note: string | null;
  comparison: string | null;
  gold: string | null;
  quality: (QualityScore & { sample_id: string }) | null;
  transcript: string | null;
  transcriptVersions: string[];
}

const TABS = [
  { id: "transcript", label: "Transcript" },
  { id: "note", label: "Clinical Note" },
  { id: "quality", label: "Quality" },
] as const;

type TabId = (typeof TABS)[number]["id"];

function VersionPicker({
  versions,
  selected,
  onChange,
  label,
}: {
  versions: string[];
  selected: string;
  onChange: (v: string) => void;
  label?: string;
}) {
  if (versions.length === 0) return null;
  return (
    <div className="flex items-center gap-2 flex-wrap">
      {label && <span className="text-xs text-gray-400">{label}</span>}
      {versions.map((v) => (
        <button
          key={v}
          onClick={() => onChange(v)}
          className="px-2 py-0.5 rounded text-xs font-medium transition-colors"
          style={{
            background: v === selected ? "var(--brand-accent)" : "#F1F5F9",
            color: v === selected ? "white" : "#64748B",
          }}
        >
          {v}
        </button>
      ))}
    </div>
  );
}

function AudioPlayer({ sampleId }: { sampleId: string }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);

  const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
  const src = `${BASE}/encounters/${sampleId}/audio`;

  const toggle = () => {
    const el = audioRef.current;
    if (!el) return;
    if (playing) { el.pause(); } else { el.play(); }
    setPlaying(!playing);
  };

  const fmt = (s: number) =>
    `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  return (
    <div
      className="flex items-center gap-3 p-3 rounded-lg mb-4"
      style={{ background: "#F8FAFC", border: "1px solid #E2E8F0" }}
    >
      <audio
        ref={audioRef}
        src={src}
        onTimeUpdate={(e) => setProgress(e.currentTarget.currentTime)}
        onDurationChange={(e) => setDuration(e.currentTarget.duration)}
        onEnded={() => setPlaying(false)}
      />
      <button
        onClick={toggle}
        className="w-8 h-8 rounded-full flex items-center justify-center text-white text-xs flex-shrink-0"
        style={{ background: "var(--brand-primary)" }}
      >
        {playing ? "\u23F8" : "\u25B6"}
      </button>
      <input
        type="range"
        min={0}
        max={duration || 1}
        value={progress}
        onChange={(e) => {
          const t = Number(e.target.value);
          setProgress(t);
          if (audioRef.current) audioRef.current.currentTime = t;
        }}
        className="flex-1 h-1 accent-teal-500"
      />
      <span className="text-xs text-gray-400 flex-shrink-0 font-mono">
        {fmt(progress)} / {fmt(duration)}
      </span>
    </div>
  );
}

export default function SampleDetailTabs({
  sampleId,
  version,
  availableVersions,
  note: initialNote,
  comparison,
  gold,
  quality,
  transcript: initialTranscript,
  transcriptVersions,
}: Props) {
  const [active, setActive] = useState<TabId>("note");

  // Note — version switching
  const [noteVersion, setNoteVersion] = useState(version);
  const [noteContent, setNoteContent] = useState<string | null>(initialNote);
  const [noteLoading, setNoteLoading] = useState(false);

  useEffect(() => {
    if (noteVersion === version) { setNoteContent(initialNote); return; }
    setNoteLoading(true);
    fetchNote(sampleId, noteVersion)
      .then((r) => setNoteContent(r.content))
      .catch(() => setNoteContent(null))
      .finally(() => setNoteLoading(false));
  }, [noteVersion, sampleId, version, initialNote]);

  // Transcript — version switching
  const [txVersion, setTxVersion] = useState(
    transcriptVersions.includes(version) ? version : (transcriptVersions[0] ?? version)
  );
  const [txContent, setTxContent] = useState<string | null>(initialTranscript);
  const [txLoading, setTxLoading] = useState(false);

  useEffect(() => {
    if (txVersion === version && initialTranscript) { setTxContent(initialTranscript); return; }
    setTxLoading(true);
    fetchTranscript(sampleId, txVersion)
      .then((r) => setTxContent(r.content))
      .catch(() => setTxContent(null))
      .finally(() => setTxLoading(false));
  }, [txVersion, sampleId, version, initialTranscript]);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100">
      {/* Tab bar */}
      <div className="flex border-b border-gray-100 px-6">
        {TABS.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setActive(id)}
            className="py-3 px-4 text-sm font-medium transition-colors border-b-2 -mb-px whitespace-nowrap"
            style={{
              borderColor: active === id ? "var(--brand-primary)" : "transparent",
              color: active === id ? "var(--brand-primary)" : "#94A3B8",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="p-6">
        {/* Transcript */}
        {active === "transcript" && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-medium text-gray-700">Audio Recording</span>
              <VersionPicker
                versions={transcriptVersions}
                selected={txVersion}
                onChange={setTxVersion}
                label="Version:"
              />
            </div>
            <AudioPlayer sampleId={sampleId} />
            {txLoading ? (
              <div className="text-center py-8 text-gray-400 text-sm">Loading transcript...</div>
            ) : txContent ? (
              <pre className="text-sm text-gray-700 whitespace-pre-wrap font-sans leading-relaxed bg-gray-50 rounded-lg p-4 max-h-[60vh] overflow-y-auto">
                {txContent}
              </pre>
            ) : (
              <Empty msg="Transcript not available for this version." />
            )}
          </div>
        )}

        {/* Clinical Note */}
        {active === "note" && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-medium text-gray-700">Generated Note</span>
              <VersionPicker
                versions={availableVersions}
                selected={noteVersion}
                onChange={setNoteVersion}
                label="Version:"
              />
            </div>
            {noteLoading ? (
              <div className="text-center py-8 text-gray-400 text-sm">Loading note...</div>
            ) : noteContent ? (
              <MarkdownViewer content={noteContent} />
            ) : (
              <Empty msg="Generated note not available for this version." />
            )}
          </div>
        )}

        {/* Quality */}
        {active === "quality" && (
          quality
            ? <QualityPanel quality={quality} />
            : <Empty msg="Quality scores not available for this encounter." />
        )}
      </div>
    </div>
  );
}

function Empty({ msg }: { msg: string }) {
  return (
    <div className="text-center py-12 text-gray-400 text-sm">{msg}</div>
  );
}
