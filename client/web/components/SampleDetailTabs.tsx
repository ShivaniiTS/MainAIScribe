"use client";

import { useState, useEffect, useRef } from "react";
import MarkdownViewer from "./MarkdownViewer";
import QualityPanel from "./QualityPanel";
import type { QualityScore } from "@/lib/api";
import { fetchNote, fetchTranscript, BASE } from "@/lib/api";

interface Props {
  sampleId: string;
  version: string;                 // current version from URL
  availableVersions: string[];     // all versions with generated notes
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
  { id: "comparison", label: "Comparison" },
  { id: "gold", label: "Gold Standard" },
  { id: "quality", label: "Quality Scores" },
  { id: "diff", label: "Compare Versions" },
] as const;

type TabId = (typeof TABS)[number]["id"];

// ── Simple line-level diff ────────────────────────────────────────────────────
type DiffLine = { type: "same" | "added" | "removed"; text: string };

function diffLines(a: string, b: string): DiffLine[] {
  const aLines = a.split("\n");
  const bLines = b.split("\n");
  const m = aLines.length, n = bLines.length;

  // LCS dp table
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = m - 1; i >= 0; i--)
    for (let j = n - 1; j >= 0; j--)
      dp[i][j] = aLines[i] === bLines[j]
        ? dp[i + 1][j + 1] + 1
        : Math.max(dp[i + 1][j], dp[i][j + 1]);

  const result: DiffLine[] = [];
  let i = 0, j = 0;
  while (i < m || j < n) {
    if (i < m && j < n && aLines[i] === bLines[j]) {
      result.push({ type: "same", text: aLines[i++] });
      j++;
    } else if (j < n && (i >= m || dp[i + 1][j] <= dp[i][j + 1])) {
      result.push({ type: "added", text: bLines[j++] });
    } else {
      result.push({ type: "removed", text: aLines[i++] });
    }
  }
  return result;
}

// ── Version selector pill ─────────────────────────────────────────────────────
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
            background: v === selected ? "var(--brand-indigo)" : "#F1F5F9",
            color: v === selected ? "white" : "#64748B",
          }}
        >
          {v}
        </button>
      ))}
    </div>
  );
}

// ── Audio player ──────────────────────────────────────────────────────────────
function AudioPlayer({ sampleId }: { sampleId: string }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);

  
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
        style={{ background: "var(--brand-green)" }}
      >
        {playing ? "⏸" : "▶"}
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
        className="flex-1 h-1 accent-green-500"
      />
      <span className="text-xs text-gray-400 flex-shrink-0 font-mono">
        {fmt(progress)} / {fmt(duration)}
      </span>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
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
  const [active, setActive] = useState<TabId>("transcript");

  // Note tab — version switching
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

  // Transcript tab — version switching
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

  // Compare tab
  const [diffLeft, setDiffLeft] = useState(availableVersions[1] ?? availableVersions[0] ?? "latest");
  const [diffRight, setDiffRight] = useState(availableVersions[0] ?? "latest");
  const [leftContent, setLeftContent] = useState<string | null>(null);
  const [rightContent, setRightContent] = useState<string | null>(null);
  const [diffLoading, setDiffLoading] = useState(false);

  useEffect(() => {
    if (active !== "diff") return;
    setDiffLoading(true);
    Promise.all([
      fetchNote(sampleId, diffLeft).then((r) => r.content).catch(() => null),
      fetchNote(sampleId, diffRight).then((r) => r.content).catch(() => null),
    ]).then(([l, r]) => {
      setLeftContent(l);
      setRightContent(r);
    }).finally(() => setDiffLoading(false));
  }, [active, sampleId, diffLeft, diffRight]);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100">
      {/* Tab bar */}
      <div className="flex border-b border-gray-100 px-6 overflow-x-auto">
        {TABS.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setActive(id)}
            className="py-3 px-4 text-sm font-medium transition-colors border-b-2 -mb-px whitespace-nowrap"
            style={{
              borderColor: active === id ? "var(--brand-green)" : "transparent",
              color: active === id ? "var(--brand-green)" : "#94A3B8",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="p-6">

        {/* ── Transcript ── */}
        {active === "transcript" && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-medium text-gray-700">Audio Recording</span>
              <VersionPicker
                versions={transcriptVersions}
                selected={txVersion}
                onChange={setTxVersion}
                label="Transcript version:"
              />
            </div>
            <AudioPlayer sampleId={sampleId} />
            {txLoading ? (
              <div className="text-center py-8 text-gray-400 text-sm">Loading transcript…</div>
            ) : txContent ? (
              <pre className="text-sm text-gray-700 whitespace-pre-wrap font-sans leading-relaxed bg-gray-50 rounded-lg p-4 max-h-[60vh] overflow-y-auto">
                {txContent}
              </pre>
            ) : (
              <Empty msg="Transcript not available for this version." />
            )}
          </div>
        )}

        {/* ── Clinical Note ── */}
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
              <div className="text-center py-8 text-gray-400 text-sm">Loading note…</div>
            ) : noteContent ? (
              <MarkdownViewer content={noteContent} />
            ) : (
              <Empty msg="Generated note not available for this version." />
            )}
          </div>
        )}

        {/* ── Comparison ── */}
        {active === "comparison" && (
          comparison
            ? <MarkdownViewer content={comparison} />
            : <Empty msg="Comparison document not available for this sample/version." />
        )}

        {/* ── Gold Standard ── */}
        {active === "gold" && (
          gold
            ? <MarkdownViewer content={gold} />
            : <Empty msg="Gold-standard note not available for this sample." />
        )}

        {/* ── Quality Scores ── */}
        {active === "quality" && (
          quality
            ? <QualityPanel quality={quality} />
            : <Empty msg="Quality scores not available for this sample/version." />
        )}

        {/* ── Compare Versions ── */}
        {active === "diff" && (
          <div>
            <div className="flex items-center gap-6 mb-5">
              <VersionPicker
                versions={availableVersions}
                selected={diffLeft}
                onChange={setDiffLeft}
                label="Left:"
              />
              <span className="text-gray-300">vs</span>
              <VersionPicker
                versions={availableVersions}
                selected={diffRight}
                onChange={setDiffRight}
                label="Right:"
              />
            </div>

            {diffLoading ? (
              <div className="text-center py-8 text-gray-400 text-sm">Loading diff…</div>
            ) : leftContent && rightContent ? (
              <DiffView left={leftContent} right={rightContent} leftLabel={diffLeft} rightLabel={diffRight} />
            ) : (
              <Empty msg="Select two versions to compare." />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Diff view component ───────────────────────────────────────────────────────
function DiffView({
  left,
  right,
  leftLabel,
  rightLabel,
}: {
  left: string;
  right: string;
  leftLabel: string;
  rightLabel: string;
}) {
  const diffs = diffLines(left, right);

  return (
    <div className="overflow-x-auto">
      <div className="flex gap-1 mb-3 text-xs">
        <span className="px-2 py-0.5 rounded" style={{ background: "#FEE2E2", color: "#991B1B" }}>
          ─ {leftLabel} (removed)
        </span>
        <span className="px-2 py-0.5 rounded" style={{ background: "#DCFCE7", color: "#166534" }}>
          + {rightLabel} (added)
        </span>
        <span className="px-2 py-0.5 rounded" style={{ background: "#F8FAFC", color: "#64748B" }}>
          unchanged
        </span>
      </div>
      <div className="font-mono text-xs rounded-lg border border-gray-100 overflow-y-auto max-h-[70vh]">
        {diffs.map((line, i) => (
          <div
            key={i}
            style={{
              background:
                line.type === "added" ? "#F0FDF4"
                  : line.type === "removed" ? "#FFF1F2"
                  : undefined,
              borderLeft: `3px solid ${
                line.type === "added" ? "#22C55E"
                  : line.type === "removed" ? "#F43F5E"
                  : "transparent"
              }`,
              padding: "1px 8px 1px 6px",
              color: line.type === "same" ? "#6B7280" : undefined,
            }}
          >
            <span style={{ color: line.type === "added" ? "#16A34A" : line.type === "removed" ? "#DC2626" : "#9CA3AF", marginRight: 8 }}>
              {line.type === "added" ? "+" : line.type === "removed" ? "−" : " "}
            </span>
            {line.text || "\u00A0"}
          </div>
        ))}
      </div>
    </div>
  );
}

function Empty({ msg }: { msg: string }) {
  return (
    <div className="text-center py-12 text-gray-400 text-sm">{msg}</div>
  );
}
