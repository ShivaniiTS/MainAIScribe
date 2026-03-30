"use client";

import { useState } from "react";
import Link from "next/link";
import type { SampleSummary } from "@/lib/api";
import ScoreBadge from "./ScoreBadge";

type SortKey = "id" | "patient" | "provider" | "mode" | "score" | "accuracy" | "completeness" | "no_hallucination" | "structure";
type SortDir = "asc" | "desc";

/** Extract patient name from sample_id: "jane_doe_123456_20260303" → "Jane Doe" */
function extractPatientName(sampleId: string): string {
  // Pattern: name parts followed by numeric ID and date
  const parts = sampleId.split("_");
  // Find where the numeric patient ID starts (5-6 digit number)
  const nameEndIdx = parts.findIndex((p) => /^\d{5,}$/.test(p));
  if (nameEndIdx <= 0) return sampleId;
  return parts
    .slice(0, nameEndIdx)
    .map((p) => p.charAt(0).toUpperCase() + p.slice(1))
    .join(" ");
}

function formatProviderName(physician: string): string {
  return physician
    .replace(/^dr_/, "Dr. ")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace(/^Dr\. /, "Dr. ");
}

export default function SamplesTable({ samples }: { samples: SampleSummary[] }) {
  const [mode, setMode] = useState<"all" | "dictation" | "ambient">("all");
  const [sortKey, setSortKey] = useState<SortKey>("score");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDir(key === "id" || key === "patient" || key === "provider" ? "asc" : "desc");
    }
  };

  const filtered = samples
    .filter((s) => mode === "all" || s.mode === mode)
    .sort((a, b) => {
      let cmp = 0;
      switch (sortKey) {
        case "id":
          cmp = a.sample_id.localeCompare(b.sample_id);
          break;
        case "patient":
          cmp = extractPatientName(a.sample_id).localeCompare(extractPatientName(b.sample_id));
          break;
        case "provider":
          cmp = (a.physician ?? "").localeCompare(b.physician ?? "");
          break;
        case "mode":
          cmp = a.mode.localeCompare(b.mode);
          break;
        case "score":
          cmp = (a.quality?.overall ?? 0) - (b.quality?.overall ?? 0);
          break;
        case "accuracy":
          cmp = (a.quality?.accuracy ?? 0) - (b.quality?.accuracy ?? 0);
          break;
        case "completeness":
          cmp = (a.quality?.completeness ?? 0) - (b.quality?.completeness ?? 0);
          break;
        case "no_hallucination":
          cmp = (a.quality?.no_hallucination ?? 0) - (b.quality?.no_hallucination ?? 0);
          break;
        case "structure":
          cmp = (a.quality?.structure ?? 0) - (b.quality?.structure ?? 0);
          break;
      }
      return sortDir === "asc" ? cmp : -cmp;
    });

  const SortHeader = ({ label, col }: { label: string; col: SortKey }) => (
    <th
      className="px-4 py-3 font-medium cursor-pointer select-none hover:text-gray-600 transition-colors"
      onClick={() => handleSort(col)}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        {sortKey === col && (
          <span className="text-indigo-500">{sortDir === "asc" ? "↑" : "↓"}</span>
        )}
      </span>
    </th>
  );

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100">
      {/* Filters */}
      <div className="px-6 py-4 border-b border-gray-100 flex items-center gap-3">
        {(["all", "dictation", "ambient"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className="px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
            style={{
              background:
                mode === m ? "var(--brand-green)" : "#F1F5F9",
              color: mode === m ? "white" : "#64748B",
            }}
          >
            {m.charAt(0).toUpperCase() + m.slice(1)}
          </button>
        ))}
        <div className="ml-auto text-xs text-gray-400">
          Click column headers to sort
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-gray-400 border-b border-gray-50">
              <SortHeader label="Patient" col="patient" />
              <SortHeader label="Provider" col="provider" />
              <SortHeader label="Mode" col="mode" />
              <th className="px-4 py-3 font-medium">Versions</th>
              <SortHeader label="Overall" col="score" />
              <SortHeader label="Accuracy" col="accuracy" />
              <SortHeader label="Completeness" col="completeness" />
              <SortHeader label="No Halluc." col="no_hallucination" />
              <SortHeader label="Structure" col="structure" />
              <th className="px-4 py-3 font-medium">Gold</th>
              <th className="px-4 py-3 font-medium"></th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((s) => (
              <tr
                key={`${s.sample_id}_${s.mode}`}
                className="border-b border-gray-50 hover:bg-gray-50 transition-colors"
              >
                <td className="px-4 py-3 text-xs text-gray-800 font-medium max-w-[200px] truncate">
                  {s.is_test && (
                    <span className="inline-block mr-1.5 px-1.5 py-0.5 rounded text-[10px] font-bold bg-amber-100 text-amber-700">
                      TEST
                    </span>
                  )}
                  {extractPatientName(s.sample_id)}
                </td>
                <td className="px-4 py-3 text-xs text-gray-600 max-w-[140px] truncate">
                  {formatProviderName(s.physician)}
                </td>
                <td className="px-4 py-3">
                  <span
                    className="text-xs px-2 py-0.5 rounded-full font-medium"
                    style={{
                      background: s.mode === "dictation" ? "#EEF2FF" : "#E6F7F2",
                      color:
                        s.mode === "dictation"
                          ? "var(--brand-indigo)"
                          : "var(--brand-green-dark, #008F62)",
                    }}
                  >
                    {s.mode}
                  </span>
                </td>
                <td className="px-4 py-3 text-xs text-gray-400">
                  {s.versions.join(", ")}
                </td>
                <td className="px-4 py-3">
                  <ScoreBadge score={s.quality?.overall} />
                </td>
                <td className="px-4 py-3 text-gray-600">
                  {s.quality?.accuracy?.toFixed(1) ?? "—"}
                </td>
                <td className="px-4 py-3 text-gray-600">
                  {s.quality?.completeness?.toFixed(1) ?? "—"}
                </td>
                <td className="px-4 py-3 text-gray-600">
                  {s.quality?.no_hallucination?.toFixed(1) ?? "—"}
                </td>
                <td className="px-4 py-3 text-gray-600">
                  {s.quality?.structure?.toFixed(1) ?? "—"}
                </td>
                <td className="px-4 py-3">
                  {s.has_gold ? (
                    <span className="text-green-600 text-xs">&#10003;</span>
                  ) : (
                    <span className="text-gray-300 text-xs">—</span>
                  )}
                </td>
                <td className="px-4 py-3">
                  <Link
                    href={`/samples/${s.sample_id}`}
                    className="text-xs font-medium hover:underline"
                    style={{ color: "var(--brand-green)" }}
                  >
                    View &rarr;
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="px-6 py-3 text-xs text-gray-400 border-t border-gray-50">
        {filtered.length} of {samples.length} samples
      </div>
    </div>
  );
}
