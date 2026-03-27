"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ResponsiveContainer,
} from "recharts";
import type { QualityScore } from "@/lib/api";
import ScoreBadge from "./ScoreBadge";

const DIMS: { key: keyof QualityScore; label: string }[] = [
  { key: "accuracy", label: "Medical Accuracy" },
  { key: "completeness", label: "Completeness" },
  { key: "no_hallucination", label: "No Hallucination" },
  { key: "structure", label: "Structure" },
  { key: "language", label: "Clinical Language" },
];

function barColor(score: number | null) {
  if (!score) return "#E2E8F0";
  if (score >= 4.5) return "#00B27A";
  if (score >= 4.0) return "#6366F1";
  return "#F59E0B";
}

export default function QualityPanel({
  quality,
}: {
  quality: QualityScore & { sample_id?: string };
}) {
  const chartData = DIMS.map(({ key, label }) => ({
    name: label,
    score: quality[key] as number | null,
  })).filter((d) => d.score != null);

  return (
    <div className="space-y-6">
      {/* Overall score */}
      <div className="flex items-center gap-4">
        <div>
          <div className="text-4xl font-bold text-gray-900">
            {quality.overall?.toFixed(2) ?? "—"}
            <span className="text-xl text-gray-400 font-normal"> / 5.0</span>
          </div>
          <div className="text-sm text-gray-500 mt-1">Overall Quality Score</div>
        </div>
        {quality.overlap && (
          <div className="ml-8">
            <div className="text-2xl font-bold text-gray-900">{quality.overlap}%</div>
            <div className="text-sm text-gray-500 mt-1">Keyword Overlap</div>
          </div>
        )}
      </div>

      {/* Dimension chart */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-3">
          Dimension Scores
        </h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ left: 12, right: 16 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" horizontal={false} />
            <XAxis
              type="number"
              domain={[0, 5]}
              tick={{ fontSize: 11, fill: "#94A3B8" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fontSize: 11, fill: "#64748B" }}
              axisLine={false}
              tickLine={false}
              width={130}
            />
            <Tooltip
              contentStyle={{
                borderRadius: 8,
                border: "1px solid #E2E8F0",
                fontSize: 12,
              }}
              formatter={(val) => typeof val === "number" ? val.toFixed(1) : val}
            />
            <Bar dataKey="score" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, i) => (
                <Cell key={i} fill={barColor(entry.score)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Dimension table */}
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-gray-400 border-b">
            <th className="py-2 font-medium">Dimension</th>
            <th className="py-2 font-medium text-right">Score</th>
          </tr>
        </thead>
        <tbody>
          {DIMS.map(({ key, label }) => (
            <tr key={key} className="border-b border-gray-50">
              <td className="py-2.5 text-gray-700">{label}</td>
              <td className="py-2.5 text-right">
                <ScoreBadge score={quality[key] as number | null} />
              </td>
            </tr>
          ))}
          <tr className="border-t-2 border-gray-200">
            <td className="py-2.5 font-semibold text-gray-800">Overall</td>
            <td className="py-2.5 text-right">
              <ScoreBadge score={quality.overall} />
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
