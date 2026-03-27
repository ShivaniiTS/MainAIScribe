"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

interface TrendEntry {
  version: string;
  score: number;
  date: string | null;
  samples: number | null;
}

export default function ProviderTrendChart({ trend }: { trend: TrendEntry[] }) {
  return (
    <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100">
      <h2 className="text-sm font-semibold text-gray-800 mb-4">
        Quality Score Trend
      </h2>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={trend} margin={{ left: 0, right: 16 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
          <XAxis
            dataKey="version"
            tick={{ fontSize: 11, fill: "#94A3B8" }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            domain={[3.5, 5]}
            tick={{ fontSize: 11, fill: "#94A3B8" }}
            axisLine={false}
            tickLine={false}
            width={28}
          />
          <ReferenceLine y={4.0} stroke="#E2E8F0" strokeDasharray="4 4" />
          <Tooltip
            contentStyle={{
              borderRadius: 8,
              border: "1px solid #E2E8F0",
              fontSize: 12,
            }}
            formatter={(val, _name, props) => [
              `${typeof val === "number" ? val.toFixed(3) : val} (${(props.payload as {samples?: number}).samples ?? "?"} samples)`,
              "Score",
            ]}
          />
          <Line
            dataKey="score"
            stroke="#00B27A"
            strokeWidth={2.5}
            dot={{ fill: "#00B27A", r: 5, strokeWidth: 2, stroke: "#fff" }}
            type="monotone"
          />
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-3 grid grid-cols-4 gap-3">
        {trend.map((t) => (
          <div key={t.version} className="text-center">
            <div className="text-lg font-bold text-gray-900">{t.score.toFixed(2)}</div>
            <div className="text-xs text-gray-400">{t.version}</div>
            {t.date && <div className="text-xs text-gray-300">{t.date}</div>}
          </div>
        ))}
      </div>
    </div>
  );
}
