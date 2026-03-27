"use client";

import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
  ResponsiveContainer,
} from "recharts";
import type { AggregateQuality, DimensionScore } from "@/lib/api";

interface Props {
  trend: AggregateQuality[];
  dimensions: DimensionScore[];
}

export default function DashboardCharts({ trend, dimensions }: Props) {
  return (
    <div className="grid grid-cols-2 gap-5">
      {/* Quality trend */}
      <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100">
        <h2 className="text-sm font-semibold text-gray-800 mb-4">
          Quality Score by Version
        </h2>
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={trend} margin={{ left: 0, right: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
            <XAxis
              dataKey="version"
              tick={{ fontSize: 11, fill: "#94A3B8" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={[3, 5]}
              tick={{ fontSize: 11, fill: "#94A3B8" }}
              axisLine={false}
              tickLine={false}
              width={28}
            />
            <Tooltip
              contentStyle={{ borderRadius: 8, border: "1px solid #E2E8F0", fontSize: 12 }}
              formatter={(val) => typeof val === "number" ? val.toFixed(3) : val}
            />
            <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />
            <Bar
              dataKey="average"
              name="Avg Score"
              fill="#E6F7F6"
              stroke="#5BC0BE"
              strokeWidth={1.5}
              radius={[4, 4, 0, 0]}
            />
            <Line
              dataKey="average"
              name="Trend"
              stroke="#5BC0BE"
              strokeWidth={2.5}
              dot={{ fill: "#5BC0BE", r: 4 }}
              type="monotone"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Dimension radar */}
      <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100">
        <h2 className="text-sm font-semibold text-gray-800 mb-4">
          Quality Dimensions
        </h2>
        <ResponsiveContainer width="100%" height={220}>
          <RadarChart
            data={dimensions.map((d) => ({
              subject: d.dimension.replace("No ", "No-"),
              score: d.score ?? 0,
            }))}
          >
            <PolarGrid stroke="#F1F5F9" />
            <PolarAngleAxis
              dataKey="subject"
              tick={{ fontSize: 10, fill: "#64748B" }}
            />
            <Radar
              name="Score"
              dataKey="score"
              stroke="#5BC0BE"
              fill="#5BC0BE"
              fillOpacity={0.18}
              strokeWidth={2}
            />
            <Tooltip
              contentStyle={{ borderRadius: 8, border: "1px solid #E2E8F0", fontSize: 12 }}
              formatter={(val) => typeof val === "number" ? val.toFixed(2) : val}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
