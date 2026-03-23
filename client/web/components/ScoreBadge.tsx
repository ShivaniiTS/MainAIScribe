export default function ScoreBadge({ score }: { score: number | null | undefined }) {
  if (score == null) return <span className="text-gray-400 text-xs">—</span>;
  const cls =
    score >= 4.5
      ? "badge-score badge-score-high"
      : score >= 4.0
      ? "badge-score badge-score-mid"
      : "badge-score badge-score-low";
  return <span className={cls}>{score.toFixed(2)}</span>;
}
