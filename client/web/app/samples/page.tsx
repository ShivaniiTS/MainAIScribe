import { fetchSamples } from "@/lib/api";
import SamplesTable from "@/components/SamplesTable";

export const dynamic = "force-dynamic";

export default async function SamplesPage() {
  const samples = await fetchSamples().catch(() => []);
  return (
    <div className="p-8 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Samples</h1>
        <p className="text-gray-500 text-sm mt-1">
          {samples.length} encounters · Click a row to view transcript, note, comparison, and quality scores
        </p>
      </div>
      <SamplesTable samples={samples} />
    </div>
  );
}
