import { fetchSamples } from "@/lib/api";
import SamplesTable from "@/components/SamplesTable";

export const dynamic = "force-dynamic";

export default async function EncountersPage() {
  const samples = await fetchSamples().catch(() => []);
  return (
    <div className="p-8 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Encounters</h1>
        <p className="text-gray-500 text-sm mt-1">
          {samples.length} encounters &middot; Click a row to view transcript and clinical note
        </p>
      </div>
      <SamplesTable samples={samples} />
    </div>
  );
}
