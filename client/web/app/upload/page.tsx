import UploadForm from "@/components/UploadForm";
import { fetchProviders } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function UploadPage() {
  const providers = await fetchProviders().catch(() => []);
  return (
    <div className="p-8 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Upload Encounter</h1>
        <p className="text-gray-500 text-sm mt-1">
          Upload an audio file to run through the AI Scribe pipeline
        </p>
      </div>
      <UploadForm providers={providers} />
    </div>
  );
}
