"use client";

import { useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { Loader2, Play, CheckCircle2, X } from "lucide-react";
import { rerunPipeline, WS_BASE } from "@/lib/api";

interface Props {
  sampleId: string;
}

export default function RerunButton({ sampleId }: Props) {
  const router = useRouter();
  const [status, setStatus] = useState<"idle" | "running" | "complete" | "error">("idle");
  const [message, setMessage] = useState("");
  const [progress, setProgress] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);

  const handleRerun = async () => {
    setStatus("running");
    setMessage("Preparing pipeline...");
    setProgress(2);

    try {
      // Step 1: Call the rerun API to get encounter_id and version
      const result = await rerunPipeline(sampleId);
      setMessage(`Connecting to pipeline → ${result.version}`);
      setProgress(3);

      // Step 2: Connect WebSocket BEFORE the pipeline task has sent many events.
      // The server's asyncio.create_task starts the pipeline, but the first real
      // work (loading provider profile) happens after an await, so we have time
      // to connect before events fire. The server also buffers via send() —
      // if no WS is connected yet, events are simply skipped (acceptable for
      // early init events at 5-10%). The critical progress events (20%+) happen
      // after the slow ASR/LLM work begins, which takes seconds to minutes.
      const ws = new WebSocket(`${WS_BASE}/ws/encounters/${result.encounter_id}`);
      wsRef.current = ws;

      ws.onopen = () => {
        setMessage(`Running pipeline → ${result.version}`);
        setProgress(5);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "connected") {
          // Server acknowledged our connection
          setProgress(5);
        } else if (data.type === "progress") {
          setProgress(data.pct ?? 0);
          setMessage(data.message ?? `Stage: ${data.stage}`);
        } else if (data.type === "complete") {
          setStatus("complete");
          setProgress(100);
          setMessage(`${result.version} generated successfully`);
          ws.close();
          // Reload the page after a short delay to show new version
          setTimeout(() => {
            router.refresh();
            router.push(`/samples/${sampleId}?version=${result.version}`);
          }, 1500);
        } else if (data.type === "error") {
          setStatus("error");
          setMessage(data.error ?? "Pipeline error");
          ws.close();
        }
        // Ignore "ping" keepalives
      };

      ws.onerror = () => {
        // WebSocket failed — show a message but don't poll.
        // The pipeline still runs server-side; user can refresh manually.
        if (status !== "complete" && status !== "error") {
          setMessage(`Running pipeline → ${result.version} (live updates unavailable)`);
        }
      };

      ws.onclose = () => {
        wsRef.current = null;
      };
    } catch (err) {
      setStatus("error");
      setMessage(err instanceof Error ? err.message : "Failed to start pipeline");
    }
  };

  if (status === "idle") {
    return (
      <button
        onClick={handleRerun}
        className="flex items-center gap-1.5 px-4 py-2 text-sm font-medium text-white rounded-lg transition-opacity hover:opacity-90"
        style={{ background: "var(--brand-indigo)" }}
      >
        <Play size={14} /> Re-run Pipeline
      </button>
    );
  }

  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm" style={{
        background: status === "error" ? "#FEE2E2" : status === "complete" ? "#E6F7F2" : "#EEF2FF",
      }}>
        {status === "running" && <Loader2 size={14} className="animate-spin text-indigo-500" />}
        {status === "complete" && <CheckCircle2 size={14} style={{ color: "var(--brand-green)" }} />}
        {status === "error" && <X size={14} className="text-red-500" />}
        <span className={status === "error" ? "text-red-700" : "text-gray-700"}>
          {message}
        </span>
      </div>
      {status === "running" && (
        <div className="w-24 bg-gray-100 rounded-full h-1.5">
          <div
            className="h-1.5 rounded-full transition-all duration-500"
            style={{ width: `${progress}%`, background: "var(--brand-indigo)" }}
          />
        </div>
      )}
      {status === "error" && (
        <button
          onClick={() => { setStatus("idle"); setMessage(""); setProgress(0); }}
          className="text-xs text-gray-500 hover:text-gray-700 underline"
        >
          Try again
        </button>
      )}
    </div>
  );
}
