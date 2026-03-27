# Streaming ASR — Implementation Guide

This document provides detailed implementation guidance for the real-time streaming ASR system. It covers the full chain from browser microphone to NeMo GPU inference and back, including every pitfall we encountered and how we solved it.

---

## 1. Architecture Overview

```
Browser (port 3000)                Provider Server (port 8000)           Pipeline Server (port 8100)
┌─────────────────────┐            ┌──────────────────────┐              ┌──────────────────────────┐
│ AudioContext 16kHz   │            │                      │              │                          │
│ ScriptProcessor      │──PCM──→   │  WS /ws/asr/{id}     │──PCM──→     │  WS /ws/asr/{id}         │
│ Int16 PCM chunks    │  WebSocket │  (asr_proxy.py)      │  WebSocket  │  (audio_stream.py)       │
│                     │            │  Bidirectional proxy  │              │  ↓                       │
│ Live transcript     │←──JSON──   │                      │←──JSON──    │  NemoStreamingServer     │
│ panel (React)       │            │                      │              │  ↓ model.transcribe()    │
│                     │            │                      │              │  PartialTranscript       │
└─────────────────────┘            └──────────────────────┘              └──────────────────────────┘
```

**Key principle:** The browser sends **raw PCM** (16kHz, mono, 16-bit signed integers). No WebM, no Opus, no server-side FFmpeg. This eliminates the biggest class of streaming bugs.

---

## 2. Client-Side Implementation (Browser)

### 2.1 Why NOT MediaRecorder for Streaming

We initially used `MediaRecorder` with `timeslice: 160` to send WebM chunks. This failed because:

1. **WebM container headers**: MediaRecorder sends a container header in the first chunk, then delta-encoded frames. The server needs to parse the full WebM container to decode audio — you can't just pipe fragments through FFmpeg.
2. **Opus codec latency**: WebM/Opus has inherent encoding latency that adds 20-60ms per chunk on top of network latency.
3. **FFmpeg pipe unreliability**: Piping streaming WebM to FFmpeg's stdin works for files but is unreliable for real-time fragments — FFmpeg buffers internally and the pipe blocks.

### 2.2 Correct Approach: AudioContext + ScriptProcessorNode

The browser captures raw PCM directly using the Web Audio API:

```typescript
// 1. Get microphone stream
const stream = await navigator.mediaDevices.getUserMedia({
  audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true },
});

// 2. Create AudioContext at 16kHz (NeMo's native sample rate)
const audioCtx = new AudioContext({ sampleRate: 16000 });
const source = audioCtx.createMediaStreamSource(stream);

// 3. ScriptProcessor captures PCM chunks
//    4096 samples at 16kHz = 256ms per chunk
const processor = audioCtx.createScriptProcessor(4096, 1, 1);

processor.onaudioprocess = (e) => {
  const float32 = e.inputBuffer.getChannelData(0);

  // Convert float32 [-1.0, 1.0] → int16 [-32768, 32767]
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }

  // Send raw PCM bytes over WebSocket
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(int16.buffer);
  }
};

source.connect(processor);
processor.connect(audioCtx.destination);
```

**Critical details:**
- `sampleRate: 16000` in both `getUserMedia` and `AudioContext` — NeMo expects 16kHz. If the browser doesn't support 16kHz natively, the AudioContext resamples automatically.
- `ScriptProcessorNode` with `bufferSize: 4096` gives 256ms chunks — small enough for responsive streaming, large enough to avoid excessive WebSocket messages.
- `channelCount: 1` — mono audio. NeMo doesn't use stereo.
- The `int16` conversion is essential — NeMo expects signed 16-bit PCM, not float32.
- Always check `ws.readyState === WebSocket.OPEN` before sending — the WebSocket may close at any time.

### 2.3 Parallel WebM Recording

While streaming PCM for live transcription, we also record WebM for the offline note-generation pipeline (which uses WhisperX for higher-quality batch transcription):

```typescript
// MediaRecorder runs in parallel — captures WebM for offline pipeline
const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
mediaRecorder.ondataavailable = (e) => {
  if (e.data.size > 0) chunks.push(e.data);
};
mediaRecorder.start(1000); // 1-second WebM chunks (for offline only)
```

After recording stops, the WebM blob is uploaded to the pipeline for WhisperX batch transcription + note generation. The live NeMo transcript is for real-time feedback only.

### 2.4 WebSocket Connection

```typescript
// format=pcm tells the server: raw PCM, no conversion needed
const wsUrl = `${WS_BASE}/ws/asr/${encounterId}?mode=dictation&format=pcm`;
const ws = new WebSocket(wsUrl);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case "connected":
      // Server confirms connection + which engine is loaded
      // { type: "connected", engine: "nemo_streaming", encounter_id: "..." }
      break;

    case "final":
      // Complete segment — append to transcript
      // { type: "final", text: "The patient presents with...", is_final: true,
      //   speaker: "SPEAKER_00", start_ms: 1200, end_ms: 4500, confidence: 0.92 }
      appendToTranscript(data.text, data.speaker);
      break;

    case "partial":
      // Interim result — show in gray, will be replaced by next final
      // { type: "partial", text: "The patient pre...", is_final: false }
      showPartialText(data.text);
      break;

    case "complete":
      // Session ended — full accumulated transcript
      // { type: "complete", transcript: "Full text...", segments: [...] }
      break;

    case "error":
      // Non-fatal — recording continues, offline pipeline handles it
      // { type: "error", message: "ASR engine error: ..." }
      showWarning(data.message);
      break;

    case "ping":
      // Keepalive — ignore
      break;
  }
};

// Non-blocking error handler — don't crash the recording
ws.onerror = () => {
  showWarning("ASR connection failed — transcript will be generated after recording");
};
```

### 2.5 Cleanup on Stop

```typescript
function stopRecording() {
  // 1. Disconnect AudioContext nodes (stops PCM streaming)
  processor.disconnect();
  source.disconnect();
  audioCtx.close();

  // 2. Stop MediaRecorder (triggers onstop → creates WebM blob)
  if (mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }

  // 3. Close ASR WebSocket (triggers server-side finalization)
  if (ws.readyState === WebSocket.OPEN) {
    ws.close();
  }
}
```

### 2.6 React Hooks Pitfall

**Never put an early return before hooks.** This caused a "Rendered fewer hooks than expected" crash:

```typescript
// WRONG — hooks below this won't run on the early-return path
function CapturePage() {
  const features = useFeatures();
  if (!features.record_audio) return <div>Not available</div>;  // ← BREAKS HOOKS
  const [state, setState] = useState(...);  // ← Not always called
  ...
}

// CORRECT — all hooks first, then conditional return
function CapturePage() {
  const features = useFeatures();
  const [state, setState] = useState(...);  // ← Always called
  ...
  if (!features.record_audio) return <div>Not available</div>;  // ← Safe here
  return <div>...</div>;
}
```

---

## 3. Provider-Facing Server (Port 8000) — ASR Proxy

The provider-facing server doesn't run NeMo. It proxies ASR requests to the pipeline server.

### 3.1 REST Proxy (`api/ws/asr_proxy.py`)

```python
@router.post("/asr/preload")
async def proxy_asr_preload(mode: str = "dictation"):
    """Forward preload request to pipeline server."""
    async with httpx.AsyncClient(base_url=pipeline_url) as client:
        resp = await client.post("/asr/preload", params={"mode": mode})
        return resp.json()

@router.get("/asr/status")
async def proxy_asr_status():
    """Forward status check to pipeline server."""
    async with httpx.AsyncClient(base_url=pipeline_url) as client:
        resp = await client.get("/asr/status")
        return resp.json()
```

### 3.2 WebSocket Proxy

The WebSocket proxy is bidirectional — audio flows client → provider → pipeline, transcripts flow pipeline → provider → client:

```python
@router.websocket("/ws/asr/{encounter_id}")
async def proxy_asr_websocket(encounter_id, websocket, mode, format):
    await websocket.accept()

    # Connect to pipeline server's WebSocket
    pipeline_ws_url = f"ws://pipeline-server:8100/ws/asr/{encounter_id}?mode={mode}&format={format}"

    async with websockets.connect(pipeline_ws_url) as pipeline_ws:
        async def client_to_pipeline():
            """Forward audio from browser to pipeline."""
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if "bytes" in message:
                    await pipeline_ws.send(message["bytes"])

        async def pipeline_to_client():
            """Forward transcripts from pipeline to browser."""
            async for msg in pipeline_ws:
                await websocket.send_text(msg)

        await asyncio.gather(client_to_pipeline(), pipeline_to_client())
```

**Key detail:** Use `websocket.receive()` (not `receive_bytes()`) to handle both binary audio and close frames. `receive_bytes()` throws on close frames.

---

## 4. Pipeline Server (Port 8100) — NeMo Streaming

### 4.1 Model Preloading (`POST /asr/preload`)

The NeMo model takes ~30s to load into GPU memory. The Capture page calls `POST /asr/preload` on mount so the model is warm by the time the user starts recording.

```python
@router.post("/asr/preload")
async def preload_streaming_model(mode: str = "dictation"):
    engine = _get_streaming_engine(mode)
    if engine._loaded:
        return {"status": "ready"}

    # Background load — returns immediately
    asyncio.create_task(asyncio.to_thread(engine._ensure_model))
    return {"status": "loading"}
```

The frontend polls `GET /asr/status` every 2 seconds until `status: "ready"`.

### 4.2 WebSocket Handler (`api/ws/audio_stream.py`)

**Producer/consumer pattern** — decouples audio receiving from ASR processing:

```python
@router.websocket("/ws/asr/{encounter_id}")
async def asr_stream(encounter_id, websocket, mode, format):
    await websocket.accept()

    audio_queue = asyncio.Queue(maxsize=100)
    client_done = asyncio.Event()

    async def receive_audio():
        """Producer: receive PCM chunks, put into queue."""
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if "bytes" in message:
                await audio_queue.put(message["bytes"])
        await audio_queue.put(None)  # Sentinel
        client_done.set()

    async def process_audio():
        """Consumer: pull from queue, run NeMo, send results."""
        while True:
            pcm_data = await asyncio.wait_for(audio_queue.get(), timeout=5.0)
            if pcm_data is None:
                break
            async for partial in engine.transcribe_stream(pcm_data, session_id, config):
                await websocket.send_text(json.dumps({...}))

    await asyncio.gather(receive_audio(), process_audio())
```

**Why producer/consumer?** Without it, the receive loop blocks during NeMo inference (~100ms per window). Audio chunks pile up and the connection stalls. With the queue, receiving and processing happen concurrently.

**Critical timeout:** `process_audio` uses `asyncio.wait_for(queue.get(), timeout=5.0)` — if no audio arrives for 5 seconds AND the client is done, it exits. Without this, the consumer hangs forever after the client disconnects.

### 4.3 NeMo Streaming Server (`mcp_servers/asr/nemo_streaming_server.py`)

**Sliding window approach** — only transcribe the latest 1 second of new audio:

```python
STREAM_WINDOW_S = 1.0  # Transcribe every 1s of new audio

async def transcribe_stream(self, audio_chunk, session_id, config):
    session = self._get_or_create_session(session_id)
    session.full_pcm += audio_chunk
    session.elapsed_ms += ...

    # Only transcribe when enough new audio has accumulated
    new_bytes = len(session.full_pcm) - session.last_transcribed_bytes
    new_seconds = new_bytes / (16000 * 2)
    if new_seconds < self.STREAM_WINDOW_S:
        return  # Not enough new audio yet

    # Extract ONLY the new window (not full buffer)
    window_pcm = session.full_pcm[session.last_transcribed_bytes:]
    session.last_transcribed_bytes = len(session.full_pcm)

    # Transcribe the window
    result = await asyncio.to_thread(self._transcribe_window_nemo, window_pcm)
    yield PartialTranscript(text=result["text"], is_final=True, ...)
```

**Why sliding window, not full-buffer re-transcription?**

We initially re-transcribed the entire accumulated audio every 3 seconds. This caused:
- **Growing latency**: 3s audio = 100ms inference, 30s audio = 1s inference, 5 min audio = 10s+ inference
- **WebSocket timeouts**: clients disconnected while waiting for long transcriptions
- **Redundant work**: re-transcribing the first 29 seconds to find 1 second of new text

The sliding window gives O(1) latency regardless of session length: each 1-second window takes ~60-100ms on A10G.

### 4.4 NeMo Model API

The NeMo `EncDecRNNTBPEModel` does NOT support the `transcribe_simulate_cache_aware_streaming` method (despite the name). The correct API:

```python
# Write PCM window to temp WAV file
sf.write(temp_path, audio_array, 16000)

# Transcribe the WAV file (batch mode on a short clip)
results = model.transcribe([temp_path])
text = results[0]  # String transcript
```

Each 1-second window transcription takes ~60-100ms on A10G GPU.

**Model loading:** `ASRModel.from_pretrained("nvidia/nemotron-speech-streaming-en-0.6b")` downloads ~600MB and takes ~30s to load. Once loaded, it stays in GPU memory (~4.6 GB) until explicitly unloaded or the idle timeout expires.

---

## 5. Latency Breakdown

| Stage | Latency | Notes |
|-------|---------|-------|
| AudioContext buffer fill | 256ms | 4096 samples at 16kHz |
| WebSocket send (browser → provider) | <5ms | localhost |
| WebSocket proxy (provider → pipeline) | <5ms | localhost |
| Audio buffer accumulation | ~750ms | Waiting for 1s window |
| NeMo inference | 60-100ms | Per 1s window on A10G |
| WebSocket response (pipeline → browser) | <10ms | Two hops |
| **Total end-to-end** | **~1.1s** | From speech to displayed text |
| **First segment (cold model)** | **~30s** | One-time model load |
| **First segment (warm model)** | **~1.1s** | Preload on page navigation |

---

## 6. VRAM Budget

| Component | VRAM | Notes |
|-----------|------|-------|
| NeMo Nemotron-Speech-Streaming | 4.6 GB | Stays loaded during streaming session |
| WhisperX (batch, for offline) | 10-12 GB | Loaded after streaming ends |
| Ollama qwen2.5:14b | ~8 GB | Loaded for note generation |
| **Constraint (A10G)** | **23 GB** | NeMo + Ollama fit; WhisperX needs NeMo unloaded first |

**Lifecycle:** NeMo loads on preload → stays loaded during recording → unloaded after idle timeout (5 min) → WhisperX loads for batch pipeline → Ollama loads for note generation.

---

## 7. Debugging Checklist

If live transcription isn't working, check in order:

1. **Is the pipeline server running?** `curl http://localhost:8100/health`
2. **Is the model loaded?** `curl http://localhost:8100/asr/status` — should be `"ready"`
3. **Does the proxy work?** `curl http://localhost:8000/asr/status` — should match above
4. **Can WebSocket connect through proxy?**
   ```python
   import websockets, asyncio
   async def test():
       async with websockets.connect("ws://localhost:8000/ws/asr/test?mode=dictation&format=pcm") as ws:
           print(await ws.recv())
   asyncio.run(test())
   ```
5. **Is the browser sending PCM?** Check browser DevTools → Network → WS tab → look for binary frames
6. **Is the server receiving audio?** Check `/tmp/ai-scribe-logs/api-pipeline.log` for `asr_stream: session started`
7. **Is NeMo transcribing?** Check logs for `Transcribing: 1it` messages from NeMo

---

## 8. Files Reference

| File | Server | Purpose |
|------|--------|---------|
| `client/web/app/capture/page.tsx` | Browser | AudioContext PCM capture + WebSocket client + live transcript UI |
| `api/ws/asr_proxy.py` | Provider (8000) | Proxies /asr/preload, /asr/status, /ws/asr/ to pipeline |
| `api/ws/audio_stream.py` | Pipeline (8100) | WebSocket handler: receives PCM, runs NeMo, sends transcripts |
| `mcp_servers/asr/nemo_streaming_server.py` | Pipeline (8100) | NeMo model wrapper: session management, sliding window transcription |
| `mcp_servers/asr/nemo_multitalker_server.py` | Pipeline (8100) | Multi-speaker streaming with speaker labels (ambient mode) |
| `mcp_servers/asr/base.py` | Both | `ASREngine` interface, `PartialTranscript` dataclass |
| `mcp_servers/registry.py` | Both | Engine registry with `nemo_streaming` + `nemo_multitalker` entries |
| `config/engines.yaml` | Both | `streaming_server: nemo_streaming` config |
| `orchestrator/state.py` | Pipeline | `streaming_transcript` field on EncounterState |
| `orchestrator/nodes/transcribe_node.py` | Pipeline | Conditional: use streaming transcript if available, else batch ASR |
| `api/main.py` | Both | Mounts audio_stream on pipeline, asr_proxy on provider |

---

## 9. Lessons Learned

1. **Never send WebM over WebSocket for real-time ASR.** Browser AudioContext → raw PCM → WebSocket is the only reliable path.
2. **Preload the model on page navigation.** A 30-second wait after clicking "record" is unacceptable UX.
3. **Use producer/consumer for WebSocket handlers.** Sequential receive-then-process blocks the event loop during inference.
4. **Sliding window, not full re-transcription.** O(1) per window vs O(n) growing with session length.
5. **Don't change React component status during live recording.** Setting `status="processing"` hides the form and mic button. Keep `status="idle"` during recording.
6. **Check WebSocket.readyState inside async callbacks.** The state can change between the check and the `.then()` resolve.
7. **Feature flags matter for routing.** The Capture page must run on the provider-facing server (which has `record_audio=true`), not the pipeline server.
8. **`websocket.receive()` handles close frames; `receive_bytes()` doesn't.** Use the former in proxy/handler code to detect disconnects cleanly.
