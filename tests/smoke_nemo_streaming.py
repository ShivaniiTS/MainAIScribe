"""
Smoke test for NemoStreamingServer.
Runs in simulation mode (no NeMo/GPU needed).
Checks: partial output, no trailing period, hotword correction, config=None safety.
"""
import asyncio
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
from mcp_servers.asr.base import ASRConfig


async def run():
    server = NemoStreamingServer(
        device="cpu",
        chunk_size_ms=160,
        hotwords=["HTN", "DM", "CAD"],   # inline hotwords
    )
    session_id = "smoke_test"

    # PCM silence chunk
    chunk = b"\x00\x00" * server.chunk_samples

    # Need enough chunks to exceed STREAM_WINDOW_S (1s) of audio
    num_chunks = int((server.STREAM_WINDOW_S * 1000) / server.chunk_size_ms) + 3
    outputs = []

    print(f"Sending {num_chunks} chunks ({num_chunks * server.chunk_size_ms}ms of audio)...")

    for i in range(num_chunks):
        async for partial in server.transcribe_stream(chunk, session_id, config=ASRConfig()):
            print(f"  PARTIAL -> text={partial.text!r}  start={partial.start_ms}ms  end={partial.end_ms}ms  conf={partial.confidence}")
            outputs.append(partial)
            # Verify: no trailing period from window boundary
            assert not partial.text.endswith("."), f"Trailing period found: {partial.text!r}"
        await asyncio.sleep(0)

    print(f"\nconfig=None safety check...")
    server2 = NemoStreamingServer(device="cpu", chunk_size_ms=160)
    session2 = "smoke_none"
    for i in range(num_chunks):
        async for partial in server2.transcribe_stream(chunk, session2, config=None):
            print(f"  (config=None) PARTIAL -> {partial.text!r}")

    print(f"\nHotword correction test...")
    corrected = server._apply_hotword_corrections("patient has htn and dm type 2")
    print(f"  Input:  'patient has htn and dm type 2'")
    print(f"  Output: {corrected!r}")
    assert "HTN" in corrected, "HTN not corrected"
    assert "DM" in corrected, "DM not corrected"

    print(f"\nSMOKE TEST PASSED. Emitted {len(outputs)} partials.")


if __name__ == "__main__":
    asyncio.run(run())
