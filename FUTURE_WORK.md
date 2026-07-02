# Future Work

## Real-time WebSocket Voice Pipeline

Currently the voice flow uses REST push-to-talk (`/api/voice/stt` → RAG → `/api/voice/tts`). This works but has a round-trip gap between recording and playback.

**Goal:** Replace with a persistent WebSocket connection so audio streams in real-time — no record-then-POST, just continuous duplex audio.

**Two options to evaluate:**

1. **Pipecat `/ws/voice`** — already scaffolded in `server.py`. Full pipeline: SileroVAD → Sarvam STT → RAGContextProcessor → LLM → Sarvam TTS. Most feature-rich (VAD, turn detection, sentence-level TTS streaming). Requires `pipecat-ai[silero,websocket]` which is a heavy dependency (~1GB with torch) — blocked Render free tier deployment.

2. **Custom lightweight WebSocket** — no Pipecat. Frontend streams audio chunks over WebSocket, server does STT → RAG → TTS and streams audio back. Simpler, no heavy deps, works on any host.

**Blocker for option 1:** `pipecat-ai` pulls in full PyTorch which OOMs on Render free tier (512MB RAM) and bloats the Docker image. Needs a paid host (EC2 t3.small, Railway, Fly.io) or a CPU-only torch install.

**Recommended next step:** Implement option 2 (custom WebSocket) — gets the efficiency of persistent connections without the Pipecat overhead, and works on Render free tier.
