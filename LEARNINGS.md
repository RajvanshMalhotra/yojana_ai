# Yojan AI — System Evolution & Learnings

## The Core Architectural Problem (2026-07-02)

### What We Discovered

The voice and text pipelines were completely separate architectures producing drastically different quality outputs for the same query.

**Text pipeline (`/api/chat`)**
```
User types query
    → Full RAG retrieval:
        - _parse_rewrite_expand() — single LLM call: parse + rewrite + step-back + multi-query expand
        - Parallel hybrid search (Pinecone dense + BM25 sparse) across all query variants
        - RRF fusion → CrossEncoder rerank
        - Firecrawl web search in parallel (country=in, full page scrape of top result)
    → Complexity routing: 7B (simple queries) or 72B (multi-constraint queries) via featherless-ai
    → Full system prompt with web-aware synthesis instructions
    → SSE stream → frontend displays text + scheme cards
```

**Voice pipeline (`/ws/voice`)**
```
User speaks
    → Silero VAD → Sarvam STT (saaras:v3)
    → RAGContextProcessor:
        - Originally: fast=True (skipped rewrite, step-back, expand)
        - Single query, no web search (skip_web=True)
    → GroqLLMService (llama-3.1-8b-instant) — DIFFERENT model, DIFFERENT context format
    → SarvamTTS (bulbul:v3, suhani)
    → Audio to browser
```

### Why Quality Was So Different

| Factor | Text | Voice |
|--------|------|-------|
| Query processing | Full rewrite + step-back + 3 expanded variants | Single query (fast path) |
| Web search | Yes, with full page scrape | No (skip_web=True) |
| Model | 72B or 7B based on routing | Always Groq 8B |
| Context | Rich — chunks + web content | Sparse — chunks only |
| Retrieval | Multi-query hybrid → RRF → rerank | Single-query hybrid |

### What Went Wrong Specifically

1. **`fast=True` in voice**: Skipped all LLM query preprocessing — no step-back, no expanded queries, no category filters. Fewer, less relevant chunks retrieved.
2. **`skip_web=True` in voice**: Web search never ran. When chunks didn't have the answer (e.g. "AI startup schemes"), voice had nothing to fall back on. Text could supplement with Firecrawl results.
3. **Groq 8B vs featherless 72B**: 8B hallucinated freely — said "I couldn't find info about GENESIS" when GENESIS was literally chunk [3] in the context. Smaller model needs stricter prompting.
4. **Separate system prompt**: Voice prompt said "2-3 sentences max" which caused it to truncate and skip relevant schemes.
5. **Firecrawl `country` kwarg**: Added `country="in"` which wasn't supported by the installed SDK version, causing web search to silently return [] for all queries.

### The Right Architecture

Voice should not be a separate agentic system. It should be:

```
Mic input → STT → [exact same RAG pipeline as text] → TTS → Audio output
```

The generation quality should be identical. Only the input (audio→text) and output (text→audio) differ.

**Decision**: Replace the Pipecat WebSocket pipeline with the REST-based push-to-talk approach:
- MediaRecorder → POST `/api/voice` (Sarvam STT → full RAG → same LLM routing)
- Response text → POST `/api/voice/tts` (Sarvam bulbul:v3) → play audio
- Both Hindi and English use same path — no language-specific branching in frontend

This eliminates the dual-architecture problem entirely.

---

## Other Key Learnings

### Hallucination Root Cause (2026-07-02)
System prompt said: *"Never say the context 'doesn't mention' something — if a scheme is related, explain how it applies."*
This directly instructed the model to force-fit irrelevant schemes and invent connections. Fixed by: "Answer ONLY using facts from CONTEXT — do not invent scheme names, amounts, or eligibility."

### Thread Safety — HuggingFace Fast Tokenizer (2026-07-02)
`multi_query_retrieval` called `encode()` from multiple threads simultaneously → Rust backend "Already borrowed" panic.
Fix: batch-encode all queries serially first, then parallelize only the Pinecone network calls.

### Complexity Router Miscalibration (2026-07-02)
Original prompt: "Does this query have multiple constraints or eligibility criteria?"
"govt schemes for AI startups" was classified Complex → 72B used → 33s generation time.
Fix: Added few-shot examples to distinguish single-topic queries (simple) from multi-constraint eligibility queries (complex).

### Web Search Query Design (2026-07-02)
Originally appended " India government" to every query before searching.
"govt schemes for AI startups India government" returns generic results.
"govt schemes for AI startups" returns specific, targeted results — Google knows the intent.
Fix: Pass user query verbatim to Firecrawl.

### Sarvam TTS Speaker Compatibility (2026-07-02)
`suhani` voice not available on `bulbul:v2`. Available on `bulbul:v3`.
`bulbul:v2` speakers: anushka, abhilash, manisha, vidya, arya, karun, hitesh.
`bulbul:v3` speakers: suhani (and others — check API docs).

### VAD Timeout Too Short (2026-07-02)
Default VAD stop timeout (0.6s) was cutting off speech mid-sentence during natural pauses.
Fix: `audio_idle_timeout=1.5`, `SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=1.5)`.

### Estonian Hallucination (2026-07-02)
Groq's Llama model generated responses in Estonian when no language instruction was given for English.
Fix: Explicit "Always respond in English only. Never use any other language." in system prompt.

---

## Latency Optimisation (2026-07-02)

### Root Cause of 33s Total Latency
The pipeline had three serial LLM bottlenecks all hitting featherless-ai, which runs at ~50 tok/s:

| Step | Time | Fix |
|------|------|-----|
| `_parse_rewrite_expand` (featherless 7B) | ~6s | → Groq 8B (~0.3s) |
| `_classify_complexity` (featherless 7B, parallel) | ~3s | → Groq 8B (~0.2s) |
| Simple generation (featherless 7B, 250 tokens) | ~15s | → Groq 8B (~1s) |
| CrossEncoder rerank | ~1.5s | → disabled |
| Web scrape (no timeout) | up to 48s | → 5s timeout |

### Groq Key Variable Name
`.env` stores `GROQ_API_KEYS` (plural, comma-separated pool) not `GROQ_API_KEY`.
Code must check `GROQ_API_KEYS` first, fall back to `GROQ_API_KEY`, then featherless-ai.
A random key is picked per startup to distribute rate-limit load across the pool.

### Web Scrape Blocking Retrieval (2026-07-02)
`scrape_url()` had no timeout. A single slow/failed URL blocked the entire retrieval for 45+ seconds.
Fix: wrap in `concurrent.futures.ThreadPoolExecutor` with `future.result(timeout=5)`.

### Sarvam TTS Input Limit (2026-07-02)
`bulbul:v3` allows **max 3 items** in the `inputs` array per API call.
Long answers (400+ tokens) produce 8+ chunks when split at 500 chars → 400 Bad Request.
Fix: batch chunks into groups of 3, make `ceil(n/3)` API calls, concatenate WAV audio.
WAV concatenation: keep full 44-byte header from chunk[0], append raw PCM from chunks[1..N], rewrite ChunkSize and Subchunk2Size fields with `struct.pack_into`.

### Streaming TTS for Low Perceived Latency (2026-07-02)
Original approach: wait for full SSE stream to complete, then POST entire answer to TTS → 20+ s before audio.
Fix: detect sentence boundaries (`.`, `!`, `?`) in tokens as they arrive, fire TTS per sentence immediately.
Audio plays from a JS promise queue (`advanceTTS`) that chains `.onended` callbacks — seamless sequential playback.
First audio starts playing ~2s after first token, before generation even finishes.

### Web Search India Anchoring (2026-07-02)
Firecrawl returned US NSF, US Executive Orders, Google Cloud programs for "Government schemes for AI startups".
Fix: append `" India"` to every Firecrawl query (not `" India government"` — that over-constrains and returns generic results).
`country=` kwarg not supported by the installed Firecrawl SDK version — raises `TypeError`.
