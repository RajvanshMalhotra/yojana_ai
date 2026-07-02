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

---

## Hindi Query Pipeline Overhaul (2026-07-03)

### Problem 1: Sequential retrieval + web search for Hindi queries

**Before:**
```python
# server.py /api/chat
if req.lang == "hi":
    chunks_raw = await loop.run_in_executor(None, lambda: retriever.retrieve(req.message))
    english_query = getattr(retriever, "last_step_back_query", req.message)
    web_candidates = await loop.run_in_executor(None, retriever.web_search, english_query, n_web)
else:
    retrieval_task = loop.run_in_executor(None, lambda: retriever.retrieve(req.message))
    web_task = loop.run_in_executor(None, retriever.web_search, req.message, n_web)
    chunks_raw, web_candidates = await asyncio.gather(retrieval_task, web_task)
```

**Root cause:** Hindi queries needed the English `step_back_query` (produced inside `retrieve()` by `_parse_rewrite_expand`) to avoid sending raw Devanagari to Firecrawl. Since `step_back_query` was only available *after* `retrieve()` returned, web search had to be sequential. This added the full web search latency (~0.5–1s) on top of retrieval.

**Fix:** Move `web_search` inside `retrieve()` itself — after `_parse_rewrite_expand` yields `step_back`, launch web search in the same `ThreadPoolExecutor` as `multi_query_retrieval` and `_classify_complexity`:

```python
# retrieval.py retrieve(query, n_web=0)
with ThreadPoolExecutor(max_workers=3) as pool:
    search_fut   = pool.submit(self.multi_query_retrieval, queries, filters)
    classify_fut = pool.submit(self._classify_complexity, query)
    web_fut      = pool.submit(self.web_search, step_back, n_web) if n_web > 0 else None
    candidates   = search_fut.result()
    is_complex   = classify_fut.result()
    self.last_web_results = web_fut.result() if web_fut else []
```

Server now calls `retriever.retrieve(query, n_web=n_web)` then reads `retriever.last_web_results` — identical path for Hindi and English, no branching.

**Result:** Retrieval + classification + web search all finish together. `embed+pinecone+classify+web=2.82s`, `pipeline_total=4.66s`.

---

### Problem 2: Web scrape timing out every single time

**Before:**
```python
# After getting search results, scraped the top URL for full page content
_fut = _ex.submit(app.scrape_url, entry["url"], formats=["markdown"])
scraped = _fut.result(timeout=4)   # always hit this
```

**Root cause:** `scrape_url` was added to give the LLM richer context than just the search snippet. But it consistently timed out (tried 5s, 10s, 4s — always failed). Every web search call silently wasted 4 seconds and produced nothing.

**Fix:** Removed the scrape entirely. Firecrawl's `search()` already returns a `description` snippet per result which is sufficient for the LLM to synthesise an answer. The scrape was a never-working optimisation.

```python
# Now:
return [
    {"title": r.title or "", "snippet": r.description or "", "url": r.url or ""}
    for r in hits
]
```

---

### Problem 3: Romanization model too slow (70B) and Hindi output breaking TTS

**Before:** `_romanize_hinglish()` hardcoded `model="llama-3.3-70b-versatile"` regardless of what was passed in. 70B on Groq is rate-limited and slower than 8B.

**Root cause:** 8B was tried first for romanization but failed (returned Devanagari). 70B was hardcoded as a workaround. The passed `model` argument was silently ignored.

**Fix:** Use the passed `model` argument (which is `draft_model = llama-3.1-8b-instant`). User confirmed 8B works fine for romanization on Groq playground. The system prompt change (`"Always respond in English."`) means the LLM usually outputs English anyway — romanization only catches the rare Devanagari fallback.

**Also removed:**
- `_HINDI_LANG_RULE` constant (heavy script-enforcement block in system prompt — unreliable, caused model confusion)
- Pre-commit trick (fake user/assistant turns to prime Roman script — added token overhead, still failed)
- Sequential Hindi branch in `/api/voice` (now uses same unified `retrieve(n_web=n_web)` path)

---

## Hindi TTS — Hinglish Translation Architecture (2026-07-03)

### What went wrong

Three approaches to get Hindi audio output all failed:

**Attempt 1 — Force LLM to generate Hinglish directly:**
Added `_HINDI_LANG_RULE` system prompt prefix + pre-commit trick (fake user/assistant turns priming Roman script). 8B model ignored both and still output Devanagari.

**Attempt 2 — Post-generation romanization (`_romanize_hinglish`):**
After LLM generated Devanagari, called Groq 70B to convert to Roman script. Then switched to 8B (user confirmed it worked in playground). In production, 8B returned Devanagari again. `_normalize_for_tts()` then stripped ALL Devanagari → only punctuation/spaces left → TTS received `','` and `':    -  ,  ,'` → garbage audio.

**Root cause of attempt 2 failure:** Romanization (Devanagari → Roman) is hard for 8B. The model confuses "write the same Hindi words in Latin letters" with "translate to English." In playground it worked on simple inputs; on real RAG responses with citations like `[1]`, `[W2]`, and mixed content it broke.

**Attempt 3 — Generate in English, romanize for TTS:**
Removed all Hindi generation instructions so LLM naturally outputs English. Then romanized English chunks at the TTS layer. But English → Roman script is a no-op — Rumik still needs Hindi input to speak Hindi.

### The fix that works

**Separate concerns: generation language vs TTS language.**

```
LLM generates English  →  chat display shows English
                       ↘
                         /api/voice/tts (lang=hi)
                           → _to_hinglish() translates English chunk → Hinglish
                           → Rumik speaks Hindi
```

`_to_hinglish()` uses Groq 8B with a clean translation prompt:
```python
"Translate this English text to Hinglish — Hindi written in Roman/Latin script only,
no Devanagari characters at all. Output only the translation."
```

**Why translation works but romanization didn't:** Translation (English → Hinglish) is a well-understood task the 8B model is trained on. Romanization (Devanagari → Roman script) is a rare task — the model wasn't reliably trained to just transliterate without changing words.

**Key:** `lang` is already sent by the frontend in every `/api/voice/tts` request body. The server reads it and only calls `_to_hinglish()` when `lang == "hi"`. English voice calls are unaffected.
