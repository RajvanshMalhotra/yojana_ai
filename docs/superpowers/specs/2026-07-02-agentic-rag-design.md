# Agentic RAG — Design Spec
**Date:** 2026-07-02  
**Project:** Yojan AI — Indian Government Scheme Finder  

---

## Problem

The Pinecone + BM25 index only contains schemes explicitly scraped and stored. Queries like "schemes for AI startups" or "grants for clean energy" may return partial or no matches even though relevant government programmes exist. When the database comes up short the LLM currently says "I couldn't find a matching scheme" — a bad user experience.

---

## Goal

Let the LLM decide whether the retrieved context is sufficient. If it isn't, automatically supplement with a live web search and fold the web findings into the generated answer. No new UI components — web content surfaces only inside the LLM's streamed text response.

---

## Approach — Parallel Speculative (Option C)

After `retrieve()` returns RAG chunks, fire two tasks **in parallel**:

1. **Judge call** — fast LLM call with a compact context (titles + benefit summaries only). Prompt asks for exactly one word: `sufficient` or `insufficient`.
2. **Web search** — DuckDuckGo search (`duckduckgo-search`, free, no API key needed) using the step-back query + `"India government"`. Returns top-N `{title, snippet, url}` dicts.

Both tasks run concurrently via `asyncio.gather` with `run_in_executor`. When both finish:

- `sufficient` → discard web results, generate from RAG context only.  
- `insufficient` → merge RAG chunks + web snippets into one combined prompt, generate from that.

The web search runs speculatively, so the unhappy path costs only **one extra LLM call** (the judge), not two.

---

## Data Flow

```
retrieve(query) → unique_chunks
                        │
          ┌─────────────┴─────────────┐  parallel
          ▼                           ▼
   _judge(chunks, query)       web_search(step_back_query)
   → "sufficient" /            → [{title, snippet, url}, …]
     "insufficient"
          │
          └──── if insufficient ────► _build_prompt(chunks + web_results)
          └──── if sufficient   ────► _build_prompt(chunks)
                                              │
                                       stream answer via SSE
```

---

## Components

### `retrieval.py` — `web_search()` method

```python
def web_search(self, query: str, n: int = 4) -> list[dict]:
    """Returns [{title, snippet, url}, …] via DuckDuckGo."""
```

- Appends `"India government"` to every search query to bias results toward Indian schemes and grants.
- Returns an empty list on any exception (network error, rate limit) — never raises.

### `server.py` — parallel judge + search

New private function `_judge(chunks, query)`:
- Builds a compact context string: `[i] title — benefit_summary` for each chunk.
- Single LLM call, system prompt: *"Reply with exactly one word: sufficient or insufficient."*
- Any response that isn't exactly `"sufficient"` is treated as `"insufficient"` (fail toward more context).

In the `chat` endpoint, after dedup:
```python
judge_task  = loop.run_in_executor(None, _judge, unique_chunks, req.message)
search_task = loop.run_in_executor(None, retriever.web_search, req.message)
verdict, web_results = await asyncio.gather(judge_task, search_task)
```

Special case: if `unique_chunks` is empty, skip the judge and treat verdict as `"insufficient"` immediately.

### `server.py` — `_build_prompt()` update

Gains an optional `web_results: list[dict]` parameter. When present, appends:

```
WEB RESULTS:
[W1] <title>
<snippet>

[W2] …
```

### System prompt update

When web results are included, the system prompt instructs the LLM:
- Cite database schemes as `[1]`, `[2]`, etc.
- Cite web results as `[W1]`, `[W2]`, etc.
- Integrate both naturally into the answer.

### `config.yaml` — new flags

```yaml
use_agentic_rag: True
agentic_web_results: 4   # number of DuckDuckGo results to fetch
```

### `requirements.txt`

Add `duckduckgo-search`.

---

## Edge Cases

| Situation | Behaviour |
|---|---|
| Web search throws (network / rate limit) | Catch exception, return `[]`, proceed with RAG-only context |
| RAG returns 0 chunks | Skip judge, go straight to web search |
| Judge returns unexpected text | Treat as `"insufficient"` |
| Web search returns 0 results | Generate from RAG context only (same as `sufficient` path) |
| `use_agentic_rag: False` | Skip judge and web search entirely — existing behaviour unchanged |

---

## What Does Not Change

- SSE streaming protocol (`type: schemes`, `type: token`, `[DONE]`) — unchanged.
- React frontend — no changes required.
- `SchemeCard` UI — web results never become cards.
- `retriever.retrieve()` signature — unchanged.
