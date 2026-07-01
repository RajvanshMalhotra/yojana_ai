# Agentic RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an LLM judge that decides post-retrieval whether the RAG context is sufficient; if not, a parallel DuckDuckGo web search runs and its results are blended into the generation prompt.

**Architecture:** After `retrieve()` returns chunks, the server fires a judge LLM call and a DuckDuckGo web search in parallel via `asyncio.gather + run_in_executor`. If the judge says "insufficient" (or RAG returned zero chunks), web snippets are appended to the generation prompt and cited as `[W1]`, `[W2]`.

**Tech Stack:** Python 3.13, FastAPI, `duckduckgo-search`, `huggingface_hub.InferenceClient`, existing Pinecone + BM25 retriever.

## Global Constraints

- All Python changes must stay compatible with the existing `retriever` class in `retrieval.py` — do not change its `__init__` signature or `retrieve()` return type.
- `use_agentic_rag: False` must be a no-op — existing behaviour unchanged when flag is off.
- Web search failures must never raise; always return `[]` silently.
- The SSE protocol (`type: schemes`, `type: token`, `data: [DONE]`) must not change.
- No frontend changes required.

---

### Task 1: Add dependency and config flags

**Files:**
- Modify: `requirements.txt`
- Modify: `config.yaml`

**Interfaces:**
- Produces: `config["use_agentic_rag"]` (bool), `config["agentic_web_results"]` (int)

- [ ] **Step 1: Add duckduckgo-search to requirements**

Open `requirements.txt` and append:
```
duckduckgo-search
```

- [ ] **Step 2: Install it**

```bash
pip install duckduckgo-search
```

Expected: installs without error. Verify: `python -c "from duckduckgo_search import DDGS; print('ok')"` prints `ok`.

- [ ] **Step 3: Add config flags**

In `config.yaml`, add after `use_rerank: True`:
```yaml
use_agentic_rag: True
agentic_web_results: 4
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt config.yaml
git commit -m "feat: add duckduckgo-search dependency and agentic RAG config flags"
```

---

### Task 2: Add `web_search()` to `retrieval.py`

**Files:**
- Modify: `retrieval.py` — add method to `retriever` class after `keyword_search()`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `retriever.web_search(query: str, n: int = 4) -> list[dict]`
  - Each dict: `{"title": str, "snippet": str, "url": str}`
  - Returns `[]` on any exception

- [ ] **Step 1: Create test file**

Create `tests/test_web_search.py`:
```python
import pytest
from unittest.mock import patch, MagicMock

def make_retriever():
    """Build a minimal retriever without loading real models."""
    import sys, types
    # Stub heavy deps so import doesn't load GPU models
    for mod in ["sentence_transformers", "rank_bm25", "huggingface_hub"]:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)
    sys.modules["sentence_transformers"].SentenceTransformer = MagicMock()
    sys.modules["sentence_transformers"].CrossEncoder = MagicMock()
    sys.modules["rank_bm25"].BM25Okapi = MagicMock()
    sys.modules["huggingface_hub"].InferenceClient = MagicMock()

    import importlib, os
    os.environ.setdefault("HF_TOKEN", "test")
    import retrieval
    importlib.reload(retrieval)

    config = {
        "top_k": 5, "embedding_model": "m", "llm_model": "m",
        "use_query_rewrite": False, "use_multi_query": False,
        "use_rerank": False, "use_step_back": False,
        "use_agentic_rag": True, "agentic_web_results": 3,
        "max_expanded_queries": 3,
    }
    index = MagicMock()
    chunks = [{"chunk_id": "1", "text": "test", "title": "T", "category": "health",
               "beneficiaries": [], "benefit_summary": "", "eligibility_summary": ""}]
    r = retrieval.retriever.__new__(retrieval.retriever)
    r.config = config
    r.use_query_rewrite = False
    r.use_multi_query = False
    r.use_rerank = False
    r.use_step_back = False
    r.use_agentic_rag = True
    r.agentic_web_results = 3
    r.top_k = 5
    r.max_expanded_queries = 3
    r.index = index
    r.chunks = chunks
    r.bm25 = MagicMock()
    r.embedding_model = MagicMock()
    r.reranker = MagicMock()
    r.client = MagicMock()
    return r


def test_web_search_returns_results():
    r = make_retriever()
    fake_results = [
        {"title": "Startup India", "href": "https://example.com", "body": "Grant for startups"},
        {"title": "DPIIT Fund",    "href": "https://dpiit.gov.in", "body": "Seed funding scheme"},
    ]
    with patch("duckduckgo_search.DDGS") as MockDDGS:
        instance = MockDDGS.return_value.__enter__.return_value
        instance.text.return_value = fake_results
        results = r.web_search("AI startup grants", n=2)

    assert len(results) == 2
    assert results[0]["title"] == "Startup India"
    assert results[0]["snippet"] == "Grant for startups"
    assert results[0]["url"] == "https://example.com"


def test_web_search_returns_empty_on_exception():
    r = make_retriever()
    with patch("duckduckgo_search.DDGS", side_effect=Exception("network error")):
        results = r.web_search("anything")
    assert results == []


def test_web_search_appends_india_government():
    r = make_retriever()
    with patch("duckduckgo_search.DDGS") as MockDDGS:
        instance = MockDDGS.return_value.__enter__.return_value
        instance.text.return_value = []
        r.web_search("AI startup grants", n=3)
        call_args = instance.text.call_args
    assert "India government" in call_args[0][0]
```

- [ ] **Step 2: Run tests — expect failure**

```bash
cd /Users/rajvanshmalhotra/adv_rag && python -m pytest tests/test_web_search.py -v
```

Expected: `ImportError` or `AttributeError` — `web_search` does not exist yet.

- [ ] **Step 3: Add `web_search()` to `retrieval.py`**

In `retrieval.py`, add the `__init__` flag after `self.use_step_back`:
```python
self.use_agentic_rag    = config.get("use_agentic_rag", False)
self.agentic_web_results = config.get("agentic_web_results", 4)
```

Then add the method after `keyword_search()` (before `_matches_filter`):
```python
def web_search(self, query: str, n: int = 4) -> list[dict]:
    """DuckDuckGo fallback — returns [{title, snippet, url}, …] or [] on failure."""
    from duckduckgo_search import DDGS
    try:
        with DDGS() as ddgs:
            raw = ddgs.text(f"{query} India government", max_results=n)
        return [{"title": r["title"], "snippet": r["body"], "url": r["href"]} for r in raw]
    except Exception:
        return []
```

- [ ] **Step 4: Run tests — expect pass**

```bash
python -m pytest tests/test_web_search.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add retrieval.py tests/test_web_search.py
git commit -m "feat: add web_search() method to retriever with DuckDuckGo"
```

---

### Task 3: Add `_judge()` to `server.py`

**Files:**
- Modify: `server.py` — add `_judge()` function after `_dedup_by_title()`

**Interfaces:**
- Consumes: `_state["gen_client"]` (InferenceClient), `unique_chunks: list[dict]`, `query: str`
- Produces: `_judge(chunks: list[dict], query: str) -> str` — returns `"sufficient"` or `"insufficient"`

- [ ] **Step 1: Create test file**

Create `tests/test_judge.py`:
```python
from unittest.mock import MagicMock, patch
import importlib


def _make_mock_client(response_text: str):
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = response_text
    client.chat.completions.create.return_value.choices = [choice]
    return client


def test_judge_returns_sufficient():
    import server
    with patch.dict(server._state, {"gen_client": _make_mock_client("sufficient")}):
        result = server._judge(
            [{"title": "PM-KISAN", "benefit_summary": "Cash to farmers"}],
            "schemes for farmers"
        )
    assert result == "sufficient"


def test_judge_returns_insufficient():
    import server
    with patch.dict(server._state, {"gen_client": _make_mock_client("insufficient")}):
        result = server._judge([], "schemes for AI drones")
    assert result == "insufficient"


def test_judge_treats_unexpected_as_insufficient():
    import server
    with patch.dict(server._state, {"gen_client": _make_mock_client("I don't know")}):
        result = server._judge(
            [{"title": "Any", "benefit_summary": "Anything"}],
            "some query"
        )
    assert result == "insufficient"
```

- [ ] **Step 2: Run tests — expect failure**

```bash
python -m pytest tests/test_judge.py -v
```

Expected: `AttributeError: module 'server' has no attribute '_judge'`

- [ ] **Step 3: Add `_judge()` to `server.py`**

Add after `_sse()`:
```python
def _judge(chunks: list[dict], query: str) -> str:
    """
    Fast LLM call that decides if RAG context fully answers the query.
    Returns 'sufficient' or 'insufficient'.
    """
    if not chunks:
        return "insufficient"

    compact = "\n".join(
        f"[{i}] {c.get('title','')} — {c.get('benefit_summary','')}"
        for i, c in enumerate(chunks, 1)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a relevance judge. Given a question and a list of retrieved scheme summaries, "
                "decide if the context fully answers the question. "
                "Reply with exactly one word: sufficient or insufficient."
            ),
        },
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{compact}"},
    ]
    response = _state["gen_client"].chat.completions.create(messages=messages)
    verdict = response.choices[0].message.content.strip().lower()
    if "</think>" in verdict:
        verdict = verdict.split("</think>", 1)[-1].strip()
    return "sufficient" if verdict == "sufficient" else "insufficient"
```

- [ ] **Step 4: Run tests — expect pass**

```bash
python -m pytest tests/test_judge.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add server.py tests/test_judge.py
git commit -m "feat: add _judge() to server for agentic RAG sufficiency check"
```

---

### Task 4: Update `_build_prompt()` and `_build_messages()`

**Files:**
- Modify: `server.py` — update `_build_prompt()` and `_build_messages()`

**Interfaces:**
- Consumes: `web_results: list[dict]` — each `{"title": str, "snippet": str, "url": str}`
- Produces: `_build_prompt(query, chunks, web_results=[]) -> str`
  `_build_messages(prompt, recent_msgs, summary, has_web=False) -> list[dict]`

- [ ] **Step 1: Write failing tests**

Create `tests/test_build_prompt.py`:
```python
from server import _build_prompt, _build_messages


CHUNKS = [{"title": "PM-KISAN", "benefit_summary": "Cash", "eligibility_summary": "Farmers", "text": "Details here"}]
WEB    = [{"title": "Startup Grant", "snippet": "Rs 10L for startups", "url": "https://example.com"}]


def test_build_prompt_no_web():
    result = _build_prompt("schemes for farmers", CHUNKS)
    assert "CONTEXT:" in result
    assert "[1] PM-KISAN" in result
    assert "WEB RESULTS" not in result


def test_build_prompt_with_web():
    result = _build_prompt("schemes for AI startups", CHUNKS, WEB)
    assert "WEB RESULTS:" in result
    assert "[W1] Startup Grant" in result
    assert "Rs 10L for startups" in result
    assert "https://example.com" in result


def test_build_messages_without_web_has_no_W_cite_instruction():
    msgs = _build_messages("prompt", [], "", has_web=False)
    assert "[W1]" not in msgs[0]["content"]


def test_build_messages_with_web_has_W_cite_instruction():
    msgs = _build_messages("prompt", [], "", has_web=True)
    assert "[W1]" in msgs[0]["content"] or "W1" in msgs[0]["content"]
```

- [ ] **Step 2: Run tests — expect failure**

```bash
python -m pytest tests/test_build_prompt.py -v
```

Expected: `TypeError` — `_build_prompt` doesn't accept `web_results` yet.

- [ ] **Step 3: Update `_build_prompt()` in `server.py`**

Replace the existing `_build_prompt` function:
```python
def _build_prompt(query: str, chunks: list[dict], web_results: list[dict] = []) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        block = [f"[{i}] {c.get('title', 'Scheme')}"]
        if c.get("benefit_summary"):
            block.append(f"Benefit: {c['benefit_summary']}")
        if c.get("eligibility_summary"):
            block.append(f"Eligibility: {c['eligibility_summary']}")
        block.append(f"Details: {c['text']}")
        parts.append("\n".join(block))
    context = "CONTEXT:\n" + "\n\n".join(parts) if parts else "CONTEXT:\n(none)"

    if web_results:
        web_parts = []
        for i, r in enumerate(web_results, 1):
            web_parts.append(f"[W{i}] {r['title']}\n{r['snippet']}\nSource: {r['url']}")
        web_section = "WEB RESULTS:\n" + "\n\n".join(web_parts)
        return f"{context}\n\n{web_section}\n\nQUESTION: {query}"

    return f"{context}\n\nQUESTION: {query}"
```

- [ ] **Step 4: Update `_build_messages()` in `server.py`**

Replace the existing `_build_messages` function:
```python
def _build_messages(prompt: str, recent_msgs: list, summary: str, has_web: bool = False) -> list[dict]:
    if has_web:
        system = (
            "You are Yojan AI, a helpful assistant for Indian government scheme discovery. "
            "Answer in 2-4 plain sentences using the CONTEXT and WEB RESULTS provided. "
            "Cite database schemes as [1], [2] and web results as [W1], [W2]. "
            "If the context contains broadly relevant schemes, describe how they can help. "
            "Do NOT use markdown, blockquotes, bullet points, or headers — plain text only."
        )
    else:
        system = (
            "You are Yojan AI, a helpful assistant for Indian government scheme discovery. "
            "Answer in 2-4 plain sentences using the CONTEXT provided. "
            "Cite each scheme by its number like [1] or [2]. "
            "If the context contains schemes that are broadly relevant — even if not an exact keyword match — describe how they can help. "
            "Only say 'I couldn't find a matching scheme in my database.' if the context is completely unrelated to the question. "
            "Do NOT use markdown, blockquotes, bullet points, or headers — plain text only."
        )
    messages = [{"role": "system", "content": system}]
    if summary:
        messages.append({"role": "user", "content": f"Conversation summary so far: {summary}"})
    messages.extend(recent_msgs)
    messages.append({"role": "user", "content": prompt})
    return messages
```

- [ ] **Step 5: Run tests — expect pass**

```bash
python -m pytest tests/test_build_prompt.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add server.py tests/test_build_prompt.py
git commit -m "feat: update _build_prompt and _build_messages to handle web results"
```

---

### Task 5: Wire parallel judge + web search in the `chat` endpoint

**Files:**
- Modify: `server.py` — update the `chat` async function

**Interfaces:**
- Consumes:
  - `_judge(chunks, query) -> str` (Task 3)
  - `retriever.web_search(query, n) -> list[dict]` (Task 2)
  - `_build_prompt(query, chunks, web_results=[]) -> str` (Task 4)
  - `_build_messages(prompt, recent_msgs, summary, has_web=False) -> list[dict]` (Task 4)
  - `config["use_agentic_rag"]` (Task 1)
  - `config["agentic_web_results"]` (Task 1)
- Produces: unchanged SSE stream (`type: schemes`, `type: token`, `[DONE]`)

- [ ] **Step 1: Update the `chat` endpoint in `server.py`**

Replace the section from `if not unique_chunks:` to `messages = _build_messages(...)` with:

```python
    use_agentic = config.get("use_agentic_rag", False)

    if not unique_chunks and not use_agentic:
        async def _empty():
            yield _sse({"type": "schemes", "schemes": []})
            yield _sse({"type": "token", "content": "I couldn't find a matching scheme in my database."})
            yield "data: [DONE]\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    web_results: list[dict] = []
    if use_agentic:
        n_web = config.get("agentic_web_results", 4)
        judge_task  = loop.run_in_executor(None, _judge, unique_chunks, req.message)
        search_task = loop.run_in_executor(
            None, retriever.web_search, req.message, n_web
        )
        verdict, web_candidates = await asyncio.gather(judge_task, search_task)
        if verdict == "insufficient":
            web_results = web_candidates

    prompt   = _build_prompt(req.message, unique_chunks, web_results)
    messages = _build_messages(prompt, recent_msgs, summary, has_web=bool(web_results))
```

The full updated `chat` function now reads:

```python
@app.post("/api/chat")
async def chat(req: ChatRequest):
    memory: Memory | None = _state.get("memory")
    retriever: Retriever  = _state["retriever"]
    gen_client: InferenceClient = _state["gen_client"]

    history = [m.model_dump() for m in req.history]

    if memory:
        recent_msgs = memory.get_recent(history)
        summary = memory.summarize(history) if memory.should_compress() else ""
    else:
        recent_msgs, summary = [], ""

    loop = asyncio.get_event_loop()
    chunks = await loop.run_in_executor(
        None, lambda: retriever.retrieve(req.message, recent_messages=recent_msgs)
    )
    unique_chunks = _dedup_by_title(chunks)
    schemes = [_chunk_to_scheme(c) for c in unique_chunks]

    use_agentic = config.get("use_agentic_rag", False)

    if not unique_chunks and not use_agentic:
        async def _empty():
            yield _sse({"type": "schemes", "schemes": []})
            yield _sse({"type": "token", "content": "I couldn't find a matching scheme in my database."})
            yield "data: [DONE]\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    web_results: list[dict] = []
    if use_agentic:
        n_web = config.get("agentic_web_results", 4)
        judge_task  = loop.run_in_executor(None, _judge, unique_chunks, req.message)
        search_task = loop.run_in_executor(
            None, retriever.web_search, req.message, n_web
        )
        verdict, web_candidates = await asyncio.gather(judge_task, search_task)
        if verdict == "insufficient":
            web_results = web_candidates

    prompt   = _build_prompt(req.message, unique_chunks, web_results)
    messages = _build_messages(prompt, recent_msgs, summary, has_web=bool(web_results))

    def _stream_gen():
        """Synchronous SSE generator — runs in FastAPI's threadpool."""
        yield _sse({"type": "schemes", "schemes": [s.model_dump() for s in schemes]})

        buffer        = ""
        thinking_done = False
        full_answer   = ""

        for chunk in gen_client.chat.completions.create(messages=messages, stream=True):
            token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if not token:
                continue

            if not thinking_done:
                buffer += token
                if "</think>" in buffer:
                    thinking_done = True
                    after = buffer.split("</think>", 1)[-1].lstrip("\n").strip()
                    if after:
                        full_answer += after
                        yield _sse({"type": "token", "content": after})
            else:
                full_answer += token
                yield _sse({"type": "token", "content": token})

        if not thinking_done and buffer:
            full_answer = buffer
            yield _sse({"type": "token", "content": buffer})

        yield "data: [DONE]\n\n"

        if memory:
            updated = history + [
                {"role": "user",      "content": req.message},
                {"role": "assistant", "content": full_answer},
            ]
            memory.track_message({"content": req.message})
            memory.track_message({"content": full_answer})
            memory.save(updated)

    return StreamingResponse(
        _stream_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 2: Smoke-test with the running server**

Restart the server:
```bash
lsof -ti :8000 | xargs kill -9 2>/dev/null
cd /Users/rajvanshmalhotra/adv_rag && uvicorn server:app --reload --port 8000
```

Send a test request:
```bash
curl -s -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "any schemes for AI startups?", "history": []}' | head -20
```

Expected: SSE events appear — first `data: {"type": "schemes", ...}`, then `data: {"type": "token", ...}` lines. If `use_agentic_rag: True` and judge returns "insufficient", web snippets will be cited as `[W1]` etc. in the token stream.

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add server.py
git commit -m "feat: wire parallel judge + web search in chat endpoint (agentic RAG)"
```
