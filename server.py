import os
import json
import yaml
import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from huggingface_hub import InferenceClient

from indexing import load_index
from retrieval import retriever as Retriever
from memory import Memory

load_dotenv()

with open("config.yaml") as f:
    config = yaml.safe_load(f)

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    index, chunks = load_index(config)
    _state["retriever"] = Retriever(config, index, chunks)
    _state["gen_client"] = InferenceClient(
        provider="featherless-ai",
        model=config["llm_model"],
        api_key=os.environ["HF_TOKEN"],
    )
    if config.get("use_memory", False):
        _state["memory"] = Memory(config)
    yield
    _state.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


# ── Models ────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []

class SchemeOut(BaseModel):
    name: str
    categories: str
    level: str
    benefits: str
    eligibility_snippet: str


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _chunk_to_scheme(c: dict) -> SchemeOut:
    return SchemeOut(
        name=c.get("title", ""),
        categories=c.get("category", "other"),
        level=c.get("location", "central"),
        benefits=c.get("benefit_summary", ""),
        eligibility_snippet=c.get("eligibility_summary", ""),
    )


def _dedup_by_title(chunks: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in chunks:
        t = c.get("title", "")
        if t not in seen:
            seen.add(t)
            out.append(c)
    return out


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


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


# ── Route ─────────────────────────────────────────────────────────────────────

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

    # Run blocking retrieval in a threadpool so we don't block the event loop
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
        insufficient = len(unique_chunks) < config.get("top_k", 5)
        search_query = getattr(retriever, "last_step_back_query", req.message)
        if insufficient:
            web_results = await loop.run_in_executor(
                None, retriever.web_search, search_query, n_web
            )

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

        # Model didn't emit </think> — buffer is the answer
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
