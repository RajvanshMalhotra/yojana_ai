import os
import re
import json
import time
import threading

import yaml
import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from huggingface_hub import InferenceClient

# Pipecat pipeline imports — optional; REST voice endpoints work without it
try:
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineTask, PipelineParams
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMContextFrame
    from pipecat.frames.frames import (
        Frame, TextFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame,
    )
    from pipecat.services.sarvam.stt import SarvamSTTService
    from pipecat.services.sarvam.tts import SarvamTTSService
    from pipecat.services.sarvam.tts import Language as SarvamLanguage
    from pipecat.transcriptions.language import Language as PipecatLanguage
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
    from pipecat.serializers.protobuf import ProtobufFrameSerializer
    from pipecat.processors.audio.vad_processor import VADProcessor
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import SpeechTimeoutUserTurnStopStrategy
    from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
    from pipecat.processors.aggregators.llm_response_universal import UserTurnStrategies, LLMUserAggregatorParams
    PIPECAT_AVAILABLE = True
except ModuleNotFoundError:
    PIPECAT_AVAILABLE = False
    FrameProcessor = object  # stub so class definitions below don't crash

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
    _state["gen_model"] = config["llm_model"]

    # Draft client: Groq llama-3.1-8b-instant for simple queries (~800 tok/s vs ~50 tok/s on featherless)
    _groq_keys = [k.strip() for k in os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", "")).split(",") if k.strip()]
    if _groq_keys:
        import random
        from groq import Groq as _Groq
        _state["draft_client"] = _Groq(api_key=random.choice(_groq_keys))
        _state["draft_model"]  = "llama-3.1-8b-instant"
        print("[startup] draft client: Groq llama-3.1-8b-instant")
    else:
        draft_model = config.get("query_llm_model", config["llm_model"])
        _state["draft_client"] = InferenceClient(
            provider="featherless-ai",
            model=draft_model,
            api_key=os.environ["HF_TOKEN"],
        )
        _state["draft_model"] = draft_model
        print(f"[startup] draft client: featherless-ai {draft_model}")
    # Gemini client for Hinglish translation (gemini-2.5-flash)
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        from google import genai as _genai
        _state["gemini_client"] = _genai.Client(api_key=gemini_key)
        print("[startup] gemini client: gemini-2.5-flash")

    if config.get("use_memory", False):
        _state["memory"] = Memory(config)

    yield
    _state.clear()


app = FastAPI(lifespan=lifespan)

_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── Models ────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []
    lang: str | None = None  # "en" | "hi"
    voice: bool = False       # True when the frontend is in voice mode

class SchemeOut(BaseModel):
    name: str
    categories: str
    level: str
    benefits: str
    eligibility_snippet: str
    source_url: str = ""


# ── Constants ─────────────────────────────────────────────────────────────────

# TTS number normalisation
_CURRENCY_RE = re.compile(r'(₹|Rs\.?\s*|INR\s*)([\d,]+(?:\.\d+)?)')
_LARGE_NUM_RE = re.compile(r'\b([\d,]{4,})\b')
_PERCENT_RE  = re.compile(r'(\d+(?:\.\d+)?)%')
_CITE_RE     = re.compile(r'\[\w*\d+\]')   # [1], [W2], etc.

def _indian_words(n: int) -> str:
    """Integer → Indian number-system words (crore / lakh / thousand)."""
    if n == 0:
        return "zero"
    parts = []
    if n >= 10_000_000:
        parts.append(f"{n // 10_000_000} crore"); n %= 10_000_000
    if n >= 100_000:
        parts.append(f"{n // 100_000} lakh");     n %= 100_000
    if n >= 1_000:
        parts.append(f"{n // 1_000} thousand");   n %= 1_000
    if n > 0:
        parts.append(str(n))
    return " ".join(parts)

def _normalize_for_tts(text: str) -> str:
    """Clean text before sending to TTS: strip markdown and normalise numbers."""
    # Strip markdown formatting (model sometimes ignores the no-markdown instruction)
    text = re.sub(r'#{1,6}\s+', '', text)                         # ## Headers
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text, flags=re.DOTALL) # **bold**
    text = re.sub(r'\*(.+?)\*', r'\1', text, flags=re.DOTALL)     # *italic*
    text = re.sub(r'`(.+?)`', r'\1', text, flags=re.DOTALL)       # `code`
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)  # bullet points
    text = re.sub(r'\n+', ' ', text)                               # newlines → space
    # Devanagari kept — sending raw to Rumik to test native support
    text = _CITE_RE.sub('', text)       # strip citation markers [1], [W2]

    # When text is Hindi (contains Devanagari), replace English words Rumik mispronounces
    if re.search(r'[ऀ-ॿ]', text):
        # Chandrabindu ँ → anusvara ं — Rumik handles anusvara more cleanly
        text = text.replace('ँ', 'ं')   # ँ → ं

        _HI_REPLACEMENTS = [
            (r'स्कीम्स', 'schemes'),
            (r'स्कीम',   'scheme'),
            (r'योजनाएं', 'yojnaaen'),
            (r'योजनाओं', 'yojnaon'),
            (r'योजनाएँ', 'yojnaaen'),
            (r'योजना',   'yojna'),
            (r'\bbenefits\b', 'लाभ'),
            (r'\beligibility\b', 'पात्रता'),
            (r'\bgovernment\b', 'सरकार'),
            (r'\bapplication\b', 'आवेदन'),
        ]
        for pat, rep in _HI_REPLACEMENTS:
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)

    # Expand common government scheme acronyms so TTS pronounces them naturally
    _ACRONYMS = {
        r'\bPMAY\b': 'Pradhan Mantri Awaas Yojana',
        r'\bPMKVY\b': 'Pradhan Mantri Kaushal Vikas Yojana',
        r'\bPMFBY\b': 'Pradhan Mantri Fasal Bima Yojana',
        r'\bPMJDY\b': 'Pradhan Mantri Jan Dhan Yojana',
        r'\bPMGSY\b': 'Pradhan Mantri Gram Sadak Yojana',
        r'\bMGNREGA\b': 'Mahatma Gandhi National Rural Employment Guarantee Act',
        r'\bMGNEGS\b': 'Mahatma Gandhi National Employment Guarantee Scheme',
        r'\bPMMY\b': 'Pradhan Mantri Mudra Yojana',
        r'\bNSP\b': 'National Scholarship Portal',
        r'\bWEP\b': 'Women Entrepreneurship Platform',
        r'\bSTARTUP\s+INDIA\b': 'Startup India',
        r'\bSTAND-?UP\s+INDIA\b': 'Stand Up India',
        r'\bTREAD\b': 'Trade Related Entrepreneurship Assistance and Development',
        r'\bDPIT\b': 'Department for Promotion of Industry and Internal Trade',
    }
    for pattern, replacement in _ACRONYMS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    def _cur(m):
        try:
            val  = float(m.group(2).replace(',', ''))
            base = _indian_words(int(val))
            frac = round(val - int(val), 2)
            return f"{base} rupees" + (f" and {int(frac*100)} paise" if frac else "")
        except Exception:
            return m.group(0)
    text = _CURRENCY_RE.sub(_cur, text)

    text = _PERCENT_RE.sub(lambda m: f"{m.group(1)} percent", text)

    def _num(m):
        try:
            n = int(m.group(1).replace(',', ''))
            return _indian_words(n)
        except Exception:
            return m.group(0)
    text = _LARGE_NUM_RE.sub(_num, text)

    return text.strip()

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
            body = r.get("content") or r.get("snippet", "")
            web_parts.append(f"[W{i}] {r['title']}\n{body}\nSource: {r['url']}")
        web_section = "WEB RESULTS:\n" + "\n\n".join(web_parts)
        return f"{context}\n\n{web_section}\n\nQUESTION: {query}"

    return f"{context}\n\nQUESTION: {query}"


def _build_messages(prompt: str, recent_msgs: list, summary: str,
                    has_web: bool = False, lang: str | None = None) -> list[dict]:
    if lang == "hi":
        if has_web:
            system = (
                "आप Yojan AI हैं, भारतीय सरकारी योजनाओं के विशेषज्ञ। "
                "आपके पास दो स्रोत हैं: CONTEXT (verified scheme database) और WEB RESULTS (live search)। "
                "CONTEXT को primary authoritative source मानें — योजनाओं को [1], [2] आदि से cite करें। "
                "WEB RESULTS से current details जोड़ें — [W1], [W2] से cite करें। "
                "दोनों को मिलाकर एक स्वाभाविक उत्तर दें। "
                "योजनाओं के नाम, benefit amounts, या eligibility criteria जो किसी भी स्रोत में नहीं हैं, वो मत बनाइए। "
                "Markdown, bullet points, या headers बिल्कुल मत use करें — plain flowing text only। "
                "हमेशा हिंदी में उत्तर दें। "
                "योजनाओं को 'scheme' या 'स्कीम' मत कहें — हमेशा 'योजना' या 'योजनाएं' लिखें। "
                "योजनाओं के अपने नाम (जैसे PM Kisan, PMAY) अंग्रेज़ी में ही रखें — बाकी सब हिंदी में।"
            )
        else:
            system = (
                "आप Yojan AI हैं, भारतीय सरकारी योजनाओं के विशेषज्ञ। "
                "CONTEXT (verified scheme database) का उपयोग करके उत्तर दें। योजनाओं को [1], [2] आदि से cite करें। "
                "योजनाओं के नाम, benefit amounts, या eligibility criteria जो CONTEXT में नहीं हैं, वो मत बनाइए। "
                "अगर context में कोई exact match नहीं है तो यह बताएं और सबसे relevant योजनाएं describe करें। "
                "Markdown, bullet points, या headers बिल्कुल मत use करें — plain flowing text only। "
                "हमेशा हिंदी में उत्तर दें। "
                "योजनाओं को 'scheme' या 'स्कीम' मत कहें — हमेशा 'योजना' या 'योजनाएं' लिखें। "
                "योजनाओं के अपने नाम (जैसे PM Kisan, PMAY) अंग्रेज़ी में ही रखें — बाकी सब हिंदी में।"
            )
    elif has_web:
        system = (
            "You are Yojan AI, an expert on Indian government schemes. "
            "You have two sources: CONTEXT (verified scheme database) and WEB RESULTS (live search). "
            "Use the CONTEXT as the primary authoritative source — cite schemes as [1], [2] etc. "
            "Use WEB RESULTS to enrich, add current details, or fill gaps — cite as [W1], [W2] etc. "
            "Synthesise both naturally into a single flowing answer. "
            "Do not invent scheme names, benefit amounts, or eligibility criteria not present in either source. "
            "Do NOT use markdown, bullet points, or headers — plain flowing text only. "
            "Always respond in English."
        )
    else:
        system = (
            "You are Yojan AI, an expert on Indian government schemes. "
            "Use the CONTEXT (verified scheme database) to answer. Cite schemes as [1], [2] etc. "
            "Do not invent scheme names, benefit amounts, or eligibility criteria not present in the CONTEXT. "
            "If no scheme in the context directly matches, say so and describe the closest relevant ones. "
            "Do NOT use markdown, bullet points, or headers — plain flowing text only. "
            "Always respond in English."
        )
    messages = [{"role": "system", "content": system}]
    if summary:
        messages.append({"role": "user", "content": f"Conversation summary so far: {summary}"})
    messages.extend(recent_msgs)

    messages.append({"role": "user", "content": prompt})

    return messages



class RAGContextProcessor(FrameProcessor):
    """Pipecat FrameProcessor that injects RAG context into the LLM context
    before each inference call. Intercepts LLMContextFrame, augments the last user
    message with retrieved scheme chunks, then passes it on.

    skip_web=True skips Firecrawl web search (for contexts where latency is critical)."""

    def __init__(self, retriever, skip_web: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._retriever = retriever
        self._skip_web = skip_web

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            ctx = frame.context
            messages = ctx.get_messages()

            user_idx = next(
                (i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "user"),
                None,
            )
            if user_idx is not None:
                query = messages[user_idx]["content"]
                loop = asyncio.get_running_loop()
                top_k = config.get("top_k", 5)

                n_web = config.get("agentic_web_results", 4)
                if self._skip_web:
                    chunks_raw = await loop.run_in_executor(
                        None, lambda: self._retriever.retrieve(query)
                    )
                    unique_chunks = _dedup_by_title(chunks_raw)
                    web_results = []
                else:
                    # Run retrieval and web search in parallel — same as text /api/chat
                    chunks_raw, web_candidates = await asyncio.gather(
                        loop.run_in_executor(None, lambda: self._retriever.retrieve(query)),
                        loop.run_in_executor(None, lambda: self._retriever.web_search(query, n_web)),
                    )
                    unique_chunks = _dedup_by_title(chunks_raw)
                    web_results = web_candidates

                if unique_chunks or web_results:
                    augmented = _build_prompt(query, unique_chunks, web_results)
                    messages[user_idx] = {"role": "user", "content": augmented}
                    ctx.set_messages(messages)

        await self.push_frame(frame, direction)


class FeatherlessLLMService(FrameProcessor):
    """Replaces GroqLLMService in the Pipecat voice pipeline.

    Receives an LLMContextFrame already enriched by RAGContextProcessor, streams
    tokens from featherless-ai using the same routing logic as /api/chat, and
    emits TextFrames so SarvamTTS can start speaking sentence-by-sentence before
    the full response is complete."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        from pipecat.processors.aggregators.llm_response_universal import LLMContextFrame as _LCF
        if not (isinstance(frame, _LCF) and direction == FrameDirection.DOWNSTREAM):
            await self.push_frame(frame, direction)
            return

        messages  = frame.context.get_messages()
        retriever = _state.get("retriever")
        is_complex   = getattr(retriever, "last_is_complex", False)
        active_client = _state["gen_client"] if is_complex else _state["draft_client"]
        max_tokens    = config.get("voice_max_tokens", 200)

        await self.push_frame(LLMFullResponseStartFrame(), direction)

        loop        = asyncio.get_running_loop()
        token_queue: asyncio.Queue = asyncio.Queue()

        def _stream():
            thinking_done = False
            buffer = ""
            try:
                for chunk in active_client.chat.completions.create(
                    messages=messages, stream=True, max_tokens=max_tokens
                ):
                    token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                    if not token:
                        continue
                    if not thinking_done:
                        buffer += token
                        if "</think>" in buffer:
                            thinking_done = True
                            after = buffer.split("</think>", 1)[-1].lstrip("\n")
                            if after:
                                loop.call_soon_threadsafe(token_queue.put_nowait, after)
                    else:
                        loop.call_soon_threadsafe(token_queue.put_nowait, token)
                if not thinking_done and buffer:
                    loop.call_soon_threadsafe(token_queue.put_nowait, buffer)
            except Exception as e:
                print(f"[voice-llm] stream error: {e}")
            finally:
                loop.call_soon_threadsafe(token_queue.put_nowait, None)

        threading.Thread(target=_stream, daemon=True).start()

        while True:
            token = await token_queue.get()
            if token is None:
                break
            await self.push_frame(TextFrame(text=token), direction)

        await self.push_frame(LLMFullResponseEndFrame(), direction)


def _chunk_to_scheme(c: dict) -> SchemeOut:
    name = c.get("title", "")
    # No direct URL in metadata — use Google search as the discovery link
    from urllib.parse import quote_plus
    link = c.get("link") or c.get("url") or c.get("source_url") or (
        f"https://www.google.com/search?q={quote_plus(name + ' India government scheme official')}"
        if name else ""
    )
    return SchemeOut(
        name=name,
        categories=c.get("category", "other"),
        level=c.get("location", "central"),
        benefits=c.get("benefit_summary", ""),
        eligibility_snippet=c.get("eligibility_summary", ""),
        source_url=link,
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


def _romanize_hinglish(text: str, client, model: str) -> str:
    """Convert Devanagari Hindi to Romanized Hinglish via one LLM call."""
    if not re.search(r'[ऀ-ॿ]', text):
        return text
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": (
                    "convert this to romanised hinglish (latin script). "
                    "example: write kaise ho, not कैसे हो.\n\n" + text
                )},
            ],
            stream=False,
            max_tokens=min(len(text) * 3, 2000),
        )
        result = resp.choices[0].message.content.strip()
        if "</think>" in result:
            result = result.split("</think>", 1)[-1].strip()
        # If the model still returned Devanagari, return original so the caller can
        # handle it (strip entirely) rather than sending garbled punctuation to TTS.
        if re.search(r'[ऀ-ॿ]', result):
            return text
        return result or text
    except Exception:
        return text


def _generate_answer(messages: list, active_client, max_tokens: int, model: str | None = None) -> str:
    """Run LLM and accumulate full answer (strips CoT <think> blocks)."""
    buffer = ""
    thinking_done = False
    full_answer = ""
    kwargs: dict = {"messages": messages, "stream": True, "max_tokens": max_tokens}
    if model:
        kwargs["model"] = model
    for chunk in active_client.chat.completions.create(**kwargs):
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
        else:
            full_answer += token
    if not thinking_done and buffer:
        full_answer = buffer
    return full_answer.strip()


def _judge(chunks: list[dict], query: str) -> str:
    """Legacy — kept so existing tests pass. Not called from chat endpoint."""
    if not chunks:
        return "insufficient"
    compact = "\n".join(
        f"[{i}] {c.get('title','')} — {c.get('benefit_summary','')}"
        for i, c in enumerate(chunks, 1)
    )
    messages = [
        {"role": "system", "content": "Reply with exactly one word: sufficient or insufficient."},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{compact}"},
    ]
    response = _state["gen_client"].chat.completions.create(messages=messages)
    verdict = response.choices[0].message.content.strip().lower()
    if "</think>" in verdict:
        verdict = verdict.split("</think>", 1)[-1].strip()
    return "sufficient" if verdict == "sufficient" else "insufficient"




# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


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
    loop = asyncio.get_running_loop()
    use_agentic = config.get("use_agentic_rag", False)

    t_retrieval_start = time.perf_counter()
    if use_agentic:
        n_web = config.get("agentic_web_results", 4)
        chunks_raw = await loop.run_in_executor(
            None, lambda: retriever.retrieve(req.message, recent_messages=recent_msgs, n_web=n_web)
        )
        unique_chunks = _dedup_by_title(chunks_raw)
        web_results = getattr(retriever, "last_web_results", [])
    else:
        chunks_raw = await loop.run_in_executor(
            None, lambda: retriever.retrieve(req.message, recent_messages=recent_msgs)
        )
        unique_chunks = _dedup_by_title(chunks_raw)
        web_results = []
    t_retrieval = time.perf_counter() - t_retrieval_start
    print(f"[latency] retrieval={t_retrieval:.2f}s  chunks={len(unique_chunks)}  web={len(web_results)}")

    if not unique_chunks and not web_results:
        async def _empty():
            yield _sse({"type": "schemes", "schemes": []})
            yield _sse({"type": "token", "content": "I couldn't find any matching schemes for your query."})
            yield "data: [DONE]\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    schemes = [_chunk_to_scheme(c) for c in unique_chunks]
    prompt   = _build_prompt(req.message, unique_chunks, web_results)
    messages = _build_messages(prompt, recent_msgs, summary, has_web=bool(web_results), lang=req.lang)

    # Standard routing: Groq 8B for simple, featherless 72B for complex.
    is_complex    = getattr(retriever, "last_is_complex", True)
    active_client = gen_client if is_complex else _state["draft_client"]
    active_model  = _state["gen_model"] if is_complex else _state["draft_model"]
    model_label   = "72B (complex)" if is_complex else "8B (simple/Groq)"
    max_tokens    = config.get("max_new_tokens", 350)

    # Approximate prompt token count (4 chars ≈ 1 token)
    prompt_chars = sum(len(m.get("content", "")) for m in messages)
    est_prompt_tokens = prompt_chars // 4

    def _stream_gen():
        """Synchronous SSE generator — runs in FastAPI's threadpool."""
        yield _sse({"type": "schemes", "schemes": [s.model_dump() for s in schemes]})

        full_answer   = ""
        completion_tokens = 0
        t_gen_start   = time.perf_counter()
        t_first_token = None

        # Think-block filter: buffer until </think> is seen, then stream freely.
        # If 20 chars pass with no <think>, assume non-thinking model and stream immediately.
        think_flushed = False
        in_think      = False
        pre_buf       = ""

        # Server-side TTS sentence detection (replaces client-side logic)
        tts_buf       = ""   # within-sentence token accumulator
        tts_accum     = ""   # complete-sentence accumulator
        tts_first_sent = False
        _TTS_PAT = re.compile(r'^([\s\S]*?[.!?])\s+([\s\S]*)$')

        hindi_voice = req.voice and req.lang == "hi"

        def _tts_events(text: str) -> list:
            """Detect sentence boundaries; return list of SSE strings to yield."""
            nonlocal tts_buf, tts_accum, tts_first_sent
            if not req.voice or not text:
                return []
            results = []
            tts_buf += text
            sm = _TTS_PAT.match(tts_buf)
            while sm:
                tts_accum += (" " if tts_accum else "") + sm.group(1)
                tts_buf = sm.group(2)
                sm = _TTS_PAT.match(tts_buf)
            threshold = 200 if tts_first_sent else 80
            sentence_ready = None
            if len(tts_accum) >= threshold:
                sentence_ready = tts_accum.strip()
                tts_accum = ""
                tts_first_sent = True
            elif len(tts_buf) > 300:
                sentence_ready = tts_buf.strip()
                tts_buf = ""
                tts_first_sent = True
            if sentence_ready:
                if hindi_voice:
                    # Translate to Hinglish (Roman script) server-side so Rumik
                    # receives clean Latin text — sounds significantly better than raw Devanagari.
                    tts_text = _to_hinglish(sentence_ready)
                    if tts_text:
                        results.append(_sse({"type": "tts", "content": tts_text}))
                else:
                    results.append(_sse({"type": "tts", "content": sentence_ready}))
            return results

        def _emit(text: str) -> list:
            """Return token SSE + any TTS SSEs for the given text."""
            nonlocal full_answer
            if not text:
                return []
            full_answer += text
            return [_sse({"type": "token", "content": text})] + _tts_events(text)

        for chunk in active_client.chat.completions.create(
            model=active_model, messages=messages, stream=True, max_tokens=max_tokens
        ):
            token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if not token:
                continue

            completion_tokens += 1
            if t_first_token is None:
                t_first_token = time.perf_counter() - t_gen_start

            if think_flushed:
                yield from _emit(token)
                continue

            pre_buf += token

            if in_think:
                if "</think>" in pre_buf:
                    _, after = pre_buf.split("</think>", 1)
                    pre_buf = ""
                    in_think = False
                    think_flushed = True
                    yield from _emit(after.lstrip("\n"))
            else:
                if "<think>" in pre_buf:
                    before, rest = pre_buf.split("<think>", 1)
                    pre_buf = rest
                    in_think = True
                    yield from _emit(before)
                elif len(pre_buf) >= 20:
                    # No <think> — not a thinking model; stream immediately
                    think_flushed = True
                    yield from _emit(pre_buf)
                    pre_buf = ""

        # Flush pre_buf if loop ended before a decision was made
        if pre_buf:
            yield from _emit(pre_buf)

        # Flush remaining TTS sentence buffer
        if req.voice:
            remaining = (tts_accum + (" " + tts_buf.strip() if tts_buf.strip() else "")).strip()
            if hindi_voice:
                if remaining:
                    tts_text = _to_hinglish(remaining)
                    if tts_text:
                        yield _sse({"type": "tts", "content": tts_text})
            elif remaining:
                yield _sse({"type": "tts", "content": remaining})

        t_total_gen = time.perf_counter() - t_gen_start
        ttft = round(t_first_token, 2) if t_first_token is not None else None
        print(
            f"[latency] model={model_label}  "
            f"prompt_tokens~{est_prompt_tokens}  "
            f"completion_tokens={completion_tokens}  "
            f"TTFT={ttft}s  "
            f"gen_total={t_total_gen:.2f}s  "
            f"pipeline_total={t_retrieval + t_total_gen:.2f}s"
        )
        yield _sse({
            "type": "stats",
            "model": model_label,
            "prompt_tokens": est_prompt_tokens,
            "completion_tokens": completion_tokens,
            "ttft": ttft,
            "retrieval_s": round(t_retrieval, 2),
            "gen_s": round(t_total_gen, 2),
            "total_s": round(t_retrieval + t_total_gen, 2),
        })

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


# ── Voice endpoints ────────────────────────────────────────────────────────────

@app.post("/api/voice")
async def voice_chat(request: Request, lang: str | None = Query(None)):
    """Push-to-talk: transcribe audio → RAG → return transcript + answer."""
    audio_bytes = await request.body()
    if not audio_bytes:
        return JSONResponse({"error": "no_audio"}, status_code=400)

    sarvam_key = os.environ.get("SARVAM_API_KEY")
    if not sarvam_key:
        return JSONResponse({"error": "sarvam_not_configured"}, status_code=503)

    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()

    # Transcribe with Sarvam saaras:v3 — sync SDK call wrapped in executor
    try:
        from sarvamai import SarvamAI
        sarvam_client = SarvamAI(api_subscription_key=sarvam_key)
        lang_code = "hi-IN" if lang == "hi" else "en-IN"

        def _transcribe():
            resp = sarvam_client.speech_to_text.transcribe(
                file=("audio.webm", audio_bytes, "audio/webm"),
                model="saaras:v3",
                language_code=lang_code,
            )
            return (resp.transcript or "").strip()

        transcript = await loop.run_in_executor(None, _transcribe)
    except Exception as exc:
        return JSONResponse({"error": f"transcription_failed: {exc}"}, status_code=500)
    t_stt = time.perf_counter() - t0
    print(f"[voice] STT={t_stt:.2f}s  transcript='{transcript[:60]}'")

    if not transcript:
        return JSONResponse({"error": "no_speech"}, status_code=400)

    # RAG pipeline (same logic as /api/chat, non-streaming)
    memory: Memory | None = _state.get("memory")
    retriever: Retriever  = _state["retriever"]

    recent_msgs, summary = [], ""
    if memory:
        recent_msgs = memory.get_recent([])
        summary = memory.summarize([]) if memory.should_compress() else ""

    t_ret_start = time.perf_counter()
    n_web = config.get("agentic_web_results", 4)
    chunks_raw = await loop.run_in_executor(
        None, lambda: retriever.retrieve(transcript, recent_messages=recent_msgs, n_web=n_web)
    )
    unique_chunks = _dedup_by_title(chunks_raw)
    web_results   = getattr(retriever, "last_web_results", [])
    t_ret = time.perf_counter() - t_ret_start
    print(f"[voice] retrieval={t_ret:.2f}s  chunks={len(unique_chunks)}  web={len(web_results)}")

    if not unique_chunks and not web_results:
        return JSONResponse({"transcript": transcript,
                             "answer": "I couldn't find any matching schemes for your query."})

    prompt = _build_prompt(transcript, unique_chunks, web_results)
    msgs   = _build_messages(prompt, recent_msgs, summary, has_web=bool(web_results), lang=lang)

    is_complex    = getattr(retriever, "last_is_complex", False)
    active_client = _state["gen_client"] if is_complex else _state["draft_client"]
    active_model  = _state["gen_model"] if is_complex else _state["draft_model"]
    model_label   = "72B (complex)" if is_complex else "8B (simple/Groq)"
    max_tokens    = config.get("voice_max_tokens", config.get("max_new_tokens", 350))
    est_prompt_tokens = sum(len(m.get("content", "")) for m in msgs) // 4

    t_gen_start = time.perf_counter()
    answer = await loop.run_in_executor(
        None, lambda: _generate_answer(msgs, active_client, max_tokens, model=active_model)
    )
    t_gen = time.perf_counter() - t_gen_start

    completion_tokens = len(answer.split())
    print(
        f"[voice] model={model_label}  "
        f"prompt_tokens~{est_prompt_tokens}  "
        f"completion_tokens~{completion_tokens}  "
        f"gen={t_gen:.2f}s  "
        f"total={time.perf_counter() - t0:.2f}s"
    )

    if memory:
        memory.track_message({"content": transcript})
        memory.track_message({"content": answer})
        memory.save([
            {"role": "user",      "content": transcript},
            {"role": "assistant", "content": answer},
        ])

    return JSONResponse({"transcript": transcript, "answer": answer})


@app.post("/api/voice/stt")
async def voice_stt(request: Request, lang: str | None = Query(None)):
    """STT only — transcribe audio via Sarvam saaras:v3 and return transcript."""
    audio_bytes = await request.body()
    if not audio_bytes:
        return JSONResponse({"error": "no_audio"}, status_code=400)

    sarvam_key = os.environ.get("SARVAM_API_KEY")
    if not sarvam_key:
        return JSONResponse({"error": "sarvam_not_configured"}, status_code=503)

    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    try:
        from sarvamai import SarvamAI
        sarvam_client = SarvamAI(api_subscription_key=sarvam_key)
        lang_code = "hi-IN" if lang == "hi" else "en-IN"

        def _transcribe():
            resp = sarvam_client.speech_to_text.transcribe(
                file=("audio.webm", audio_bytes, "audio/webm"),
                model="saaras:v3",
                language_code=lang_code,
            )
            return (resp.transcript or "").strip()

        transcript = await loop.run_in_executor(None, _transcribe)
    except Exception as exc:
        return JSONResponse({"error": f"transcription_failed: {exc}"}, status_code=500)

    print(f"[stt] {time.perf_counter()-t0:.2f}s  transcript='{transcript[:60]}'")
    if not transcript:
        return JSONResponse({"error": "no_speech"}, status_code=400)
    return JSONResponse({"transcript": transcript})


_HINGLISH_PROMPT = (
    "Convert the following text to Hinglish — Hindi written in Roman/Latin script. "
    "The input may be in English OR Hindi (Devanagari); handle both.\n"
    "Rules:\n"
    "1. Keep ALL scheme names, abbreviations and proper nouns exactly as-is "
    "(e.g. 'PM Kisan', 'Pradhan Mantri', 'PMAY', 'Ayushman Bharat').\n"
    "2. Write everything else in colloquial spoken Hindi using only Roman letters.\n"
    "3. Zero Devanagari characters in output — Roman script only.\n"
    "4. Output only the converted text, nothing else.\n"
    "5. Domain glossary: scheme→yojana, government→sarkar, eligibility→patrata, "
    "benefit→laabh, application→aavedan, ministry→mantralaya, farmer→kisan.\n\n"
)

def _to_hinglish(text: str) -> str:
    """Translate to Romanized Hinglish for TTS.
    Chain: Gemini 2.5 Flash → Groq fallback → raw text (Rumik handles Devanagari natively)."""
    if not re.search(r'[ऀ-ॿ]', text):
        return text  # already Latin script, nothing to translate

    print(f"[hinglish] in='{text[:120]}'")

    # 1. Try Gemini (best quality, but free tier = 20 req/day)
    gemini = _state.get("gemini_client")
    if gemini:
        try:
            from google.genai import types as _gtypes
            resp = gemini.models.generate_content(
                model="gemini-2.5-flash",
                contents=_HINGLISH_PROMPT + text,
                config=_gtypes.GenerateContentConfig(
                    thinking_config=_gtypes.ThinkingConfig(thinking_budget=0),
                ),
            )
            result = (resp.text or "").strip()
            if result and not re.search(r'[ऀ-ॿ]', result):
                print(f"[hinglish] gemini out='{result[:120]}'")
                return result
        except Exception as e:
            print(f"[hinglish] gemini error ({type(e).__name__}), trying Groq fallback")

    # 2. Groq fallback (fast, high rate-limits)
    draft_client = _state.get("draft_client")
    draft_model  = _state.get("draft_model", "")
    if draft_client:
        result = _romanize_hinglish(text, draft_client, draft_model)
        if result and not re.search(r'[ऀ-ॿ]', result):
            print(f"[hinglish] groq out='{result[:120]}'")
            return result

    # 3. Last resort: send raw text — Rumik Mulberry handles Devanagari natively
    print("[hinglish] all translators failed, sending raw text to Rumik")
    return text


@app.post("/api/voice/tts")
async def voice_tts(request: Request):
    """Convert text to speech via Rumik Mulberry — returns WAV audio."""
    import requests as _req
    body = await request.json()
    text = (body.get("text") or "").strip()
    lang       = body.get("lang", "en")
    translated = body.get("translated", False)  # True = already Hinglish, skip re-translation
    if not text:
        return JSONResponse({"error": "empty_text"}, status_code=400)

    rumik_key = os.environ.get("RUMIK_API_KEY")
    if not rumik_key:
        return JSONResponse({"error": "rumik_not_configured"}, status_code=503)

    if lang == "hi" and not translated:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, _to_hinglish, text)

    clean = _normalize_for_tts(text)
    # Reject if nothing readable remains (bare punctuation/whitespace)
    if not clean or not re.search(r'[a-zA-Z0-9ऀ-ॿ]', clean):
        return JSONResponse({"error": "empty_after_normalisation"}, status_code=400)

    # Rumik 2000-char limit; frontend sends ≤250-char chunks so this is a safety cap.
    payload_text = clean[:450]
    loop = asyncio.get_running_loop()

    def _call_rumik():
        print(f"[tts] rumik request: model=mulberry  chars={len(payload_text)}  text='{payload_text[:60]}'")
        resp = _req.post(
            "https://silk-api.rumik.ai/v1/tts",
            headers={
                "Authorization": f"Bearer {rumik_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "mulberry",
                "text": payload_text,
                # description only — per Rumik docs, speaker takes precedence over description
                # so never send both; description gives us the Indian accent we need.
                "description": "a female 30s hindi accent voice, brisk pacing, energetic emotion, neutral register, like a helpful government advisor",
            },
            timeout=20,
        )
        print(f"[tts] status={resp.status_code}  content-type={resp.headers.get('content-type')}")
        if not resp.ok:
            print(f"[tts] rumik error body: {resp.text[:500]}")
        resp.raise_for_status()
        return resp.content  # direct WAV bytes, no base64

    try:
        audio_bytes = await loop.run_in_executor(None, _call_rumik)
    except Exception as exc:
        print(f"[tts] exception: {exc}")
        return JSONResponse({"error": str(exc)}, status_code=500)

    return StreamingResponse(
        iter([audio_bytes]),
        media_type="audio/wav",
        headers={"Cache-Control": "no-cache"},
    )


# ── WebSocket voice endpoint (Pipecat pipeline) ───────────────────────────────

@app.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket, lang: str | None = Query(None)):
    if not PIPECAT_AVAILABLE:
        await websocket.close(code=1011, reason="pipecat not installed")
        return
    """
    Pipecat pipeline:
      browser (PCM via @pipecat-ai/client-js) →
      FastAPIWebsocketTransport (ProtobufFrameSerializer) →
      SileroVAD → DeepgramSTT → RAGContextProcessor →
      GroqLLM → CartesiaTTS → transport → browser audio
    """
    await websocket.accept()
    retriever: Retriever = _state["retriever"]

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=ProtobufFrameSerializer(),
        ),
    )
    # 1.5s idle before finalising the user turn — prevents cutting off mid-sentence
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(),
        audio_idle_timeout=1.5,
    )

    stt_language = PipecatLanguage.HI_IN if lang == "hi" else PipecatLanguage.EN_IN
    stt = SarvamSTTService(
        api_key=os.environ["SARVAM_API_KEY"],
        settings=SarvamSTTService.Settings(
            model="saaras:v3",
            language=stt_language,
        ),
    )

    llm = FeatherlessLLMService()

    sarvam_lang = SarvamLanguage.HI_IN if lang == "hi" else SarvamLanguage.EN_IN
    tts = SarvamTTSService(
        api_key=os.environ["SARVAM_API_KEY"],
        settings=SarvamTTSService.Settings(
            model="bulbul:v3",
            voice="suhani",
            language=sarvam_lang,
            pace=1.2,
        ),
    )

    system_content = (
        "You are Yojan AI, an expert on Indian government schemes. "
        "The CONTEXT contains verified scheme data. Use it — every scheme listed is real and available. "
        "Do NOT say you 'couldn't find' or 'don't have information about' any scheme that appears in the CONTEXT. "
        "Do not invent scheme names, amounts, or eligibility details not present in the CONTEXT. "
        "Lead with the most relevant scheme, then mention 1-2 others. "
        "This is a voice conversation — speak naturally, no bullet points, no markdown, no citations. "
        "Aim for 4-5 sentences. "
        "Always respond in English only. Never use any other language."
    )

    context     = LLMContext(messages=[{"role": "system", "content": system_content}])
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                # 1.5s silence to end the turn — long enough for natural speech pauses
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=1.5)],
            )
        ),
    )
    rag = RAGContextProcessor(retriever=retriever, skip_web=False)

    pipeline = Pipeline([
        transport.input(),
        vad,
        stt,
        aggregators.user(),
        rag,
        llm,
        tts,
        transport.output(),
        aggregators.assistant(),
    ])

    task   = PipelineTask(pipeline, params=PipelineParams(), idle_timeout_secs=None)
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
