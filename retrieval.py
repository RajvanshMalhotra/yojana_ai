import os
import re
import json
import time as _time
import numpy as np

# Queries that contain personal pronouns, relational nouns, or vague intent words
# are ambiguous and benefit from rewriting. Specific queries ("PM Kisan eligibility",
# "scholarship for SC student") are already clear and should be passed through unchanged.
_NEEDS_REWRITE_RE = re.compile(
    r"\b(i|my|me|we|our|us|i'm|i've|i am|i have|i need|i want|"
    r"help|need|want|looking|situation|issue|problem|"
    r"daughter|son|father|mother|wife|husband|brother|sister|family|child|children|"
    r"he|she|they|his|her|their)\b",
    re.IGNORECASE,
)
import nltk
from concurrent.futures import ThreadPoolExecutor
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from huggingface_hub import InferenceClient
stopwords = set(stopwords.words("english"))


class retriever:
    def __init__(self, config, index, chunks):
        self.config          = config
        self.index           = index           
        self.chunks          = chunks      
        self.top_k           = config["top_k"]


        self.use_query_rewrite    = config.get("use_query_rewrite", False)
        self.use_multi_query      = config.get("use_multi_query", False)
        self.use_rerank           = config.get("use_rerank", False)
        self.use_step_back        = config.get("use_step_back", False)
        self.use_agentic_rag      = config.get("use_agentic_rag", False)
        self.agentic_web_results  = config.get("agentic_web_results", 4)
        self.max_expanded_queries = config.get("max_expanded_queries", 3)

        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        print(f"Device    : {device}")

        self.embedding_model = SentenceTransformer(config["embedding_model"], device=device)
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
        # Use Groq for query processing (parse/rewrite/expand/classify) — much faster than featherless-ai.
        # Falls back to featherless-ai if GROQ_API_KEY is not set.
        _groq_keys = [k.strip() for k in os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", "")).split(",") if k.strip()]
        if _groq_keys:
            import random
            from groq import Groq as _Groq
            self.client = _Groq(api_key=random.choice(_groq_keys))
            self._query_model = "llama-3.1-8b-instant"
            print("Query LLM  : Groq llama-3.1-8b-instant")
        else:
            query_model = config.get("query_llm_model", config["llm_model"])
            self.client = InferenceClient(
                provider="featherless-ai",
                model=query_model,
                api_key=os.environ["HF_TOKEN"],
            )
            self._query_model = query_model
            print(f"Query LLM  : featherless-ai {query_model}")

        tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

   
    def _tokenize(self, text: str) -> list[str]:
        """Lowercase → strip punctuation → remove stopwords → list of tokens."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return [w for w in text.split() if w not in stopwords]

    def _llm(self, system: str, user: str) -> str:
        """Single helper for every LLM call so the rest of the code stays clean."""
        response = self.client.chat.completions.create(
            model=self._query_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ]
        )
        content = response.choices[0].message.content.strip()
        # Some thinking models close with </think> but omit the opening tag;
        # splitting on the last </think> and taking what follows handles both cases.
        if "</think>" in content:
            content = content.split("</think>", 1)[-1].strip()
        return content



    def parse_query(self, query: str) -> dict:
        """
        Turns the user's free-text query into structured fields we can use
        as Pinecone metadata filters BEFORE vector search runs.

        "pension for poor elderly women in Delhi"
        → {"beneficiaries": ["elderly", "women"], "category": "pension",
           "location": "delhi"}
        """
        system = """You are a query parser for an Indian government scheme finder.

Extract structured fields from the user query. Return ONLY a raw JSON object:

  "beneficiaries" : list — from: women, farmers, students, elderly, disabled,
                    bpl, sc, st, obc, minority, entrepreneur, widow, child, youth.
                    Use [] if none apply.
  "category"      : one of: agriculture, health, education, scholarship, housing,
                    employment, skills, pension, senior, business, entrepreneurship,
                    social_welfare, women, child, banking, insurance, financial,
                    sports, culture, technology, science, it, transport,
                    infrastructure, utility, sanitation, travel, tourism,
                    disability, public_safety, law, household, other.
  "location"      : "delhi", a state name, "central", or "any".

Return ONLY raw JSON. No markdown."""

        raw = self._llm(system, f"Query: {query}")
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {}

    # Maps LLM-returned category names to the actual values stored in chunk metadata.
    # Keys are what the LLM returns; values are exact Pinecone category strings.
    _CATEGORY_MAP = {
        "agriculture":    "agriculturerural_&_environment",
        "health":         "health_&_wellness",
        "education":      "education_&_learning",
        "scholarship":    "education_&_learning",
        "housing":        "housing_&_shelter",
        "employment":     "skills_&_employment",
        "skills":         "skills_&_employment",
        "pension":        "senior_citizen",
        "senior":         "senior_citizen",
        "business":       "business_&_entrepreneurship",
        "entrepreneurship": "business_&_entrepreneurship",
        "social_welfare": "social_welfare_&_empowerment",
        "women":          "women_and_child",
        "child":          "women_and_child",
        "banking":        "bankingfinancial_services_and_insurance",
        "insurance":      "bankingfinancial_services_and_insurance",
        "financial":      "bankingfinancial_services_and_insurance",
        "sports":         "sports_&_culture",
        "culture":        "sports_&_culture",
        "technology":     "science_it_&_communications",
        "science":        "science_it_&_communications",
        "it":             "science_it_&_communications",
        "transport":      "transport_&_infrastructure",
        "infrastructure": "transport_&_infrastructure",
        "utility":        "utility_&_sanitation",
        "sanitation":     "utility_&_sanitation",
        "travel":         "travel_&_tourism",
        "tourism":        "travel_&_tourism",
        "disability":     "social_welfare_&_empowerment",
        "public_safety":  "public_safetylaw_&_justice",
        "law":            "public_safetylaw_&_justice",
        "household":      "household",
    }

    # Regex patterns for common Hindi/Devanagari terms → Pinecone category.
    # Used by _detect_hindi_filter() to produce a deterministic filter before the
    # LLM parse runs — avoids relying on an 8B model to return exact English strings
    # when the query is in Hindi.
    _HINDI_FILTER_PATTERNS = [
        (re.compile(r'महिला|महिलाओं|औरत|नारी|स्त्री|बेटी|लड़की'), "women_and_child"),
        (re.compile(r'किसान|कृषि|खेती|फसल|बागवानी'), "agriculturerural_&_environment"),
        (re.compile(r'छात्र|छात्रा|विद्यार्थी|शिक्षा|पढ़ाई|स्कॉलरशिप|छात्रवृत्ति'), "education_&_learning"),
        (re.compile(r'उद्यमी|व्यापार|व्यवसाय|स्टार्टअप|कारोबार'), "business_&_entrepreneurship"),
        (re.compile(r'बुजुर्ग|वृद्ध|वरिष्ठ\s*नागरिक|पेंशन'), "senior_citizen"),
        (re.compile(r'विकलांग|दिव्यांग|अपंग'), "social_welfare_&_empowerment"),
        (re.compile(r'स्वास्थ्य|स्वास्थ|अस्पताल|चिकित्सा|इलाज'), "health_&_wellness"),
        (re.compile(r'आवास|घर|मकान|आवासीय'), "housing_&_shelter"),
        (re.compile(r'रोजगार|नौकरी|कौशल|प्रशिक्षण'), "skills_&_employment"),
        (re.compile(r'बैंक|बीमा|ऋण|लोन|वित्त'), "bankingfinancial_services_and_insurance"),
    ]

    def _detect_hindi_filter(self, query: str) -> dict | None:
        """Deterministic keyword filter for Hindi queries — runs before the LLM parse.
        Returns a Pinecone filter dict if a known Devanagari pattern matches, else None."""
        for pattern, category in self._HINDI_FILTER_PATTERNS:
            if pattern.search(query):
                return {"category": category}
        return None

    def _build_pinecone_filter(self, parsed: dict) -> dict | None:
        """
        Converts _parse_rewrite_expand() output into a Pinecone filter.
        Used for English queries — Hindi queries use _detect_hindi_filter() first.
        """
        raw_cat = parsed.get("category", "other")
        mapped  = self._CATEGORY_MAP.get(raw_cat)
        if mapped:
            return {"category": mapped}
        return None

  

    @staticmethod
    def _needs_rewrite(query: str) -> bool:
        """Returns True only for ambiguous/personal queries that benefit from rewriting.
        Specific queries like 'PM Kisan eligibility' are passed through unchanged."""
        return bool(_NEEDS_REWRITE_RE.search(query))

    def rewrite_query(self, query: str, recent_messages: list = []) -> str:
        """Rewrites only if the query is ambiguous — skips the LLM call for clear queries."""
        if not self._needs_rewrite(query):
            return query

        system = """You are a query rewriter for an Indian government scheme search.

Rewrite the user's query into a clean, standalone English search query.
  1. Resolve references (it, they, that) using chat history.
  2. Fix grammar, spelling, Hinglish.
  3. Remove filler ("can you", "tell me").
  4. Keep all important entities and constraints.

Output ONLY the rewritten query."""

        return self._llm(system, f"Chat history: {recent_messages}\nQuery: {query}")

    

    def expand_query(self, query: str) -> list[str]:
        """
        Generates N alternative phrasings of the query.
        Different vocabulary → different chunks get surfaced → better recall.
        """
        system = f"""You are a query expansion assistant for an Indian government scheme search.

Generate exactly {self.max_expanded_queries} diverse rephrasing of the user's query.
  - Preserve intent exactly.
  - Use synonyms, related terms, alternative phrasing.
  - One query per line. No numbering, no bullets, no explanation."""

        raw = self._llm(system, f"Query: {query}")
        return [q.strip() for q in raw.split("\n") if q.strip()]

   

    def semantic_search(self, query: str, filters: dict = None, top_k: int = 20) -> list[dict]:
        """Single-query path — encodes then queries Pinecone. Not used inside
        multi_query_retrieval (which batch-encodes to avoid thread-safety issues)."""
        embedding = self.embedding_model.encode(
            [query], prompt_name="query", normalize_embeddings=True,
        )[0].tolist()
        return self._pinecone_query(embedding, filters, top_k)

    def _pinecone_query(self, embedding: list, filters: dict | None, top_k: int) -> list[dict]:
        """Thread-safe: pure Pinecone network call, no tokenization.
        Retries once on transient gRPC h2 errors (stale connection after reload)."""
        import time as _time
        for attempt in range(2):
            try:
                results = self.index.query(
                    vector=embedding, filter=filters, top_k=top_k, include_metadata=True,
                )
                break
            except Exception as e:
                if attempt == 0:
                    print(f"[pinecone] transient error, retrying: {e}")
                    _time.sleep(0.5)
                    continue
                raise
        chunks = []
        for match in results["matches"]:
            chunk = match["metadata"].copy()
            chunk["score"] = match["score"]
            chunks.append(chunk)
        return chunks


    def _matches_filter(self, chunk: dict, filters: dict) -> bool:
        """Returns True if the chunk satisfies every condition in the metadata filter."""
        for key, condition in filters.items():
            val = chunk.get(key)
            # Skip filter keys whose field is absent in the chunk data
            if val is None or val == "":
                continue
            if isinstance(condition, dict):
                if "$in" in condition:
                    haystack = val if isinstance(val, list) else [val]
                    if not any(v in condition["$in"] for v in haystack):
                        return False
            else:
                if val != condition:
                    return False
        return True

    def keyword_search(self, query: str, top_k: int = 20, filters: dict = None) -> list[dict]:
        """
        BM25 keyword search — great for exact scheme names like "PM-KISAN"
        or "Ayushman Bharat" that dense embeddings can sometimes miss.

        filters — same dict as passed to semantic_search; applied in-memory so
                  BM25 results honour the same metadata constraints as vector search.
        """
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1]   # descending
        results = []
        for i in top_indices:
            if len(results) >= top_k:
                break
            chunk = self.chunks[i]
            if filters and not self._matches_filter(chunk, filters):
                continue
            results.append(chunk)
        return results

    def web_search(self, query: str, n: int = 4) -> list[dict]:
        """Firecrawl search — returns [{title, snippet, url}, …] or [] on failure."""
        try:
            from firecrawl import FirecrawlApp
            app = FirecrawlApp(api_key=os.environ["FIRECRAWL_API_KEY"])
            search_query = f"{query} India"
            data = app.search(search_query, limit=n)
            hits = data.web or []
            print(f"[web] query='{search_query[:60]}'  hits={len(hits)}")
            return [
                {"title": r.title or "", "snippet": r.description or "", "url": r.url or ""}
                for r in hits
            ]
        except Exception as e:
            print(f"[web] search failed: {e}")
            return []

    def _reciprocal_rank_fusion(
        self, dense: list[dict], sparse: list[dict], k: int = 60
    ) -> list[dict]:
        """
        Merges two ranked lists into one.
        Chunks that appear high in BOTH lists get the best combined score.

        score = 1/(k + dense_rank) + 1/(k + sparse_rank)
        k=60 is a standard constant that prevents the #1 result from dominating.
        """
        scores    = {}
        chunk_map = {}

        for rank, chunk in enumerate(dense):
            cid = chunk.get("chunk_id", chunk.get("title", rank))
            scores[cid]    = scores.get(cid, 0) + 1 / (k + rank + 1)
            chunk_map[cid] = chunk

        for rank, chunk in enumerate(sparse):
            cid = chunk.get("chunk_id", chunk.get("title", rank))
            scores[cid]    = scores.get(cid, 0) + 1 / (k + rank + 1)
            chunk_map[cid] = chunk

        ranked_ids = sorted(scores, key=scores.get, reverse=True)
        return [chunk_map[cid] for cid in ranked_ids]

    def multi_query_retrieval(self, queries: list[str], filters: dict = None) -> list[dict]:
        """Runs hybrid search for every query variant in parallel and fuses all results.

        Embeddings are batch-encoded serially BEFORE entering the thread pool because
        the HuggingFace fast tokenizer's Rust backend is not thread-safe (raises
        'Already borrowed' when encode() is called from concurrent threads).
        Only the Pinecone network calls and BM25 scoring run in parallel."""
        # Single batch encode — safe, serialised
        embeddings = self.embedding_model.encode(
            queries, prompt_name="query", normalize_embeddings=True, batch_size=32,
        )

        def _dense(emb):  return self._pinecone_query(emb.tolist(), filters, top_k=15)
        def _sparse(q):   return self.keyword_search(q, top_k=15, filters=filters)

        n = len(queries)
        with ThreadPoolExecutor(max_workers=n * 2) as pool:
            dense_futs  = [pool.submit(_dense, emb) for emb in embeddings]
            sparse_futs = [pool.submit(_sparse, q)  for q in queries]
            all_dense   = [c for f in dense_futs  for c in f.result()]
            all_sparse  = [c for f in sparse_futs for c in f.result()]

        return self._reciprocal_rank_fusion(all_dense, all_sparse)



    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """
        CrossEncoder reads (query, chunk) pairs TOGETHER and scores each one.
        Unlike embedding similarity which scores them separately, the CrossEncoder
        catches subtle relevance signals — but it's slower, so we only run it
        on the ~30 candidates from fusion, not all 4785 chunks.
        """
        if not candidates:
            return []

        pairs  = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]

  

    def _parse_rewrite_expand(self, query: str, recent_messages: list = []) -> tuple[dict, str, str, list[str]]:
        """
        Single LLM call: parse + rewrite + step-back + expand.
        Returns (filters_dict, rewritten_query, step_back_query, expanded_queries).
        """
        system = f"""You are a query processor for an Indian government scheme search engine.

Return ONLY a raw JSON object — no markdown, no explanation:

{{
  "filters": {{
    "beneficiaries": ["from: women,farmers,students,elderly,disabled,bpl,sc,st,obc,minority,entrepreneur,widow,child,youth — empty list if none"],
    "category": "one of: agriculture,health,education,scholarship,housing,employment,skills,pension,senior,business,entrepreneurship,social_welfare,women,child,banking,insurance,financial,sports,culture,technology,science,it,transport,infrastructure,utility,sanitation,travel,tourism,disability,public_safety,law,household,other",
    "location": "delhi | <state name> | central | any"
  }},
  "rewritten_query": "clean standalone English query — resolve pronouns from history, fix grammar, strip filler",
  "step_back_query": "a broader, more abstract version of the query that captures the general topic (e.g. 'schemes for AI startups' → 'government support for technology innovation and startups')",
  "expanded_queries": ["exactly {self.max_expanded_queries} alternative phrasings using synonyms and related terms"]
}}"""

        raw = self._llm(system, f"Chat history: {recent_messages}\nQuery: {query}")
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(raw[start:end + 1])
                return (
                    data.get("filters", {}),
                    data.get("rewritten_query", query),
                    data.get("step_back_query", query),
                    data.get("expanded_queries", []),
                )
            except json.JSONDecodeError:
                pass
        return {}, query, query, []

    def retrieve(self, query: str, recent_messages: list = [], fast: bool = False, n_web: int = 0) -> list[dict]:
        """
        Full retrieval pipeline — one LLM call handles parse + rewrite + step-back + expand,
        then parallel hybrid search and rerank.

        fast=True skips all LLM pre-processing (query rewrite, step-back, expand,
        complexity classification). Used for voice mode where latency is critical.
        Cuts retrieval from ~12s down to ~1-2s.
        """
        if fast:
            # Voice fast path: rewrite only if the query is ambiguous (regex gate, no LLM for clear queries)
            rewritten = self.rewrite_query(query, recent_messages) if self.use_query_rewrite else query
            queries = [rewritten]
            self.last_step_back_query = rewritten
            self.last_is_complex = False  # always route to 7B in voice mode
            self.last_web_results = []
            print(f"[fast] rewritten='{rewritten[:80]}'")

            candidates = self.multi_query_retrieval(queries, None)
        elif self.use_query_rewrite or self.use_multi_query or self.use_step_back:
            _t0 = _time.perf_counter()
            filters_raw, rewritten, step_back, expanded = self._parse_rewrite_expand(query, recent_messages)
            print(f"[ret] parse_rewrite_expand={_time.perf_counter()-_t0:.2f}s")

            # Deterministic Hindi keyword filter takes priority over LLM-parsed result.
            # The LLM (8B Groq) returns inconsistent strings for Hindi beneficiary terms
            # (e.g. "woman", "महिलाओं", "females") so regex is more reliable.
            filters = self._detect_hindi_filter(query) or self._build_pinecone_filter(filters_raw)

            primary = rewritten if self.use_query_rewrite else query
            queries = [primary]

            if self.use_step_back:
                queries.insert(0, step_back)

            if self.use_multi_query:
                queries.extend(expanded)

            self.last_step_back_query = step_back
            print(f"Filters   : {filters}")
            print(f"Step-back : {queries[0] if self.use_step_back else '(off)'}")
            print(f"Primary   : {queries[0] if not self.use_step_back else queries[1] if len(queries) > 1 else '—'}")
            print(f"[ret] {len(queries)} queries to embed+search")

            _t1 = _time.perf_counter()
            with ThreadPoolExecutor(max_workers=3) as pool:
                search_fut   = pool.submit(self.multi_query_retrieval, queries, filters)
                classify_fut = pool.submit(self._classify_complexity, query)
                web_fut      = pool.submit(self.web_search, primary, n_web) if n_web > 0 else None
                candidates   = search_fut.result()
                is_complex   = classify_fut.result()
                self.last_web_results = web_fut.result() if web_fut else []
            print(f"[ret] embed+pinecone+classify+web={_time.perf_counter()-_t1:.2f}s")

            self.last_is_complex = is_complex
            print(f"Complex   : {is_complex}")
        else:
            filters_raw = self.parse_query(query)
            filters = self._build_pinecone_filter(filters_raw)
            queries = [query]
            self.last_step_back_query = query
            self.last_web_results = []
            print(f"Filters   : {filters}")

            with ThreadPoolExecutor(max_workers=2) as pool:
                search_fut   = pool.submit(self.multi_query_retrieval, queries, filters)
                classify_fut = pool.submit(self._classify_complexity, query)
                candidates   = search_fut.result()
                is_complex   = classify_fut.result()

            self.last_is_complex = is_complex
            print(f"Complex   : {is_complex}")

        # Deduplicate by chunk_id → text fingerprint → title
        seen = set()
        deduped = []
        for c in candidates:
            key = c.get("chunk_id") or c.get("text", "")[:120] or c.get("title", "")
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        candidates = deduped

        if self.use_rerank and not fast:
            candidates = self.rerank(queries[0], candidates, top_k=self.top_k)
        else:
            candidates = candidates[:self.top_k]

        return candidates

    def _classify_complexity(self, query: str) -> bool:
        """Fast LLM call — runs in parallel with search, decides 7B vs 72B routing.

        Complex = multiple simultaneous hard eligibility constraints (caste + income + location + profession).
        Simple  = broad topic, single beneficiary type, or named scheme lookup."""
        resp = self.client.chat.completions.create(
            model=self._query_model,
            messages=[{"role": "user", "content": (
                "Does this query specify MULTIPLE simultaneous hard eligibility constraints "
                "(e.g. caste AND income limit AND location AND profession together)?\n"
                "Simple examples: 'govt schemes for students', 'PM Kisan eligibility', "
                "'startup grants for AI companies', 'health scheme for farmers'.\n"
                "Complex examples: 'schemes for SC women farmers in UP earning below 1 lakh', "
                "'disabled OBC student scholarship in Maharashtra for engineering'.\n"
                "Reply with only one word: complex or simple\n\n"
                f"Query: {query}"
            )}]
        )
        verdict = resp.choices[0].message.content.strip().lower()
        if "</think>" in verdict:
            verdict = verdict.split("</think>", 1)[-1].strip()
        return "complex" in verdict


if __name__ == "__main__":
    import pickle
    import yaml
    from dotenv import load_dotenv
    from pinecone import Pinecone
    load_dotenv()


    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Load chunks
    with open("data/mlx_enriched/scheme_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    # Connect to Pinecone
    pc    = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(config["pinecone_index_name"])

    # Init retriever
    r = retriever(config, index, chunks)

    # Test query
    query   = "pension scheme for elderly women in Delhi"
    results = r.retrieve(query)

    print(f"\nTop {config['top_k']} results for: '{query}'\n")
    for i, chunk in enumerate(results, 1):
        print(f"[{i}] {chunk['title']}")
        print(f"     {chunk.get('benefit_summary', '')}")
        print()
