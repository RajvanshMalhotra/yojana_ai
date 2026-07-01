import os
import re
import json
import numpy as np
import nltk
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        self.max_expanded_queries = config.get("max_expanded_queries", 3)

        self.embedding_model = SentenceTransformer(config["embedding_model"])

        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.client = InferenceClient(
            provider="featherless-ai",
            model=config["llm_model"],
            api_key=os.environ["HF_TOKEN"],
        )

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
  "category"      : one of: agriculture, health, education, housing, employment,
                    pension, scholarship, business, social_welfare, other.
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

    # Maps LLM-returned category names to the actual values stored in chunk metadata
    _CATEGORY_MAP = {
        "agriculture":    "agriculturerural_&_environment",
        "health":         "health_&_wellness",
        "education":      "education_&_learning",
        "housing":        "housing_&_shelter",
        "employment":     "skills_&_employment",
        "pension":        "senior_citizen",
        "scholarship":    "education_&_learning",
        "business":       "business_&_entrepreneurship",
        "social_welfare": "social_welfare_&_empowerment",
    }

    def _build_pinecone_filter(self, parsed: dict) -> dict | None:
        """
        Converts parse_query() output into a Pinecone filter.

        Only category is used — location doesn't exist in the index and
        beneficiaries are raw scheme tags that don't match semantic labels.
        """
        raw_cat = parsed.get("category", "other")
        mapped  = self._CATEGORY_MAP.get(raw_cat)
        if mapped:
            return {"category": mapped}
        return None

  

    def rewrite_query(self, query: str, recent_messages: list = []) -> str:
        """
        Fixes grammar / Hinglish and resolves references ("it", "that") using
        chat history, giving the embedding model a clean search query.
        """
        system = """You are a query rewriter for an Indian government scheme search.

Rewrite the user's query into a clean, standalone English search query.
  1. Resolve references (it, they, that) using chat history.
  2. Fix grammar, spelling, Hinglish.
  3. Remove filler ("can you", "tell me").
  4. Keep all important entities and constraints.
  5. If already clear, return unchanged.

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
        """
        Embeds the query and finds the most similar chunks in Pinecone.

        filters — Pinecone metadata filter from _build_pinecone_filter().
                  Narrows the search space before vector comparison runs.
        top_k   — pull more candidates here; reranking trims to final top_k later.
        """

        query_embedding = self.embedding_model.encode(
            [query],
            prompt_name="query",
            normalize_embeddings=True,
        )

        results = self.index.query(
            vector=query_embedding[0].tolist(),
            filter=filters,
            top_k=top_k,
            include_metadata=True,
        )

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
        """Runs hybrid search for every query variant in parallel and fuses all results."""
        all_dense  = []
        all_sparse = []

        def _dense(q):  return self.semantic_search(q, filters=filters, top_k=15)
        def _sparse(q): return self.keyword_search(q, top_k=15, filters=filters)

        tasks = [(fn, q) for q in queries for fn in (_dense, _sparse)]
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(fn, q): (fn.__name__, q) for fn, q in tasks}
            for fut in as_completed(futures):
                fn_name, _ = futures[fut]
                results = fut.result()
                if fn_name == "_dense":
                    all_dense.extend(results)
                else:
                    all_sparse.extend(results)

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
    "category": "one of: agriculture,health,education,housing,employment,pension,scholarship,business,social_welfare,other",
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

    def retrieve(self, query: str, recent_messages: list = []) -> list[dict]:
        """
        Full retrieval pipeline — one LLM call handles parse + rewrite + step-back + expand,
        then parallel hybrid search and rerank.
        """
        if self.use_query_rewrite or self.use_multi_query or self.use_step_back:
            filters_raw, rewritten, step_back, expanded = self._parse_rewrite_expand(query, recent_messages)
            filters = self._build_pinecone_filter(filters_raw)

            primary = rewritten if self.use_query_rewrite else query
            queries = [primary]

            if self.use_step_back:
                queries.insert(0, step_back)   # step-back first → highest RRF weight

            if self.use_multi_query:
                queries.extend(expanded)
        else:
            filters_raw = self.parse_query(query)
            filters = self._build_pinecone_filter(filters_raw)
            queries = [query]

        print(f"Filters   : {filters}")
        print(f"Step-back : {queries[0] if self.use_step_back else '(off)'}")
        print(f"Primary   : {queries[0] if not self.use_step_back else queries[1] if len(queries) > 1 else '—'}")

    
        candidates = self.multi_query_retrieval(queries, filters=filters)

        # Deduplicate by chunk_id → text fingerprint → title, in that priority order
        seen = set()
        deduped = []
        for c in candidates:
            key = c.get("chunk_id") or c.get("text", "")[:120] or c.get("title", "")
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        candidates = deduped

        if self.use_rerank:
            candidates = self.rerank(queries[0], candidates, top_k=self.top_k)
        else:
            candidates = candidates[:self.top_k]

        return candidates


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
