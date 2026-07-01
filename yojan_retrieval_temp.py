# =============================================================================
# YOJAN AI — RETRIEVAL PIPELINE (TEMP / LEARNING FILE)
# =============================================================================
#
# WHAT THIS FILE IS
# -----------------
# A side-by-side companion to retrieval.py.
# retrieval.py is the production class. This file explains WHY each piece
# exists, what you'd type to use it, and how the pieces connect.
#
# THE OVERALL FLOW
# ----------------
# User query
#     │
#     ▼
# 1. Query parsing     — extract WHAT the user is asking for (beneficiary,
#                        category, location) as structured fields
#     │
#     ▼
# 2. Query rewrite     — fix grammar, clarify vague terms
#     │
#     ▼
# 3. Multi-query       — expand into 3 variants to cover different phrasings
#     │
#     ▼
# 4. Hybrid retrieval  — for EACH variant:
#                          a) dense search  (embedding similarity in Pinecone)
#                          b) sparse search (BM25 keyword match on chunks)
#                        merge results (Reciprocal Rank Fusion)
#     │
#     ▼
# 5. Rerank            — CrossEncoder scores each retrieved chunk against the
#                        original query and re-orders by relevance
#     │
#     ▼
# 6. Answer            — top-k chunks sent to LLM with the user's question
#
# WHY NOT HyDE?
# -------------
# HyDE generates a hypothetical document that answers the query, then embeds
# THAT instead of the query.  It works well for encyclopaedia-style corpora
# where the query and document are in very different styles.
#
# For Yojan AI, scheme descriptions are already short and factual.
# A hypothetical document risks hallucinating specific ₹ amounts, age limits,
# or income thresholds — which is dangerous for a scheme-finder.
# Metadata filtering + hybrid search gives us precision without hallucination.
#
# =============================================================================


# =============================================================================
# STEP 1: QUERY PARSING
# =============================================================================
#
# WHAT IT DOES
# ------------
# Converts "schemes for poor women farmers in Delhi" into structured fields:
#
#   {
#     "beneficiaries": ["women", "farmers"],
#     "category":      "agriculture",
#     "location":      "delhi",
#     "income_group":  "bpl"
#   }
#
# WHY
# ---
# Our Pinecone chunks already have these fields stored as metadata
# (we put them there in yojan_chunking_temp.py).
# If we extract them from the query too, we can do:
#
#   pinecone.query(
#     vector=query_embedding,
#     filter={"beneficiaries": {"$in": ["women", "farmers"]}},
#     top_k=20
#   )
#
# This narrows the search BEFORE embedding comparison even runs.
# Result: much higher precision, no irrelevant schemes surfacing.
#
# HOW TO CALL IT
# --------------
# from yojan_retrieval_temp import parse_query
# result = parse_query("schemes for poor elderly women in Delhi", llm_client)
# print(result)
# → {"beneficiaries": ["elderly", "women"], "category": "social_welfare",
#    "location": "delhi", "income_group": "bpl"}

QUERY_PARSE_PROMPT = """You are a query parser for an Indian government scheme finder.

Extract structured fields from the user's query. Return a JSON object with these keys:

  "beneficiaries"  : list of strings — who the user belongs to.
                     Pick from: ["women", "farmers", "students", "elderly",
                     "disabled", "bpl", "sc", "st", "obc", "minority",
                     "entrepreneur", "widow", "child", "youth", "unemployed"]
                     Use [] if none apply.

  "category"       : string — the domain of help needed.
                     Pick from: agriculture, health, education, housing,
                     employment, pension, scholarship, business, social_welfare,
                     or "other" if unclear.

  "location"       : string — "delhi", a state name, "central" (nationwide),
                     or "any" if not mentioned.

  "income_group"   : string — "bpl" (below poverty line), "low_income"
                     (annual income < ₹1-2L), "any" if not mentioned.

Rules:
- Return ONLY a raw JSON object. No markdown. No explanation.
- If uncertain, use "any" or [].

User query: {query}"""


# =============================================================================
# STEP 2: QUERY REWRITE
# =============================================================================
#
# WHAT IT DOES
# ------------
# Fixes grammatical issues, informal Hindi-English, or vague phrasing so the
# embedding model gets a clean, standard English sentence.
#
# Example:
#   "mujhe koi scheme chahiye for old age" →
#   "What government schemes provide financial support for senior citizens?"
#
# WHY
# ---
# The embedding model was trained on clean English text.
# Hinglish or typo-laden queries produce a worse embedding.
# One LLM call to clean the query is cheap and improves retrieval quality.
#
# HOW TO CALL IT
# --------------
# rewritten = rewrite_query("koi scheme for beti education hai?", llm_client)
# print(rewritten)
# → "What government schemes exist for funding girl child education in India?"

REWRITE_PROMPT = """You are a query rewriter for an Indian government scheme finder.

Rewrite the user's query as a clean, formal English question suitable for
semantic search over government scheme descriptions.

Rules:
- Fix grammar, spelling, and Hinglish
- Keep the meaning exactly the same — do NOT add assumptions
- Return ONLY the rewritten query. No explanation, no quotes.

Original query: {query}"""


# =============================================================================
# STEP 3: MULTI-QUERY EXPANSION
# =============================================================================
#
# WHAT IT DOES
# ------------
# Generates N variants of the query, each phrasing the intent differently.
#
# Example for "pension for old farmers":
#   1. "What pension schemes are available for elderly farmers in India?"
#   2. "Monthly financial support for aged agricultural workers"
#   3. "Retirement benefits for farmers above 60 years"
#
# WHY
# ---
# A single query embedding can miss chunks that use different vocabulary.
# Variant 1 uses "pension", variant 2 uses "financial support", variant 3
# uses "retirement benefits" — together they cover more ground.
# We retrieve top-k for EACH variant, then merge with RRF (see Step 4b).
#
# HOW TO CALL IT
# --------------
# variants = expand_query("pension for old farmers", n=3, llm_client)
# for v in variants:
#     print(v)

MULTI_QUERY_PROMPT = """You are a query expansion assistant for an Indian government scheme search engine.

Generate {n} different phrasings of the user's query that preserve the same intent
but use different vocabulary. Each variant should be a complete, standalone search
query.

Rules:
- One variant per line
- No numbering, no bullet points, no explanation
- Do NOT change the intent — only rephrase

Original query: {query}"""


# =============================================================================
# STEP 4A: DENSE RETRIEVAL (Pinecone)
# =============================================================================
#
# WHAT IT DOES
# ------------
# Embeds the query and finds the most similar chunks by cosine similarity.
# This is the "semantic" part — it understands meaning, not just keywords.
#
# WHAT YOU TYPE
# -------------
# (inside the retriever class)
#
#   query_embedding = self.embedding_model.encode(
#       [query],
#       prompt_name="query",       # Qwen3 asymmetric: queries use a different
#       normalize_embeddings=True  # prompt than documents
#   )
#   results = pinecone_index.query(
#       vector=query_embedding[0].tolist(),
#       filter=parsed_fields,      # from Step 1
#       top_k=20,
#       include_metadata=True
#   )
#
# WHY prompt_name="query"?
# Qwen3-Embedding-0.6B is an ASYMMETRIC model — documents and queries are
# encoded differently. Documents use the default prompt; queries must pass
# prompt_name="query". Skipping this gives worse retrieval quality.


# =============================================================================
# STEP 4B: SPARSE RETRIEVAL (BM25) + FUSION
# =============================================================================
#
# WHAT BM25 DOES
# --------------
# BM25 is a keyword search algorithm.  It counts how often the query words
# appear in a chunk, weighted by how rare those words are across all chunks.
#
# WHY WE NEED IT ALONGSIDE DENSE
# --------------------------------
# Dense search understands meaning but can miss exact matches.
# If a user types "PM-KISAN" or "Ayushman Bharat", BM25 finds those chunks
# instantly by exact keyword match — dense embeddings sometimes drift.
#
# RRF (Reciprocal Rank Fusion)
# ----------------------------
# We combine dense and sparse results using RRF:
#   score(chunk) = 1/(k + dense_rank) + 1/(k + sparse_rank)
# where k=60 is a smoothing constant.
# This rewards chunks that rank high in BOTH lists.
#
# WHAT YOU TYPE
# -------------
#   # Build BM25 index once from all chunks
#   tokenized = [chunk["text"].lower().split() for chunk in chunks]
#   bm25 = BM25Okapi(tokenized)
#
#   # At query time
#   query_tokens = query.lower().split()
#   bm25_scores = bm25.get_scores(query_tokens)
#   top_bm25_ids = np.argsort(bm25_scores)[::-1][:top_k]

def reciprocal_rank_fusion(dense_ids: list, sparse_ids: list, k: int = 60) -> list:
    """
    Merges two ranked lists into one using RRF.
    Returns chunk IDs ordered by combined score, highest first.
    """
    scores = {}
    for rank, chunk_id in enumerate(dense_ids):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    for rank, chunk_id in enumerate(sparse_ids):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


# =============================================================================
# STEP 5: RERANKING
# =============================================================================
#
# WHAT IT DOES
# ------------
# A CrossEncoder takes (query, chunk) pairs and scores them together.
# Unlike the embedding model which scores query and chunk independently,
# the CrossEncoder reads them jointly — it can catch subtle relevance signals.
#
# WHY
# ---
# After fusion we have ~30-40 candidate chunks.  Many will be tangentially
# relevant.  The CrossEncoder picks the truly relevant ones before we send
# them to the LLM (we can't send 40 chunks — too many tokens).
#
# WHAT YOU TYPE
# -------------
#   pairs = [(query, chunk["text"]) for chunk in candidates]
#   scores = self.cross_encoder.predict(pairs)
#   ranked = sorted(zip(scores, candidates), reverse=True)
#   top_chunks = [chunk for _, chunk in ranked[:top_k]]
#
# GOOD MODEL CHOICES
# ------------------
# "cross-encoder/ms-marco-MiniLM-L-6-v2" — fast, English, great for short queries
# "BAAI/bge-reranker-base"               — multilingual, handles Hindi queries better


# =============================================================================
# PUTTING IT ALL TOGETHER — WHAT YOU CALL AT QUERY TIME
# =============================================================================
#
# from yojan_retrieval_temp import ...
#
# user_query = "mujhe widow pension chahiye in Delhi"
#
# 1. parsed   = parse_query(user_query, llm_client)
#    → {"beneficiaries": ["widow"], "category": "pension", "location": "delhi"}
#
# 2. clean    = rewrite_query(user_query, llm_client)
#    → "What widow pension schemes are available in Delhi?"
#
# 3. variants = expand_query(clean, n=3, llm_client)
#    → ["Pension for widows in Delhi", "Financial support for widowed women...", ...]
#
# 4. For each variant:
#      dense_results  = pinecone.query(vector=embed(variant), filter=parsed, top_k=15)
#      sparse_results = bm25.get_top_k(variant, top_k=15)
#      merged         = reciprocal_rank_fusion(dense_results, sparse_results)
#    Combine all variants → deduplicate → top 30 candidates
#
# 5. final_chunks = cross_encoder.rerank(user_query, candidates, top_k=5)
#
# 6. answer = llm.chat(prompt=build_prompt(user_query, final_chunks))
#
# =============================================================================
