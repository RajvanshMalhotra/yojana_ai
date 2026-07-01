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
