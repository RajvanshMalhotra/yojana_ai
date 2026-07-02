import os
import pytest
from unittest.mock import patch, MagicMock

def make_retriever():
    """Build a minimal retriever without loading real models."""
    import sys, types
    for mod in ["sentence_transformers", "rank_bm25", "huggingface_hub"]:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)
    sys.modules["sentence_transformers"].SentenceTransformer = MagicMock()
    sys.modules["sentence_transformers"].CrossEncoder = MagicMock()
    sys.modules["rank_bm25"].BM25Okapi = MagicMock()
    sys.modules["huggingface_hub"].InferenceClient = MagicMock()

    import importlib
    os.environ.setdefault("HF_TOKEN", "test")
    os.environ.setdefault("FIRECRAWL_API_KEY", "test-key")
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


def _make_fc_result(title, description, url):
    m = MagicMock()
    m.title = title
    m.description = description
    m.url = url
    return m


def test_web_search_returns_results():
    r = make_retriever()
    fake_web = [
        _make_fc_result("Startup India", "Grant for startups", "https://example.com"),
        _make_fc_result("DPIIT Fund",    "Seed funding scheme", "https://dpiit.gov.in"),
    ]
    mock_data = MagicMock()
    mock_data.web = fake_web
    with patch("firecrawl.FirecrawlApp") as MockApp:
        MockApp.return_value.search.return_value = mock_data
        results = r.web_search("AI startup grants", n=2)

    assert len(results) == 2
    assert results[0]["title"] == "Startup India"
    assert results[0]["snippet"] == "Grant for startups"
    assert results[0]["url"] == "https://example.com"


def test_web_search_returns_empty_on_exception():
    r = make_retriever()
    with patch("firecrawl.FirecrawlApp", side_effect=Exception("network error")):
        results = r.web_search("anything")
    assert results == []


def test_web_search_appends_india_government():
    r = make_retriever()
    mock_data = MagicMock()
    mock_data.web = []
    with patch("firecrawl.FirecrawlApp") as MockApp:
        MockApp.return_value.search.return_value = mock_data
        r.web_search("AI startup grants", n=3)
        call_args = MockApp.return_value.search.call_args
    assert "India government" in call_args[0][0]
