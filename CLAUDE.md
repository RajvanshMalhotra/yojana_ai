# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Advanced RAG (Retrieval-Augmented Generation) pipeline in Python with a planned Streamlit UI. Indexes two document sources — a Human Nutrition PDF and an Investopedia economics dataset — into Pinecone, then answers queries using HuggingFace-hosted LLMs with advanced retrieval techniques.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Required env vars in `.env`:
- `PINECONE_API_KEY`
- `GEMINI_API_KEY`
- `HF_TOKEN`

## Running the Modules

```bash
# Build/load index and test a query against Pinecone
python indexing.py

# Test chunking logic and print chunk stats
python chunking.py

# List available Pinecone inference models
python pinecone_setup.py
```

All scripts read `config.yaml` from the working directory — run them from the project root.

## Architecture

### Data Flow
1. **`chunking.py`** — loads source documents and produces `data/chunks.pkl`
   - PDF path (`config.pdf_path`) is downloaded once to `data/human_nutrition.pdf`
   - Economics articles loaded from HuggingFace (`ksrepo/investopedia-dataset`) when `use_economics_subset: True`
   - `recursive_split()` splits by paragraphs → sentences → fixed word windows, controlled by `chunk_size` and `min_chunk_size`
   - Chunks are cached to `data/chunks.pkl`; delete that file to force re-chunking

2. **`indexing.py`** — embeds chunks and upserts to Pinecone
   - Embeddings via `SentenceTransformer` using the model in `config.embedding_model` (`Qwen/Qwen3-Embedding-0.6B`)
   - Pinecone index name set by `config.pinecone_index_name`; created automatically if missing (AWS us-east-1 serverless, cosine metric)
   - Upserts in batches of 100 using `ThreadPoolExecutor(max_workers=10)`
   - `load_index()` is the entry point — builds if the index doesn't exist, otherwise reuses it

3. **`memory.py`** — conversation memory with token-budget compression
   - `Memory` class persists chat history to a JSON file (default `chat_history.json`)
   - Tracks cumulative token count using the LLM's own tokenizer
   - When `token_count >= token_wall` (default 200,000), `summarize()` compresses older turns to ≤50 words and keeps the `memory_recent_k` most recent messages

### Configuration (`config.yaml`)
All runtime behaviour is controlled here. Key flags:

| Flag | Effect |
|------|--------|
| `use_query_rewrite` | Rewrites user query before retrieval |
| `use_multi_query` | Expands query into multiple variants |
| `use_hyde` | Generates a hypothetical document to embed instead of the query |
| `use_rerank` | Re-ranks retrieved chunks |
| `use_routing` | Routes query to correct domain (nutrition vs. economics) |
| `use_memory` | Enables `Memory` class for multi-turn conversations |

### Models
- **Embeddings**: `Qwen/Qwen3-Embedding-0.6B` via `sentence-transformers`; query embeddings use `prompt_name="query"` for asymmetric retrieval
- **LLM**: `Qwen/Qwen2.5-3B-Instruct` via `huggingface_hub.InferenceClient` (HF Inference API)

## Known Issues

- `chunking.py:86` has a typo: `current.srip()` should be `current.strip()` — this causes a crash when a chunk boundary is hit inside `recursive_split()`
- `create_chunks()` checks `config.get("us_econ_subset", False)` but the config key is `use_economics_subset` — economics chunks are silently skipped unless this is corrected
