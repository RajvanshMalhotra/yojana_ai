import os
import pickle
import yaml
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
from chunking import create_chunks
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer
load_dotenv()


def get_embeddings(texts: list[str],config: dict,task_type: str = "RETRIEVAL_DOCUMENT",) -> list[list[float]]:
    model = SentenceTransformer(config["embedding_model"])
    
    if task_type == "RETRIEVAL_QUERY":
        embeddings = model.encode(texts, prompt_name="query", normalize_embeddings=True, show_progress_bar=True)
    else:
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    
    return embeddings.tolist()



def build_index(config):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    # Load scheme chunks from yojan_chunking_temp.py output
    chunks_path = os.path.join(config["data_dir"], "mlx_enriched", "scheme_chunks.pkl")
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    valid_chunks = [c for c in chunks if c and c.get("text", "").strip()]
    texts = [c["text"] for c in valid_chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")

    embeddings = get_embeddings(texts, config)
    dim = len(embeddings[0])

    index_name = config["pinecone_index_name"]

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            vector_type="dense",
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            deletion_protection="disabled",
            tags={"environment": "development"}
        )
    print("Index ready")
    index = pc.Index(index_name)

    vectors = []
    for i, chunk in enumerate(valid_chunks):
        vectors.append({
            "id": str(chunk["chunk_id"]),
            "values": embeddings[i],
            "metadata": {
                "text":                chunk["text"],
                "title":               chunk["title"],
                "domain":              chunk["domain"],
                "source":              chunk.get("source", ""),
                "start_page":          chunk.get("start_page") or 0,
                # scheme-specific fields — used as Pinecone filters in retrieval.py
                "beneficiaries":       chunk.get("beneficiaries", []),
                "category":            chunk.get("category", "other"),
                "benefit_summary":     chunk.get("benefit_summary", ""),
                "eligibility_summary": chunk.get("eligibility_summary", ""),
                "tags":                chunk.get("tags", []),
            }
        })

    batch_size = 100
    batches = [vectors[i:i + batch_size] for i in range(0, len(vectors), batch_size)]
    print(f"Upserting {len(vectors)} vectors in {len(batches)} batches...")

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(index.upsert, vectors=batch, timeout=60) for batch in batches]
        [f.result() for f in futures]

    print("Upserted all vectors")
    return index, valid_chunks


def load_index(config):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = config.get("pinecone_index_name")

    if not pc.has_index(index_name):
        print(f"Index '{index_name}' not found — building...")
        return build_index(config)

    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    if stats["total_vector_count"] == 0:
        print(f"Index '{index_name}' is empty — building...")
        return build_index(config)

    print("Index loaded from Pinecone")
    chunks_path = os.path.join(config["data_dir"], "mlx_enriched", "scheme_chunks.pkl")
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    valid_chunks = [c for c in chunks if c and c.get("text", "").strip()]
    return index, valid_chunks
if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    print("=== Loading / Building Index ===")
   
    index, chunks = load_index(config)

    print("\n=== Testing Query ===")
    test_query = "pension scheme for elderly women in Delhi"
    print(f"Query: {test_query}")

    query_embedding = query_embedding = get_embeddings([test_query], config, task_type="RETRIEVAL_QUERY")[0]

    matches = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    print(f"\nTop {len(matches['matches'])} results:")
    for i, match in enumerate(matches["matches"]):
        print(f"\n--- Match {i+1} ---")
        print(f"Score : {match['score']:.4f}")
        print(f"Domain: {match['metadata']['domain']}")
        print(f"Title : {match['metadata']['title']}")
        print(f"Page  : {match['metadata']['start_page']}")
        print(f"Text  : {match['metadata']['text'][:200]}...")

