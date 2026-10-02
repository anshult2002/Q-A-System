import os

# ---------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./my_chroma_db")

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "my_documents")


# ---------------------------------------------------------
# Chunking
# ---------------------------------------------------------

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))

CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


# ---------------------------------------------------------
# Embeddings
# ---------------------------------------------------------

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))


# ---------------------------------------------------------
# Reranking
# ---------------------------------------------------------

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

DEFAULT_TOP_K_RETRIEVE = int(os.getenv("DEFAULT_TOP_K_RETRIEVE", "20"))

DEFAULT_TOP_K_RERANK = int(os.getenv("DEFAULT_TOP_K_RERANK", "5"))

# bge-reranker-v2-m3 outputs raw logits, not a 0-1 probability.
# Run a few real queries (good match vs. clearly irrelevant) and
# print rerank_score to calibrate a sane cutoff for your data.
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.0"))


# ---------------------------------------------------------
# Generation (Ollama)
# ---------------------------------------------------------

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.1"))


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------

PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "processed_data")
