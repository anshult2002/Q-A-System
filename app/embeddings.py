from typing import List

import logfire
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE


# Load the embedding model once.
model = SentenceTransformer(
    EMBEDDING_MODEL,
    trust_remote_code=True
)


def embed_query(query: str) -> List[float]:
    """
    Generate an embedding for a single query.

    Args:
        query: User's search query.

    Returns:
        Embedding vector as a list of floats.
    """

    with logfire.span(
        "Embed query",
        model=EMBEDDING_MODEL,
        query_length=len(query)
    ):

        embedding = model.encode(
            query,
            normalize_embeddings=True
        )

        logfire.info(
            "Query embedding generated",
            embedding_dimension=len(embedding)
        )

        return embedding.tolist()


def embed_documents(
    documents: List[str],
    batch_size: int = EMBEDDING_BATCH_SIZE
) -> List[List[float]]:
    """
    Generate embeddings for document chunks in batches.

    Args:
        documents:
            List of chunk texts.

        batch_size:
            Number of chunks embedded at once.

    Returns:
        List of embedding vectors.
    """

    all_embeddings: List[List[float]] = []

    for i in range(0, len(documents), batch_size):

        batch = documents[i:i + batch_size]

        with logfire.span(
            "Embed batch",
            model=EMBEDDING_MODEL,
            start=i,
            size=len(batch)
        ):

            embeddings = model.encode(
                batch,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            all_embeddings.extend(
                embeddings.tolist()
            )

            logfire.info(
                "Embedding batch completed",
                start=i,
                size=len(batch),
                total_embeddings=len(all_embeddings)
            )

    return all_embeddings
