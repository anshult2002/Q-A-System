from typing import List, Dict, Any

import logfire
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL, DEFAULT_TOP_K_RERANK


# ---------------------------------------------------------
# Global Reranker
# ---------------------------------------------------------

reranker = CrossEncoder(
    RERANKER_MODEL,
    trust_remote_code=True
)


# ---------------------------------------------------------
# Rerank
# ---------------------------------------------------------

def rerank(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int = DEFAULT_TOP_K_RERANK
) -> List[Dict[str, Any]]:
    """
    Rerank retrieved documents using BGE reranker.

    Args:
        query:
            User's query.

        documents:
            Results returned by retrieve.py.

        top_k:
            Number of final results.

    Returns:
        Reranked documents, sorted best-first, each with an
        added "rerank_score" field.
    """

    if not query or not documents:
        return []

    with logfire.span(
        "Document Reranking",
        candidates=len(documents)
    ):

        # -------------------------------------------------
        # 1. Create query-document pairs
        # -------------------------------------------------

        pairs = [
            [
                query,
                document["text"]
            ]
            for document in documents
        ]

        # -------------------------------------------------
        # 2. Calculate reranker scores
        # -------------------------------------------------

        scores = reranker.predict(
            pairs
        )

        # -------------------------------------------------
        # 3. Add scores
        # -------------------------------------------------

        ranked_documents = []

        for document, score in zip(
            documents,
            scores
        ):

            result = document.copy()

            result["rerank_score"] = float(
                score
            )

            ranked_documents.append(
                result
            )

        # -------------------------------------------------
        # 4. Sort by reranker score
        # -------------------------------------------------

        ranked_documents.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        # -------------------------------------------------
        # 5. Return top-K
        # -------------------------------------------------

        final_results = ranked_documents[
            :top_k
        ]

        logfire.info(
            "Documents reranked",
            returned=len(final_results)
        )

        return final_results
