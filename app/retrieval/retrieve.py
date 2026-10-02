from typing import List, Dict, Any

import chromadb
import logfire

from app.embeddings import embed_query
from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    DEFAULT_TOP_K_RETRIEVE,
)


# ---------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------
#
# get_or_create_collection (not get_collection) so importing
# this module before ingestion has ever run doesn't crash —
# it'll just be empty until process_directory() populates it.

client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH
)

collection = client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME,
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    }
)


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K_RETRIEVE
) -> List[Dict[str, Any]]:
    """
    Retrieve the most similar chunks from ChromaDB.

    Flow:

        Query
          ↓
        BGE-M3
          ↓
        ChromaDB
          ↓
        Top-K chunks
    """

    if not query or not query.strip():
        return []

    with logfire.span(
        "Document Retrieval",
        top_k=top_k
    ):

        # -------------------------------------------------
        # 1. Embed query
        # -------------------------------------------------

        query_embedding = embed_query(
            query
        )

        if not query_embedding:
            return []

        # -------------------------------------------------
        # 2. Search ChromaDB
        # -------------------------------------------------

        results = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        # -------------------------------------------------
        # 3. Build results
        # -------------------------------------------------

        retrieved_results = []

        for i in range(
            len(documents)
        ):

            metadata = metadatas[i]

            retrieved_results.append({

                "text": documents[i],

                "retrieval_distance": float(
                    distances[i]
                ),

                "pdf_name": metadata.get(
                    "pdf_name",
                    "Unknown PDF"
                ),

                # Local path back to the source file — used by the
                # UI to open/download the exact PDF this chunk came from
                "pdf_path": metadata.get(
                    "pdf_path",
                    ""
                ),

                "page_numbers": metadata.get(
                    "page_numbers",
                    ""
                ),

                "start_page": metadata.get(
                    "start_page"
                ),

                "end_page": metadata.get(
                    "end_page"
                ),
            })

        logfire.info(
            "Documents retrieved",
            count=len(retrieved_results)
        )

        return retrieved_results
