import os
import sys
import json

import chromadb
import logfire

from app.loaders import parse_pdf
from app.chunking import chunk_pages
from app.embeddings import embed_documents
from config import (
    PROCESSED_DATA_DIR,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
)


# ---------------------------------------------------------
# Logfire
# ---------------------------------------------------------

logfire.configure(
    service_name="enterprise-ingestion-service"
)


# ---------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------

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
# Save processed data
# ---------------------------------------------------------

def save_processed_locally(
    data: dict,
    filename: str
) -> str:
    """
    Save processed PDF data as JSON.
    """

    folder = os.path.join(
        PROCESSED_DATA_DIR
    )

    os.makedirs(
        folder,
        exist_ok=True
    )

    dest = os.path.join(
        folder,
        f"{filename}.json"
    )

    with open(
        dest,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    return dest


# ---------------------------------------------------------
# Process one PDF
# ---------------------------------------------------------

def process_file(
    file_path: str,
    filename: str
):
    """
    Process one PDF:

        PDF
        ↓
        parse_pdf()
        ↓
        chunk_pages()
        ↓
        save processed JSON
        ↓
        BGE-M3 embeddings
        ↓
        ChromaDB
    """

    with logfire.span(
        "Processing File",
        file=filename
    ):

        try:

            # -------------------------------------------------
            # 1. Parse PDF page-by-page
            # -------------------------------------------------

            pages = parse_pdf(
                file_path
            )

            if not pages:

                logfire.warning(
                    f"No text extracted from "
                    f"{filename} — skipping."
                )

                return

            # -------------------------------------------------
            # 2. Chunk while preserving page information
            # -------------------------------------------------

            chunks = chunk_pages(
                pages
            )

            if not chunks:

                logfire.warning(
                    f"No chunks generated from "
                    f"{filename} — skipping."
                )

                return

            # -------------------------------------------------
            # 3. Save processed chunks locally
            # -------------------------------------------------

            processed_data = {
                "filename": filename,
                "chunks": chunks
            }

            local_path = save_processed_locally(
                processed_data,
                filename
            )

            logfire.info(
                f"Saved processed data → {local_path}"
            )

            # -------------------------------------------------
            # 4. Extract chunk text
            # -------------------------------------------------

            texts = [
                chunk["text"]
                for chunk in chunks
            ]

            # -------------------------------------------------
            # 5. Generate BGE-M3 embeddings
            # -------------------------------------------------

            with logfire.span(
                "Embedding Documents",
                chunks=len(texts)
            ):

                embeddings = embed_documents(
                    texts
                )

            # -------------------------------------------------
            # 6. Prepare ChromaDB records
            # -------------------------------------------------

            ids = []
            documents = []
            metadatas = []

            for chunk in chunks:

                chunk_id = chunk["chunk_id"]
                metadata = chunk["metadata"]

                # Unique ID across all PDFs
                vector_id = (
                    f"{metadata['pdf_name']}_{chunk_id}"
                )

                ids.append(
                    vector_id
                )

                documents.append(
                    chunk["text"]
                )

                metadatas.append({

                    # PDF filename
                    "pdf_name": metadata["pdf_name"],

                    # Full local path — used to open/download the
                    # source file straight from the UI
                    "pdf_path": metadata["pdf_path"],

                    # Example: "10,11"
                    "page_numbers": ",".join(
                        map(
                            str,
                            metadata["page_numbers"]
                        )
                    ),

                    # First page containing the chunk
                    "start_page": metadata["start_page"],

                    # Last page containing the chunk
                    "end_page": metadata["end_page"],

                })

            # -------------------------------------------------
            # 7. Store in ChromaDB
            # -------------------------------------------------
            #
            # upsert() (not add()) so re-running ingestion on the
            # same PDFs updates existing vectors instead of
            # raising a duplicate-ID error.

            with logfire.span(
                "Storing in ChromaDB",
                chunks=len(chunks)
            ):

                collection.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )

            logfire.info(
                f"Indexed {len(chunks)} chunks "
                f"from {filename} into ChromaDB."
            )

        except Exception as e:

            logfire.error(
                f"Failed to process {filename}: {e}"
            )

            raise


# ---------------------------------------------------------
# Process directory
# ---------------------------------------------------------

def process_directory(
    dir_path: str
):
    """
    Process every PDF in a single directory.
    """

    with logfire.span(
        "Scanning Directory",
        path=dir_path
    ):

        files = [
            filename
            for filename in os.listdir(dir_path)
            if os.path.isfile(
                os.path.join(
                    dir_path,
                    filename
                )
            )
            and filename.lower().endswith(".pdf")
        ]

        logfire.info(
            f"Found {len(files)} PDF files "
            f"in {dir_path}."
        )

        for filename in files:

            file_path = os.path.join(
                dir_path,
                filename
            )

            process_file(
                file_path,
                filename
            )


# ---------------------------------------------------------
# Run ingestion
# ---------------------------------------------------------

def run_ingestion(
    base_dir: str,
    wipe: bool = False
):
    """
    Ingest every PDF from one directory.

    Args:
        base_dir:
            Directory containing all PDFs.

        wipe:
            Delete the existing ChromaDB collection
            before ingestion.
    """

    global collection

    # -----------------------------------------------------
    # Wipe existing collection
    # -----------------------------------------------------

    if wipe:

        with logfire.span(
            "Wiping ChromaDB Collection"
        ):

            try:

                client.delete_collection(
                    name=CHROMA_COLLECTION_NAME
                )

                logfire.info(
                    f"Deleted collection "
                    f"'{CHROMA_COLLECTION_NAME}'."
                )

            except Exception:

                logfire.info(
                    "Collection did not exist."
                )

            # Recreate collection
            collection = client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                configuration={
                    "hnsw": {
                        "space": "cosine"
                    }
                }
            )

    # -----------------------------------------------------
    # Process all PDFs
    # -----------------------------------------------------

    process_directory(
        base_dir
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    """
    Usage:

        python -m app.ingestion.processor DATA

    To delete the existing ChromaDB collection first:

        python -m app.ingestion.processor DATA --wipe
    """

    wipe_requested = (
        "--wipe" in sys.argv
    )

    clean_args = [
        arg
        for arg in sys.argv
        if arg != "--wipe"
    ]

    target_dir = (
        clean_args[1]
        if len(clean_args) > 1
        else "DATA"
    )

    if not os.path.exists(
        target_dir
    ):

        print(
            f"Error: path '{target_dir}' does not exist."
        )

        sys.exit(1)

    run_ingestion(
        target_dir,
        wipe=wipe_requested
    )

    logfire.info(
        "Ingestion job completed."
    )
