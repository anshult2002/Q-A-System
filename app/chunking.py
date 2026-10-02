from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
import logfire

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_pages(pages: List[dict]) -> List[dict]:
    """
    Split PDF text into overlapping chunks while preserving
    PDF name, path, and page references.

    Args:
        pages:
            Output from parse_pdf():

            [
                {
                    "pdf_name": "paper.pdf",
                    "pdf_path": "data/paper.pdf",
                    "page_number": 1,
                    "text": "..."
                },
                ...
            ]

    Returns:
        A list of chunks containing text and metadata.
    """

    with logfire.span("Text Chunking"):

        if not pages:
            return []

        # -----------------------------------------
        # Build continuous text
        # -----------------------------------------

        full_text = ""
        page_ranges = []

        for page in pages:

            start = len(full_text)
            full_text += page["text"]
            end = len(full_text)

            page_ranges.append({
                "page_number": page["page_number"],
                "start": start,
                "end": end,
            })

            # Keep pages separated
            full_text += "\n\n"

        # -----------------------------------------
        # Split text
        # -----------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ".",
                "!",
                "?",
                ",",
                " ",
                ""
            ],
        )

        chunks = splitter.split_text(full_text)

        # -----------------------------------------
        # Add page references
        # -----------------------------------------

        results = []
        search_position = 0

        for chunk_id, chunk in enumerate(chunks):

            chunk = chunk.strip()

            if not chunk:
                continue

            # Find chunk in original continuous text
            start_position = full_text.find(
                chunk,
                search_position,
            )

            if start_position == -1:
                continue

            end_position = start_position + len(chunk)
            search_position = start_position + 1

            # Find pages covered by this chunk
            page_numbers = []

            for page in page_ranges:

                if (
                    start_position < page["end"]
                    and end_position > page["start"]
                ):
                    page_numbers.append(
                        page["page_number"]
                    )

            if not page_numbers:
                continue

            results.append({
                "chunk_id": chunk_id,
                "text": chunk,
                "metadata": {
                    "pdf_name": pages[0]["pdf_name"],
                    "pdf_path": pages[0]["pdf_path"],
                    "page_numbers": page_numbers,
                    "start_page": page_numbers[0],
                    "end_page": page_numbers[-1],
                },
            })

        logfire.info(
            "Generated chunks",
            pdf_name=pages[0]["pdf_name"],
            chunks=len(results),
        )

        return results
