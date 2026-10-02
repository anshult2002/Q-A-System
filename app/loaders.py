from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
import logfire


def parse_pdf(pdf_path: str) -> list[dict]:
    """
    Extract PDF text page-by-page while preserving
    the PDF name, path, and page number.

    Args:
        pdf_path: Path to the PDF.

    Returns:
        List of pages:
        [
            {
                "pdf_name": "paper.pdf",
                "pdf_path": "data/paper.pdf",
                "page_number": 1,
                "text": "..."
            },
            ...
        ]
    """

    with logfire.span("PDF Loading", pdf_path=pdf_path):

        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()

            pdf_name = Path(pdf_path).name

            pages = []

            for page_number, document in enumerate(
                documents,
                start=1
            ):
                text = document.page_content.strip()

                if not text:
                    continue

                pages.append({
                    "pdf_name": pdf_name,
                    "pdf_path": pdf_path,
                    "page_number": page_number,
                    "text": text,
                })

            logfire.info(
                "PDF loaded successfully",
                pdf_name=pdf_name,
                pages=len(pages),
                characters=sum(
                    len(page["text"])
                    for page in pages
                ),
            )

            return pages

        except Exception as e:

            logfire.error(
                "Failed to load PDF",
                error=str(e),
            )

            raise
