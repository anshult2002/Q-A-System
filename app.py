import os

import streamlit as st

from app.retrieval.retrieve import retrieve
from app.retrieval.rerank import rerank
from app.generation.generate import generate_answer, FALLBACK_ANSWER
from config import (
    DEFAULT_TOP_K_RETRIEVE,
    DEFAULT_TOP_K_RERANK,
    RERANK_SCORE_THRESHOLD,
)


st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("📚 Document Q&A")

query = st.text_input("Ask a question about your documents:")

top_k_retrieve = st.sidebar.slider(
    "Chunks to retrieve", 5, 50, DEFAULT_TOP_K_RETRIEVE
)
top_k_rerank = st.sidebar.slider(
    "Chunks after reranking", 1, 10, DEFAULT_TOP_K_RERANK
)
score_threshold = st.sidebar.number_input(
    "Rerank score threshold (tune to your data)",
    value=RERANK_SCORE_THRESHOLD,
    step=0.1,
    help=(
        "bge-reranker-v2-m3 outputs raw logits, not 0-1 probabilities. "
        "Run a few real queries and check the printed scores below to "
        "find a sane cutoff for 'not actually relevant'."
    ),
)

if st.button("Search") and query.strip():

    with st.spinner("Retrieving relevant chunks..."):
        retrieved = retrieve(query, top_k=top_k_retrieve)

    if not retrieved:
        st.warning("No relevant documents found.")
    else:
        with st.spinner("Reranking..."):
            reranked = rerank(query, retrieved, top_k=top_k_rerank)

        best_score = reranked[0]["rerank_score"] if reranked else float("-inf")

        st.sidebar.caption(f"Top rerank score: {best_score:.3f}")

        if best_score < score_threshold:
            st.warning(
                "I couldn't find anything in your documents that actually "
                "answers this — try rephrasing, or it may just not be covered."
            )
        else:
            with st.spinner("Generating answer..."):
                answer = generate_answer(query, reranked)

            st.markdown("### Answer")

            if answer == FALLBACK_ANSWER:
                st.info(answer)
            else:
                st.write(answer)

            st.markdown("### Sources")

            for chunk in reranked:
                pdf_name = chunk.get("pdf_name", "Unknown")
                pdf_path = chunk.get("pdf_path", "")
                start_page = chunk.get("start_page")
                end_page = chunk.get("end_page")
                score = chunk.get("rerank_score", 0)

                col1, col2 = st.columns([4, 1])

                with col1:
                    st.markdown(
                        f"**{pdf_name}** — pages {start_page}-{end_page} "
                        f"(score: {score:.3f})"
                    )
                    with st.expander("Show excerpt"):
                        st.write(chunk["text"])

                with col2:
                    if pdf_path and os.path.exists(pdf_path):
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                label="Open PDF",
                                data=f,
                                file_name=pdf_name,
                                mime="application/pdf",
                                key=f"{pdf_name}_{start_page}_{end_page}",
                            )
                    else:
                        st.caption("File not found locally")
