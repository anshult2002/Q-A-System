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


st.set_page_config(page_title="Document Chatbot", layout="wide")
st.title("💬 Document Chatbot")

# ---------------------------------------------------------
# Sidebar settings
# ---------------------------------------------------------

top_k_retrieve = st.sidebar.slider("Chunks to retrieve", 5, 50, DEFAULT_TOP_K_RETRIEVE)
top_k_rerank = st.sidebar.slider("Chunks after reranking", 1, 10, DEFAULT_TOP_K_RERANK)
score_threshold = st.sidebar.number_input(
    "Rerank score threshold", value=RERANK_SCORE_THRESHOLD, step=0.1
)
history_turns_to_keep = st.sidebar.slider(
    "Conversation turns to remember", 1, 10, 3,
    help="How many previous Q&A exchanges the model can see, for follow-up questions."
)

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "sources": [...]}]


def render_sources(sources, key_prefix):
    with st.expander("Sources"):
        for chunk in sources:
            pdf_name = chunk.get("pdf_name", "Unknown")
            pdf_path = chunk.get("pdf_path", "")
            start_page = chunk.get("start_page")
            end_page = chunk.get("end_page")
            score = chunk.get("rerank_score", 0)

            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{pdf_name}** — pages {start_page}-{end_page} (score: {score:.3f})")
            with col2:
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="Open PDF",
                            data=f,
                            file_name=pdf_name,
                            mime="application/pdf",
                            key=f"{key_prefix}_{pdf_name}_{start_page}_{end_page}",
                        )
                else:
                    st.caption("File not found locally")


# ---------------------------------------------------------
# Render existing conversation
# ---------------------------------------------------------

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_sources(message["sources"], key_prefix=f"hist{i}")

# ---------------------------------------------------------
# Handle new input
# ---------------------------------------------------------

user_input = st.chat_input("Ask a question about your documents...")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Retrieving relevant chunks..."):
            retrieved = retrieve(user_input, top_k=top_k_retrieve)

        answer = None
        sources = []

        if not retrieved:
            answer = "No relevant documents found."
            st.warning(answer)
        else:
            with st.spinner("Reranking..."):
                reranked = rerank(user_input, retrieved, top_k=top_k_rerank)

            best_score = reranked[0]["rerank_score"] if reranked else float("-inf")

            if best_score < score_threshold:
                answer = FALLBACK_ANSWER
                st.info(answer)
            else:
                # Prior turns (excluding the one just appended), trimmed to N exchanges
                prior = st.session_state.messages[:-1]
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in prior
                ][-(history_turns_to_keep * 2):]

                with st.spinner("Generating answer..."):
                    answer = generate_answer(user_input, reranked, history=history)

                sources = reranked

                if answer == FALLBACK_ANSWER:
                    st.info(answer)
                else:
                    st.write(answer)

                if sources:
                    render_sources(sources, key_prefix=f"new_{len(st.session_state.messages)}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })