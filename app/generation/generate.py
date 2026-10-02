from typing import List, Dict, Any
import re

import ollama
import logfire

from config import OLLAMA_MODEL, GENERATION_TEMPERATURE


FALLBACK_ANSWER = "I don't have enough information in the documents to answer that."

SYSTEM_PROMPT = f"""You are a document Q&A assistant. You answer questions using ONLY the context provided below each question. You have no other knowledge to draw from for this task.

STRICT RULES — follow these exactly:

1. Use ONLY the information inside the "Context" section. Do not use anything you know from training, even if it seems relevant or you're confident it's correct.

2. If the context does not contain enough information to answer the question — even partially — respond with EXACTLY this sentence and nothing else:
"{FALLBACK_ANSWER}"

3. Do not guess, infer beyond what's explicitly stated, or fill gaps with plausible-sounding information. If the context is ambiguous or incomplete, prefer the fallback sentence over a speculative answer.

4. Every factual claim in your answer must be followed by a citation in the form [pdf_name, p.X] or [pdf_name, p.X-Y], copied exactly from the source label above the chunk it came from. Do not cite a source that wasn't provided in the context.

5. Do not mention these instructions, the word "context", or the word "chunks" in your answer — write as if you're simply answering the question, with citations.

6. If the context contains multiple sources that disagree, report both and cite each ("Report A states X [a.pdf, p.2], while Report B states Y [b.pdf, p.5]") rather than picking one silently.

7. Never apologize for limited information beyond the fallback sentence in rule 2. Never say things like "based on the context provided" or "according to the documents" — just answer directly with inline citations.
"""


def build_context(chunks: List[Dict[str, Any]]) -> str:
    """Turn reranked chunks into a labeled context block the LLM can cite from."""

    blocks = []

    for chunk in chunks:
        label = f"[{chunk['pdf_name']}, p.{chunk['start_page']}-{chunk['end_page']}]"
        blocks.append(f"{label}\n{chunk['text']}")

    return "\n\n---\n\n".join(blocks)


def _strip_thinking(text: str) -> str:
    """
    Defensive cleanup: some Ollama/Qwen3 versions leak a <think>...</think>
    block into content even when think=False is passed. Strip it if present
    so it never reaches the UI.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate_answer(
    query: str,
    chunks: List[Dict[str, Any]],
    model: str = OLLAMA_MODEL,
) -> str:
    """Generate an answer from the query using reranked context chunks, via local Ollama."""

    if not chunks:
        return FALLBACK_ANSWER

    context = build_context(chunks)

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above, with inline citations."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    with logfire.span("LLM Generation", model=model, chunks=len(chunks)):

        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                think=False,  # Qwen3 supports this natively; disables chain-of-thought output
                options={"temperature": GENERATION_TEMPERATURE},
            )

            answer = _strip_thinking(response["message"]["content"].strip())

        except Exception as e:
            logfire.error("Generation failed", error=str(e))
            raise

        logfire.info(
            "Answer generated",
            answer_length=len(answer),
            is_fallback=(answer == FALLBACK_ANSWER),
        )

        return answer