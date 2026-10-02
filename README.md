# Local Document Q&A (RAG)

PDF ingestion → BGE-M3 embeddings → ChromaDB → BGE reranker → local Ollama LLM → Streamlit UI with links back to source PDFs.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Set up Ollama

```bash
# install Ollama: https://ollama.com/download
ollama pull llama3.1
ollama serve
```

Change the model in `config.py` (`OLLAMA_MODEL`) if you pull a different one.

## 3. Put your PDFs in a folder

```bash
mkdir DATA
# copy your PDFs into DATA/
```

## 4. Ingest the PDFs

```bash
python -m app.ingestion.processor DATA
```

Add `--wipe` to clear the existing ChromaDB collection first:

```bash
python -m app.ingestion.processor DATA --wipe
```

Re-running this command on the same folder is safe — it upserts, so existing chunks are updated rather than duplicated.

## 5. Run the app

```bash
streamlit run app.py
```

## Notes

- **Source links**: chunk metadata stores the original `pdf_path`, so the UI can serve the exact source PDF via a download button. Browsers block `file://` links for security, so a direct clickable path isn't possible — the download button is the reliable local equivalent.
- **Rerank threshold**: `bge-reranker-v2-m3` returns raw logits, not a 0–1 score. Run a few real queries, note the "Top rerank score" shown in the sidebar for both good and irrelevant queries, and adjust the threshold in the sidebar (or `RERANK_SCORE_THRESHOLD` in `config.py`) accordingly.
- **Config**: all tunable settings (chunk size, model names, top-k, DB path) live in `config.py` and can be overridden via environment variables of the same name.
- **Re-ingestion**: if you change chunking or metadata logic, re-run step 4 — old vectors in `my_chroma_db` won't have new metadata fields until re-processed.
