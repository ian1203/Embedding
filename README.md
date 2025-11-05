# Local Semantic Search MVP

An end-to-end **vector embedding + retrieval** MVP focused on **automation and UX**.  
Upload documents (PDF/TXT), chunk & embed them with `SentenceTransformers`, index in FAISS, and query via a simple UI or JSON API.

> Vision: make **semantic search / RAG pipelines** turnkey for non-technical users—fast ingestion, sane defaults, and “it just works” quality (cosine sim, reranking, soon hybrid BM25 + dense and monitoring).

---

## Features (current)
- 📄 **Upload** PDF/TXT via `/upload` or a minimal `/ui` page
- ✂️ **Chunking**: word-window with overlap (token-aware coming next)
- 🔡 **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- 🔎 **Vector index**: FAISS `IndexFlatIP` (true cosine via L2-normalized vectors)
- 🔁 **(Optional) Rerank**: CrossEncoder `ms-marco-MiniLM-L-6-v2`
- 🧪 **Swagger**: `/docs`
- 🩺 **Health**: `/health`

---

## Quick start

### Requirements
- Python **3.12**
- macOS (x86_64) tested; should work on Linux with similar versions.

### Install
```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel setuptools
pip install \
  "numpy<2.0" \
  torch==2.2.2 \
  sentence-transformers==2.6.1 \
  faiss-cpu==1.8.0.post1 \
  fastapi==0.115.4 \
  uvicorn==0.30.6 \
  ujson==5.10.0 \
  pypdf==5.1.0
