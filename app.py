import os, json, uuid
from typing import List, Optional, Dict
from pathlib import Path
import io
from fastapi import UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from typing import Any
from pypdf import PdfReader
import faiss
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
reranker = CrossEncoder(RERANK_MODEL)


# ---------- Config ----------
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
INDEX_PATH = DATA_DIR / "index.faiss"
META_PATH  = DATA_DIR / "meta.json"

MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))
USE_COSINE = True  # we’ll index normalized vectors in a L2 index to simulate cosine
DIM = None         # filled after model loads

# ---------- Load model ----------
embedder = SentenceTransformer(MODEL_NAME)
DIM = embedder.get_sentence_embedding_dimension()

# --- FAISS index helpers ---
def _new_index(dim: int):
    # Cosine similarity = inner product on L2-normalized vectors
    return faiss.IndexFlatIP(dim)

def embed_texts(texts: List[str]):
    vecs = embedder.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    # L2-normalize so IP == cosine
    import numpy as np
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms

if INDEX_PATH.exists():
    try:
        index = faiss.read_index(str(INDEX_PATH))
        # quick sanity: if its dimension doesn't match, recreate
        if index.d != DIM:
            index = _new_index(DIM)
    except Exception:
        index = _new_index(DIM)
else:
    index = _new_index(DIM)


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as bio:
        reader = PdfReader(bio)
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                texts.append(t)
        return "\n\n".join(texts)


# ---------- Metadata store ----------
# Keeps: global_id -> {"doc_id":..., "chunk":..., "meta": {...}}
if META_PATH.exists():
    with open(META_PATH, "r", encoding="utf-8") as f:
        META: Dict[str, Dict[str, Any]] = json.load(f)
else:
    META = {}

def _persist():
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(META, f, ensure_ascii=False)

# ---------- Chunking ----------
def chunk_text(text: str, max_tokens: int = 220, overlap: int = 40) -> List[str]:
    """
    Token-agnostic simple splitter (approximate by words). Good enough for day 1.
    Later: token-aware using tiktoken or nltk sentence splits + budget.
    """
    words = text.split()
    chunks = []
    i = 0
    step = max_tokens - overlap
    while i < len(words):
        chunk = " ".join(words[i:i+max_tokens])
        if chunk.strip():
            chunks.append(chunk)
        i += step
    return chunks


# ---------- API Schemas ----------
class IngestItem(BaseModel):
    text: str
    doc_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class IngestRequest(BaseModel):
    items: List[IngestItem]
    chunk_max_tokens: int = Field(default=220, ge=60, le=1000)
    chunk_overlap: int = Field(default=40, ge=0, le=500)

class IngestResponse(BaseModel):
    doc_ids: List[str]
    chunks_added: int

class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    include_text: bool = True

class Hit(BaseModel):
    id: str
    score: float
    doc_id: str
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    query: str
    results: List[Hit]

# ---------- App ----------
app = FastAPI(title="Local Semantic Search MVP", version="0.1.0")

@app.get("/health")
def health():
    return {"ok": True, "index_ntotal": index.ntotal, "meta_items": len(META), "model": MODEL_NAME}

@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    global META, index
    all_chunks = []
    owners = []
    metadata_list = []

    # Prepare chunks
    for item in req.items:
        doc_id = item.doc_id or str(uuid.uuid4())
        chunks = chunk_text(item.text, req.chunk_max_tokens, req.chunk_overlap)
        all_chunks.extend(chunks)
        owners.extend([doc_id] * len(chunks))
        metadata_list.extend([item.metadata or {}] * len(chunks))

    if not all_chunks:
        return IngestResponse(doc_ids=[], chunks_added=0)

    # Embed and add to FAISS
    vecs = embed_texts(all_chunks)

    # Build ids and meta entries
    import numpy as np
    start_id = len(META)
    # FAISS stores contiguous ids implicitly; we’ll simulate by order + META mapping
    # Note: IndexFlat doesn’t store ids, so we keep our own mapping (global list order).
    index.add(vecs)

    # Add metadata keyed by global order (stringified)
    for i, (chunk, doc_id, meta) in enumerate(zip(all_chunks, owners, metadata_list)):
        gid = str(start_id + i)
        META[gid] = {"doc_id": doc_id, "chunk": chunk, "meta": meta}

    _persist()
    return IngestResponse(doc_ids=list(set(owners)), chunks_added=len(all_chunks))

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if index.ntotal == 0:
        return QueryResponse(query=req.query, results=[])

    # 1) Dense retrieval (get a wider pool for better reranking)
    qvec = embed_texts([req.query])
    kprime = max(20, req.top_k)
    D, I = index.search(qvec, kprime)  # IP index -> D are cosine sims in [-1,1]

    # 2) Build preliminary hits (ALWAYS fetch text for reranker)
    prelim: List[Hit] = []
    for idx, sim in zip(I[0], D[0]):
        if idx < 0:
            continue
        gid = str(idx)
        meta = META.get(gid, {})
        prelim.append(Hit(
            id=gid,
            score=float(sim),                      # cosine similarity (higher better)
            doc_id=meta.get("doc_id", "unknown"),
            text=meta.get("chunk"),                # keep text here for reranker
            metadata=meta.get("meta"),
        ))

    # 3) Optional rerank with CrossEncoder (uses query, text)
    if prelim:
        pairs = [(req.query, h.text or "") for h in prelim]
        rerank_scores = reranker.predict(pairs).tolist()
        for h, s in zip(prelim, rerank_scores):
            h.score = float(s)
        prelim.sort(key=lambda h: h.score, reverse=True)

    # 4) Respect include_text flag for the response
    results = prelim[:req.top_k]
    if not req.include_text:
        for h in results:
            h.text = None

    return QueryResponse(query=req.query, results=results)

@app.post("/upload", response_model=IngestResponse)
def upload_file(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    chunk_max_tokens: int = Form(220),
    chunk_overlap: int = Form(40),
):
    """
    Accepts .pdf or .txt, extracts text, then reuses the /ingest pipeline.
    metadata: optional JSON string (stored as dict)
    """
    # Read file
    raw = file.file.read()
    if not raw:
        raise HTTPException(400, "Empty file")

    name = (file.filename or "").lower()
    ctype = (file.content_type or "").lower()
    text = ""

    try:
        if name.endswith(".pdf") or "pdf" in ctype:
            text = _extract_text_from_pdf(raw)
        elif name.endswith(".txt") or "text" in ctype:
            text = raw.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(415, f"Unsupported file type: {name} ({ctype})")
    finally:
        file.file.close()

    if not text.strip():
        raise HTTPException(422, "No extractable text found in the file.")

    # Parse metadata form field if provided
    meta_dict: Dict[str, Any] = {}
    if metadata:
        try:
            meta_dict = json.loads(metadata)
        except Exception:
            # ignore malformed metadata; keep it empty
            meta_dict = {}

    req = IngestRequest(
        items=[IngestItem(text=text, doc_id=doc_id, metadata=meta_dict)],
        chunk_max_tokens=chunk_max_tokens,
        chunk_overlap=chunk_overlap,
    )
    return ingest(req)


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Semantic Search MVP</title>
  <style>body{font-family:system-ui;margin:2rem;max-width:900px} .box{border:1px solid #ddd;padding:1rem;border-radius:12px;margin-bottom:1rem}</style>
</head>
<body>
  <h1>Semantic Search MVP</h1>

  <div class="box">
    <h3>1) Upload PDF/TXT</h3>
    <form id="up">
      <input type="file" name="file" accept=".pdf,.txt" required />
      <input type="text" name="doc_id" placeholder="doc_id (optional)" />
      <input type="text" name="metadata" placeholder='metadata JSON (optional)' />
      <button type="submit">Upload & Index</button>
    </form>
    <pre id="up_res"></pre>
  </div>

  <div class="box">
    <h3>2) Query</h3>
    <form id="qq">
      <input type="text" id="q" placeholder="Type your query..." style="width:70%" required />
      <input type="number" id="k" value="5" min="1" max="50" />
      <label><input type="checkbox" id="inc" checked /> include_text</label>
      <button type="submit">Search</button>
    </form>
    <pre id="q_res"></pre>
  </div>

<script>
document.getElementById('up').addEventListener('submit', async (e)=>{
  e.preventDefault();
  const fd = new FormData(e.target);
  const r = await fetch('/upload',{method:'POST', body:fd});
  document.getElementById('up_res').textContent = await r.text();
});

document.getElementById('qq').addEventListener('submit', async (e)=>{
  e.preventDefault();
  const body = {
    query: document.getElementById('q').value,
    top_k: parseInt(document.getElementById('k').value),
    include_text: document.getElementById('inc').checked
  };
  const r = await fetch('/query',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  document.getElementById('q_res').textContent = await r.text();
});
</script>
</body></html>
"""
