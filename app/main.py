"""
DocMind AI v2 — FastAPI Backend
Enhanced RAG PDF Q&A with:
  - Multi-turn conversation memory
  - Per-document enable/disable
  - Auto-summary on upload
  - Confidence scoring
  - Smart sentence-aware chunking
  - Chat history export
"""

import os, math, time, re, logging
from collections import defaultdict
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
import PyPDF2, io

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="DocMind AI v2", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

API_KEY = os.environ.get("GEMINI_API_KEY", "")
client  = genai.Client(api_key=API_KEY) if API_KEY else None
MODEL   = "gemini-2.5-flash"

# ── Stores ────────────────────────────────────────────────────────────────────
chunk_store:   list[dict] = []   # {id, doc_id, doc, text, start}
doc_registry:  list[dict] = []   # {id, name, num_chunks, size, uploaded_at, enabled, summary}
conversations: dict[str, list]= {}  # session_id -> [{role, content}]

# ─────────────────────────────────────────────────────────────────────────────
# SMART CHUNKING (sentence-aware)
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_WORDS  = 400
OVERLAP_WORDS = 80

def sentence_chunk(text: str, doc_name: str, doc_id: str) -> list[dict]:
    """Split on sentence boundaries to avoid cutting mid-thought."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, buf, buf_words, cid = [], [], 0, 0
    for sent in sentences:
        w = len(sent.split())
        if buf_words + w > CHUNK_WORDS and buf:
            chunks.append({"id": cid, "doc_id": doc_id, "doc": doc_name,
                           "text": " ".join(buf), "start": cid * CHUNK_WORDS})
            cid += 1
            # keep overlap
            overlap, ow = [], 0
            for s in reversed(buf):
                sw = len(s.split())
                if ow + sw > OVERLAP_WORDS: break
                overlap.insert(0, s); ow += sw
            buf, buf_words = overlap, ow
        buf.append(sent); buf_words += w
    if buf:
        chunks.append({"id": cid, "doc_id": doc_id, "doc": doc_name,
                       "text": " ".join(buf), "start": cid * CHUNK_WORDS})
    log.info("Chunked '%s' -> %d sentence-aware chunks", doc_name, len(chunks))
    return chunks

# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────
STOPWORDS = {"the","a","an","is","it","in","on","at","to","for","of","and","or",
             "but","this","that","was","are","be","as","with","from","by","not","its","into","also"}

def tokenize(text: str) -> list[str]:
    return [w for w in re.sub(r'[^a-z\s]','',text.lower()).split()
            if w not in STOPWORDS and len(w) > 2]

def tfidf_score(q_terms: set, chunk: dict, total: int, df: dict) -> float:
    cw = tokenize(chunk["text"])
    if not cw: return 0.0
    return sum((cw.count(t)/len(cw)) * (math.log((total+1)/(df.get(t,0)+1))+1)
               for t in q_terms)

def retrieve(query: str, k: int = 6, enabled_docs: set = None) -> list[dict]:
    pool = [c for c in chunk_store
            if enabled_docs is None or c["doc_id"] in enabled_docs]
    if not pool: return []
    df: dict[str,int] = defaultdict(int)
    for c in pool:
        for t in set(tokenize(c["text"])): df[t] += 1
    q_terms = set(tokenize(query))
    scored = sorted([{**c,"score":tfidf_score(q_terms,c,len(pool),df)} for c in pool],
                    key=lambda x: x["score"], reverse=True)
    return [c for c in scored[:k] if c["score"] > 0]

def confidence(top_score: float) -> str:
    if top_score > 0.05: return "high"
    if top_score > 0.01: return "medium"
    return "low"

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM = """You are DocMind AI, a precise document Q&A assistant.

Rules:
- Answer ONLY from the provided document context below.
- If the answer isn't present, say: "This information is not in the loaded documents."
- Be concise but thorough. Use bullet points for lists.
- Cite which document/section the info comes from.
- Never invent facts outside the provided context."""

def qa_prompt(query: str, chunks: list[dict], history: list[dict]) -> str:
    ctx = "\n\n---\n\n".join(
        f'[Chunk {i+1} | "{c["doc"]}"]\n{c["text"]}'
        for i, c in enumerate(chunks)
    ) if chunks else "No relevant context found."

    hist_text = ""
    if history:
        hist_text = "\n\nConversation so far:\n" + "\n".join(
            f'{"User" if m["role"]=="user" else "Assistant"}: {m["content"]}'
            for m in history[-6:]  # last 3 turns
        )

    return f"Document context:\n\n{ctx}{hist_text}\n\n---\n\nQuestion: {query}\n\nAnswer:"

SUMMARY_PROMPT = """Read this document excerpt and write a 2-3 sentence summary covering:
1. What this document is about
2. Key topics or entities mentioned
Be concise and factual.

Document:
{text}"""

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    query:       str
    session_id:  Optional[str] = "default"
    top_k:       Optional[int] = 6
    enabled_docs:Optional[list[str]] = None  # list of doc_ids; None = all

class AskResponse(BaseModel):
    answer:           str
    sources:          list[str]
    chunks_retrieved: int
    top_score:        float
    confidence:       str
    context_tokens:   int
    model:            str
    latency_ms:       int
    session_id:       str

class UploadResponse(BaseModel):
    doc_id:     str
    doc_name:   str
    num_chunks: int
    summary:    str
    message:    str

class ToggleRequest(BaseModel):
    doc_id:  str
    enabled: bool

class ClearHistoryRequest(BaseModel):
    session_id: str

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def frontend():
    p = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    try:
        return HTMLResponse(open(p, encoding="utf-8").read())
    except FileNotFoundError:
        return HTMLResponse("<h2>Frontend not found.</h2>")

@app.get("/health")
def health():
    return {"status":"ok","chunks":len(chunk_store),"docs":len(doc_registry),
            "model":MODEL,"api_key_set":bool(API_KEY)}

@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".pdf",".txt",".md"}:
        raise HTTPException(400, f"Unsupported: {ext}")
    raw = await file.read()
    text = ""
    if ext == ".pdf":
        try:
            r = PyPDF2.PdfReader(io.BytesIO(raw))
            text = "\n".join(p.extract_text() or "" for p in r.pages)
        except Exception as e:
            raise HTTPException(422, f"PDF error: {e}")
    else:
        text = raw.decode("utf-8", errors="replace")
    if not text.strip():
        raise HTTPException(422, "Document is empty or unreadable.")

    doc_id = f"doc_{int(time.time()*1000)}"
    new_chunks = sentence_chunk(text, file.filename, doc_id)
    chunk_store.extend(new_chunks)

    # Auto-summarise first ~1500 words
    preview = " ".join(text.split()[:1500])
    summary = "Summary unavailable."
    if client:
        try:
            r = client.models.generate_content(
                model=MODEL,
                contents=SUMMARY_PROMPT.format(text=preview),
                config=types.GenerateContentConfig(max_output_tokens=200, temperature=0.3),
            )
            summary = r.text.strip()
        except Exception as e:
            log.warning("Summary failed: %s", e)

    size_kb = round(len(raw)/1024, 1)
    doc_registry.append({
        "id": doc_id, "name": file.filename,
        "num_chunks": len(new_chunks), "size": f"{size_kb} KB",
        "uploaded_at": time.strftime("%H:%M %d %b"), "enabled": True,
        "summary": summary,
    })
    return UploadResponse(doc_id=doc_id, doc_name=file.filename,
                          num_chunks=len(new_chunks), summary=summary,
                          message=f"Indexed {len(new_chunks)} chunks.")

@app.post("/toggle")
def toggle_doc(req: ToggleRequest):
    for doc in doc_registry:
        if doc["id"] == req.doc_id:
            doc["enabled"] = req.enabled
            return {"doc_id": req.doc_id, "enabled": req.enabled}
    raise HTTPException(404, "Document not found.")

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not client:
        raise HTTPException(500, "GEMINI_API_KEY not set.")
    if not chunk_store:
        raise HTTPException(400, "No documents loaded.")
    if not req.query.strip():
        raise HTTPException(400, "Empty query.")

    enabled = set(req.enabled_docs) if req.enabled_docs else \
              {d["id"] for d in doc_registry if d["enabled"]}

    t0         = time.time()
    top_chunks = retrieve(req.query, k=req.top_k, enabled_docs=enabled)
    history    = conversations.get(req.session_id, [])
    prompt     = qa_prompt(req.query, top_chunks, history)
    ctx_tokens = len(prompt.split())

    try:
        resp   = client.models.generate_content(
            model=MODEL, contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM, max_output_tokens=1200, temperature=0.2),
        )
        answer = resp.text
    except Exception as e:
        raise HTTPException(502, f"Gemini error: {e}")

    # Save to conversation history
    if req.session_id not in conversations:
        conversations[req.session_id] = []
    conversations[req.session_id].append({"role":"user",    "content": req.query})
    conversations[req.session_id].append({"role":"assistant","content": answer})
    # cap history at 20 messages
    conversations[req.session_id] = conversations[req.session_id][-20:]

    top_score  = top_chunks[0]["score"] if top_chunks else 0.0
    return AskResponse(
        answer=answer, sources=list({c["doc"] for c in top_chunks}),
        chunks_retrieved=len(top_chunks), top_score=round(top_score,4),
        confidence=confidence(top_score), context_tokens=ctx_tokens,
        model=MODEL, latency_ms=int((time.time()-t0)*1000),
        session_id=req.session_id,
    )

@app.get("/history/{session_id}")
def get_history(session_id: str):
    return {"session_id": session_id, "messages": conversations.get(session_id, [])}

@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    conversations.pop(session_id, None)
    return {"cleared": True}

@app.get("/docs-list")
def list_docs():
    return doc_registry

@app.delete("/docs-list")
def clear_docs():
    chunk_store.clear(); doc_registry.clear(); conversations.clear()
    return {"message": "Cleared."}