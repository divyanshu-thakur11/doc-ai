"""
DocMind AI — FastAPI Backend (Gemini Free Tier)
RAG-based PDF Q&A System using Google Gemini API (100% free, no billing)

Get your free API key at: https://aistudio.google.com/app/apikey
"""

import os
import math
import time
import logging
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
import PyPDF2
import io

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DocMind AI",
    description="RAG-based PDF Q&A using Google Gemini (free tier)",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Gemini client ─────────────────────────────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY", "")
client  = genai.Client(api_key=API_KEY) if API_KEY else None
MODEL   = "gemini-2.5-flash"

# ── In-memory store ───────────────────────────────────────────────────────────
chunk_store:  list[dict] = []
doc_registry: list[dict] = []

# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_SIZE   = 500
CHUNK_STRIDE = 400

def chunk_text(text: str, doc_name: str) -> list[dict]:
    words = text.split()
    chunks, i, cid = [], 0, 0
    while i < len(words):
        chunks.append({
            "id":    cid,
            "doc":   doc_name,
            "text":  " ".join(words[i : i + CHUNK_SIZE]),
            "start": i,
        })
        cid += 1
        i   += CHUNK_STRIDE
    log.info("Chunked '%s' -> %d chunks", doc_name, len(chunks))
    return chunks

# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────
STOPWORDS = {
    "the","a","an","is","it","in","on","at","to","for","of","and","or",
    "but","this","that","was","are","be","as","with","from","by","not","its","into"
}

def tokenize(text: str) -> list[str]:
    return [w for w in text.lower().split() if w.isalpha() and w not in STOPWORDS and len(w) > 2]

def tfidf_score(query: str, chunk: dict, total: int, df: dict) -> float:
    q_terms = set(tokenize(query))
    c_words  = tokenize(chunk["text"])
    if not c_words:
        return 0.0
    score = 0.0
    for t in q_terms:
        tf  = c_words.count(t) / len(c_words)
        idf = math.log((total + 1) / (df.get(t, 0) + 1)) + 1
        score += tf * idf
    return score

def retrieve_top_k(query: str, k: int = 5) -> list[dict]:
    if not chunk_store:
        return []
    df: dict[str, int] = defaultdict(int)
    for c in chunk_store:
        for t in set(tokenize(c["text"])):
            df[t] += 1
    scored = [{**c, "score": tfidf_score(query, c, len(chunk_store), df)} for c in chunk_store]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return [c for c in scored[:k] if c["score"] > 0]

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are DocMind AI, a precise document Q&A assistant.
Rules:
- Answer ONLY from the provided document chunks below.
- If the answer is not present, say: "This information is not in the loaded documents."
- Be concise but complete. Use bullet points for lists.
- Mention which document or section the info comes from.
- Never invent facts not present in the context."""

def build_prompt(query: str, chunks: list[dict]) -> str:
    if not chunks:
        context = "No relevant chunks found."
    else:
        context = "\n\n---\n\n".join(
            f'[Chunk {i+1} from "{c["doc"]}" - words {c["start"]}-{c["start"]+CHUNK_SIZE}]\n{c["text"]}'
            for i, c in enumerate(chunks)
        )
    return f"Document context:\n\n{context}\n\n---\n\nQuestion: {query}\n\nAnswer based only on the context above:"

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class AskResponse(BaseModel):
    answer:           str
    sources:          list[str]
    chunks_retrieved: int
    top_score:        float
    context_tokens:   int
    model:            str
    latency_ms:       int

class UploadResponse(BaseModel):
    doc_name:   str
    num_chunks: int
    message:    str

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h2>Frontend not found.</h2>")


@app.get("/health", tags=["Health"])
def health():
    return {
        "status":        "ok",
        "chunks_loaded": len(chunk_store),
        "docs_loaded":   len(doc_registry),
        "model":         MODEL,
        "api_key_set":   bool(API_KEY),
    }


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".pdf", ".txt", ".md"}:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use .pdf, .txt, or .md")

    raw = await file.read()

    if ext == ".pdf":
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(raw))
            text   = "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as e:
            raise HTTPException(422, f"PDF parse error: {e}")
    else:
        text = raw.decode("utf-8", errors="replace")

    if not text.strip():
        raise HTTPException(422, "Document appears empty or unreadable.")

    new_chunks = chunk_text(text, file.filename)
    chunk_store.extend(new_chunks)
    doc_registry.append({
        "name":        file.filename,
        "num_chunks":  len(new_chunks),
        "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    return UploadResponse(
        doc_name=file.filename,
        num_chunks=len(new_chunks),
        message=f"Indexed '{file.filename}' -> {len(new_chunks)} chunks.",
    )


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
def ask(req: AskRequest):
    if not client:
        raise HTTPException(500, "GEMINI_API_KEY not set. Re-run run.bat and paste your key.")
    if not chunk_store:
        raise HTTPException(400, "No documents loaded. Upload a file first.")
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")

    t0         = time.time()
    top_chunks = retrieve_top_k(req.query, k=req.top_k)
    prompt     = build_prompt(req.query, top_chunks)
    ctx_tokens = len(prompt.split())

    log.info("Calling Gemini with %d chunks in context", len(top_chunks))
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=1024,
                temperature=0.2,
            ),
        )
        answer = response.text
    except Exception as e:
        raise HTTPException(502, f"Gemini API error: {str(e)}")

    latency_ms = int((time.time() - t0) * 1000)
    sources    = list({c["doc"] for c in top_chunks})
    top_score  = top_chunks[0]["score"] if top_chunks else 0.0

    log.info("Answered in %d ms", latency_ms)
    return AskResponse(
        answer=answer,
        sources=sources,
        chunks_retrieved=len(top_chunks),
        top_score=round(top_score, 4),
        context_tokens=ctx_tokens,
        model=MODEL,
        latency_ms=latency_ms,
    )


@app.get("/docs-list", tags=["Documents"])
def list_docs():
    return doc_registry


@app.delete("/docs-list", tags=["Documents"])
def clear_docs():
    n = len(chunk_store)
    chunk_store.clear()
    doc_registry.clear()
    return {"message": "Cleared.", "chunks_removed": n}