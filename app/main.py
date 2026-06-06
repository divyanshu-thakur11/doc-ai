"""
DocMind AI v3 — FastAPI Backend
Features:
  - Streaming responses (SSE)
  - Smart follow-up question suggestions
  - Source passages returned with answers
  - Document quiz generator
  - Multi-language support
  - Sentence-aware chunking + TF-IDF retrieval
  - Conversation memory (last 3 turns)
  - Auto-summary on upload
  - Confidence scoring
  - Per-doc enable/disable toggle
  - URL-to-document ingestion
"""

import os, math, time, re, logging, json
from collections import defaultdict
from typing import Optional, AsyncGenerator
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
import PyPDF2, io, httpx
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="DocMind AI v3", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

API_KEY = os.environ.get("GEMINI_API_KEY", "")
client  = genai.Client(api_key=API_KEY) if API_KEY else None
MODEL   = "gemini-2.5-flash"

# ── Stores ────────────────────────────────────────────────────────────────────
chunk_store:   list[dict] = []
doc_registry:  list[dict] = []
conversations: dict[str, list] = {}

# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_WORDS   = 400
OVERLAP_WORDS = 80

def sentence_chunk(text: str, doc_name: str, doc_id: str) -> list[dict]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, buf, buf_words, cid = [], [], 0, 0
    for sent in sentences:
        w = len(sent.split())
        if buf_words + w > CHUNK_WORDS and buf:
            chunks.append({"id": cid, "doc_id": doc_id, "doc": doc_name,
                           "text": " ".join(buf), "start": cid * CHUNK_WORDS})
            cid += 1
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
    return sum((cw.count(t)/len(cw)) * (math.log((total+1)/(df.get(t,0)+1))+1) for t in q_terms)

def retrieve(query: str, k: int = 6, enabled_docs: set = None) -> list[dict]:
    pool = [c for c in chunk_store if enabled_docs is None or c["doc_id"] in enabled_docs]
    if not pool: return []
    df: dict[str,int] = defaultdict(int)
    for c in pool:
        for t in set(tokenize(c["text"])): df[t] += 1
    q_terms = set(tokenize(query))
    scored = sorted([{**c, "score": tfidf_score(q_terms,c,len(pool),df)} for c in pool],
                    key=lambda x: x["score"], reverse=True)
    return scored[:k]

def confidence(top_score: float) -> str:
    if top_score > 0.05: return "high"
    if top_score > 0.01: return "medium"
    return "low"

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM = """You are DocMind AI, a helpful document Q&A assistant.

Rules:
- Always give the most useful answer possible using the provided document context.
- If the exact answer is present: answer directly and cite the source document/section.
- If related information exists but not exact: say "The document doesn't directly cover this, but here's the closest relevant information:" then provide it.
- If completely absent: briefly say so, then answer from general knowledge labeled "[General knowledge — not from documents]".
- Never refuse to answer — always provide the most helpful response possible.
- Be concise but thorough. Use markdown formatting: **bold**, bullet points, `code`, headers where helpful."""

def qa_prompt(query: str, chunks: list[dict], history: list[dict], lang: str = "English") -> str:
    ctx = "\n\n---\n\n".join(
        f'[Chunk {i+1} | "{c["doc"]}"]\n{c["text"]}' for i, c in enumerate(chunks)
    ) if chunks else "No relevant context found."
    hist_text = ""
    if history:
        hist_text = "\n\nConversation so far:\n" + "\n".join(
            f'{"User" if m["role"]=="user" else "Assistant"}: {m["content"]}' for m in history[-6:]
        )
    lang_instruction = f"\n\nIMPORTANT: Respond in {lang} language only." if lang != "English" else ""
    return f"Document context:\n\n{ctx}{hist_text}{lang_instruction}\n\n---\n\nQuestion: {query}\n\nAnswer:"

SUMMARY_PROMPT = """Read this document excerpt and write a 2-3 sentence summary:
1. What this document is about
2. Key topics or entities mentioned
Be concise and factual.\n\nDocument:\n{text}"""

FOLLOWUP_PROMPT = """Based on this Q&A exchange about a document, generate exactly 3 short follow-up questions the user might want to ask next.
Return ONLY a JSON array of 3 strings, no other text, no markdown, no explanation.
Example: ["What is X?", "How does Y work?", "Why does Z matter?"]

Question asked: {query}
Answer given: {answer}
Document topics: {doc_names}"""

QUIZ_PROMPT = """Generate exactly 5 multiple-choice questions from this document content.
Return ONLY valid JSON, no markdown, no extra text.
Format: [{{"question":"...","options":["A...","B...","C...","D..."],"answer":"A..."}}]

Document content:
{text}"""

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    query:        str
    session_id:   Optional[str]       = "default"
    top_k:        Optional[int]       = 6
    enabled_docs: Optional[list[str]] = None
    language:     Optional[str]       = "English"
    stream:       Optional[bool]      = False

class AskResponse(BaseModel):
    answer:           str
    sources:          list[str]
    source_passages:  list[dict]
    chunks_retrieved: int
    top_score:        float
    confidence:       str
    context_tokens:   int
    model:            str
    latency_ms:       int
    session_id:       str
    follow_ups:       list[str]

class UploadResponse(BaseModel):
    doc_id:     str
    doc_name:   str
    num_chunks: int
    summary:    str
    message:    str

class ToggleRequest(BaseModel):
    doc_id:  str
    enabled: bool

class UrlUploadRequest(BaseModel):
    url: str

class QuizResponse(BaseModel):
    questions: list[dict]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def index_text(text: str, filename: str) -> dict:
    doc_id     = f"doc_{int(time.time()*1000)}"
    new_chunks = sentence_chunk(text, filename, doc_id)
    chunk_store.extend(new_chunks)
    summary    = "Summary unavailable."
    if client:
        try:
            preview = " ".join(text.split()[:1500])
            r       = client.models.generate_content(
                model=MODEL, contents=SUMMARY_PROMPT.format(text=preview),
                config=types.GenerateContentConfig(max_output_tokens=200, temperature=0.3),
            )
            summary = r.text.strip()
        except Exception as e:
            log.warning("Summary failed: %s", e)
    doc_registry.append({
        "id": doc_id, "name": filename, "num_chunks": len(new_chunks),
        "uploaded_at": time.strftime("%H:%M %d %b"), "enabled": True, "summary": summary,
    })
    return {"doc_id": doc_id, "num_chunks": len(new_chunks), "summary": summary}

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
    return {"status":"ok","chunks":len(chunk_store),"docs":len(doc_registry),"model":MODEL,"api_key_set":bool(API_KEY)}

@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".pdf",".txt",".md"}:
        raise HTTPException(400, f"Unsupported: {ext}. Use .pdf .txt .md")
    raw  = await file.read()
    text = ""
    if ext == ".pdf":
        try:
            r    = PyPDF2.PdfReader(io.BytesIO(raw))
            text = "\n".join(p.extract_text() or "" for p in r.pages)
        except Exception as e:
            raise HTTPException(422, f"PDF error: {e}")
    else:
        text = raw.decode("utf-8", errors="replace")
    if not text.strip():
        raise HTTPException(422, "Document is empty or unreadable.")
    info = index_text(text, file.filename)
    return UploadResponse(doc_name=file.filename, message=f"Indexed {info['num_chunks']} chunks.", **info)

@app.post("/upload-url", response_model=UploadResponse)
async def upload_url(req: UrlUploadRequest):
    if not req.url.startswith("http"):
        raise HTTPException(400, "URL must start with http:// or https://")
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as c:
            resp = await c.get(req.url, headers={"User-Agent": "DocMindAI/3.0"})
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(422, f"Failed to fetch URL: {e}")
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script","style","nav","footer","header","aside"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) < 100:
        raise HTTPException(422, "Page has too little readable text.")
    from urllib.parse import urlparse
    filename = urlparse(req.url).netloc + urlparse(req.url).path.rstrip("/").split("/")[-1] + ".html"
    info = index_text(text, filename)
    return UploadResponse(doc_name=filename, message=f"Fetched & indexed {info['num_chunks']} chunks.", **info)

@app.post("/toggle")
def toggle_doc(req: ToggleRequest):
    for doc in doc_registry:
        if doc["id"] == req.doc_id:
            doc["enabled"] = req.enabled
            return {"doc_id": req.doc_id, "enabled": req.enabled}
    raise HTTPException(404, "Document not found.")

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not client:   raise HTTPException(500, "GEMINI_API_KEY not set.")
    if not chunk_store: raise HTTPException(400, "No documents loaded.")
    if not req.query.strip(): raise HTTPException(400, "Empty query.")

    enabled    = set(req.enabled_docs) if req.enabled_docs else {d["id"] for d in doc_registry if d["enabled"]}
    t0         = time.time()
    top_chunks = retrieve(req.query, k=req.top_k, enabled_docs=enabled)
    history    = conversations.get(req.session_id, [])
    prompt     = qa_prompt(req.query, top_chunks, history, req.language or "English")
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

    # Save history
    if req.session_id not in conversations:
        conversations[req.session_id] = []
    conversations[req.session_id] += [{"role":"user","content":req.query},{"role":"assistant","content":answer}]
    conversations[req.session_id]   = conversations[req.session_id][-20:]

    # Follow-up suggestions
    follow_ups = []
    try:
        doc_names = ", ".join({c["doc"] for c in top_chunks}) or "unknown"
        fu_resp   = client.models.generate_content(
            model=MODEL,
            contents=FOLLOWUP_PROMPT.format(query=req.query, answer=answer[:500], doc_names=doc_names),
            config=types.GenerateContentConfig(max_output_tokens=200, temperature=0.7),
        )
        raw = fu_resp.text.strip()
        raw = re.sub(r'^```json|^```|```$','',raw,flags=re.MULTILINE).strip()
        follow_ups = json.loads(raw)
        if not isinstance(follow_ups, list): follow_ups = []
        follow_ups = [str(q) for q in follow_ups[:3]]
    except Exception as e:
        log.warning("Follow-up generation failed: %s", e)

    top_score = top_chunks[0]["score"] if top_chunks else 0.0
    source_passages = [{"doc": c["doc"], "text": c["text"][:300]+"…"} for c in top_chunks[:3]]

    return AskResponse(
        answer=answer, sources=list({c["doc"] for c in top_chunks}),
        source_passages=source_passages,
        chunks_retrieved=len(top_chunks), top_score=round(top_score,4),
        confidence=confidence(top_score), context_tokens=ctx_tokens,
        model=MODEL, latency_ms=int((time.time()-t0)*1000),
        session_id=req.session_id, follow_ups=follow_ups,
    )

@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """SSE streaming endpoint — yields tokens as they arrive."""
    if not client:   raise HTTPException(500, "GEMINI_API_KEY not set.")
    if not chunk_store: raise HTTPException(400, "No documents loaded.")
    if not req.query.strip(): raise HTTPException(400, "Empty query.")

    enabled    = set(req.enabled_docs) if req.enabled_docs else {d["id"] for d in doc_registry if d["enabled"]}
    top_chunks = retrieve(req.query, k=req.top_k, enabled_docs=enabled)
    history    = conversations.get(req.session_id, [])
    prompt     = qa_prompt(req.query, top_chunks, history, req.language or "English")

    async def event_generator() -> AsyncGenerator[str, None]:
        full_answer = ""
        try:
            for chunk in client.models.generate_content_stream(
                model=MODEL, contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM, max_output_tokens=1200, temperature=0.2),
            ):
                if chunk.text:
                    full_answer += chunk.text
                    yield f"data: {json.dumps({'type':'token','text':chunk.text})}\n\n"

            # Save to history
            if req.session_id not in conversations:
                conversations[req.session_id] = []
            conversations[req.session_id] += [
                {"role":"user","content":req.query},
                {"role":"assistant","content":full_answer}
            ]
            conversations[req.session_id] = conversations[req.session_id][-20:]

            # Follow-ups
            follow_ups = []
            try:
                doc_names = ", ".join({c["doc"] for c in top_chunks}) or "unknown"
                fu_resp   = client.models.generate_content(
                    model=MODEL,
                    contents=FOLLOWUP_PROMPT.format(query=req.query, answer=full_answer[:500], doc_names=doc_names),
                    config=types.GenerateContentConfig(max_output_tokens=200, temperature=0.7),
                )
                raw = fu_resp.text.strip()
                raw = re.sub(r'^```json|^```|```$','',raw,flags=re.MULTILINE).strip()
                follow_ups = json.loads(raw)
                if not isinstance(follow_ups, list): follow_ups = []
                follow_ups = [str(q) for q in follow_ups[:3]]
            except Exception as e:
                log.warning("Follow-up generation failed: %s", e)

            top_score = top_chunks[0]["score"] if top_chunks else 0.0
            source_passages = [{"doc": c["doc"], "text": c["text"][:300]+"…"} for c in top_chunks[:3]]

            yield f"data: {json.dumps({'type':'done','sources':list({c['doc'] for c in top_chunks}),'source_passages':source_passages,'confidence':confidence(top_score),'follow_ups':follow_ups,'latency_ms':0})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type':'error','text':str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.post("/quiz", response_model=QuizResponse)
def generate_quiz(body: dict = None):
    if not client:      raise HTTPException(500, "GEMINI_API_KEY not set.")
    if not chunk_store: raise HTTPException(400, "No documents loaded.")
    sample = " ".join(c["text"] for c in chunk_store[:8])[:4000]
    try:
        resp = client.models.generate_content(
            model=MODEL, contents=QUIZ_PROMPT.format(text=sample),
            config=types.GenerateContentConfig(max_output_tokens=1200, temperature=0.5),
        )
        raw = resp.text.strip()
        raw = re.sub(r'^```json|^```|```$','',raw,flags=re.MULTILINE).strip()
        questions = json.loads(raw)
        if not isinstance(questions, list): raise ValueError("Not a list")
        return QuizResponse(questions=questions[:5])
    except Exception as e:
        raise HTTPException(502, f"Quiz generation failed: {e}")

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