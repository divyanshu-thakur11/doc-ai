# DocMind AI — RAG PDF Q&A System

> Ask questions about any PDF or text document using Claude (Anthropic API) + a custom TF-IDF retrieval pipeline.

---

## Architecture

```
User uploads PDF/TXT
        │
        ▼
┌──────────────────────────────────────────────┐
│              FastAPI Backend                 │
│                                              │
│  POST /upload                                │
│    → Parse PDF (PyPDF2) or read TXT          │
│    → Split into 500-word chunks (20% overlap)│
│    → Store in in-memory chunk list           │
│                                              │
│  POST /ask                                   │
│    → Tokenize query (remove stopwords)       │
│    → Score all chunks via TF-IDF             │
│    → Select top-5 most relevant chunks       │
│    → Build prompt with injected context      │
│    → Call Claude (claude-sonnet-4)           │
│    → Return: answer + sources + debug info   │
└──────────────────────────────────────────────┘
        │
        ▼
 Browser Frontend (frontend/index.html)
  — animated RAG pipeline visualization
  — source chip citations
  — per-query debug metadata panel
```

---

## Quick Start

### 1. Install dependencies

```bash
cd docmind-ai
python -m venv venv

# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Add your API key

```bash
cp .env.example .env
# Open .env and replace sk-ant-your-key-here with your real key
# Get a key at: https://console.anthropic.com/
```

### 3. Start the server

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here   # Mac/Linux
set ANTHROPIC_API_KEY=sk-ant-your-key-here       # Windows

uvicorn app.main:app --reload --port 8000
```

### 4. Open the app

```
http://localhost:8000
```

Or use the interactive API docs:
```
http://localhost:8000/docs
```

---

## API Reference

### `POST /upload`
Upload a PDF, TXT, or MD file.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf"
```

```json
{
  "doc_name": "your_document.pdf",
  "num_chunks": 14,
  "message": "Indexed 'your_document.pdf' → 14 chunks."
}
```

### `POST /ask`
Ask a question against all loaded documents.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main findings?", "top_k": 5}'
```

```json
{
  "answer": "The main findings are...",
  "sources": ["your_document.pdf"],
  "chunks_retrieved": 4,
  "top_score": 0.0312,
  "context_tokens": 847,
  "model": "claude-sonnet-4-20250514",
  "latency_ms": 1243
}
```

### `GET /docs-list` — list loaded documents
### `DELETE /docs-list` — clear all documents
### `GET /health` — server status

---

## Prompt Engineering Notes

### Grounding system prompt
```
"Answer ONLY from the provided document chunks.
 If the answer is not present, say: 'This information is not in the loaded documents.'
 Never invent facts not present in the context."
```
This prevents hallucination by giving Claude a clear fallback instruction.

### Context injection format
Each chunk is labeled with source and position:
```
[Chunk 1 from "report.pdf" — words 0–500]
<chunk text>

---

[Chunk 2 from "report.pdf" — words 400–900]
<chunk text>
```
Labeling enables natural source citations in the answer.

### Chunking strategy
- **500-word windows** — large enough for full paragraphs
- **20% overlap** — prevents answers from falling across chunk boundaries
- **Word-level splitting** — simple and fast, no external NLP library needed

### Why TF-IDF (not embeddings)?
- Zero extra dependencies — no vector DB, no embedding model needed
- Scores are explainable (visible in the debug panel)
- Fast enough for documents under ~200 pages
- For production at scale: replace with ChromaDB + sentence-transformers

---

## Debug Metadata
Every `/ask` response returns:
- `chunks_retrieved` — how many chunks matched
- `top_score` — TF-IDF score of the best chunk (low score = vocabulary mismatch)
- `context_tokens` — approximate tokens sent to Claude
- `latency_ms` — total response time

---

## Tech Stack

| Layer     | Technology               |
|-----------|--------------------------|
| API       | FastAPI + Uvicorn        |
| LLM       | Claude (Anthropic SDK)   |
| Retrieval | Custom TF-IDF            |
| PDF Parse | PyPDF2                   |
| Frontend  | Vanilla HTML/CSS/JS      |
| Validation| Pydantic v2              |

---

## Project Structure

```
docmind-ai/
├── app/
│   └── main.py          ← FastAPI backend (all logic)
├── frontend/
│   └── index.html       ← Browser UI (served at GET /)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
