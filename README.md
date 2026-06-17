# 🛠️ Technical Services RAG Assistant

An AI-powered, **production-ready** Retrieval-Augmented Generation (RAG) assistant that helps
Technical Services / AV / event staff get fast, **source-grounded** answers about equipment,
rooms, troubleshooting steps, manuals, and internal documentation.

The assistant **only answers from your uploaded documents**, always shows **citations**
(filename + page + preview), and **refuses to guess** when the answer isn't in the docs —
so staff never get made-up equipment instructions.

```
Question ─▶ embed ─▶ retrieve top-K chunks ─▶ relevant?
                                                 │  no ─▶ "I could not find this in the Technical Services documents."
                                                 │  yes ─▶ build grounded prompt ─▶ LLM ─▶ answer + citations
```

---

## ✨ Features

- 📤 Upload **PDF, TXT, DOCX, Markdown** documents
- ✂️ Automatic parsing + overlapping chunking (page numbers preserved for PDFs)
- 🧠 Embeddings via **SentenceTransformers** (local, free) or **OpenAI** (optional, paid)
- 🗄️ Vector storage in **ChromaDB** (local) behind a **pluggable interface** (swap in Pinecone/Supabase later)
- 💬 Chat UI with **conversation history** for the session
- 📎 **Source citations**: filename, page number (if available), relevance score, chunk preview
- 🚫 **Refuses** when context is insufficient (no hallucinated instructions)
- 🤖 Modular LLM provider: **OpenAI** (if key present) or **Ollama** (local, free)
- 🔐 Admin page: uploaded files, chunk counts, embedding model, LLM provider, vector DB status
- 🛡️ Production hardening: env-based config, rate limiting, admin auth, input validation,
  error handling, logging, health check, CORS, Docker, tests

---

## 🧱 Tech stack

| Layer        | Choice                                                    |
|--------------|-----------------------------------------------------------|
| Backend      | Python + **FastAPI**                                      |
| Frontend     | **Streamlit** (simplest option first)                    |
| Vector DB    | **ChromaDB** (local); pluggable for Pinecone/Supabase     |
| Embeddings   | **SentenceTransformers** (default) / OpenAI (optional)    |
| LLM          | **Ollama** (default/free) / OpenAI (optional/paid)        |
| Rate limit   | **slowapi**                                               |
| Container    | **Docker** + docker-compose                               |

---

## 📂 Project structure & what each file does

```
Technical-Services-RAG-AI-Assistant/
├── backend/
│   ├── main.py           # FastAPI app: all routes, CORS, rate-limit & error wiring
│   ├── config.py         # Typed settings loaded from env/.env (pydantic-settings)
│   ├── utils.py          # Logging setup, ID/hashing/file helpers, allowed types
│   ├── vector_store.py   # BaseVectorStore interface + ChromaDB implementation
│   ├── embeddings.py     # Modular embedders: SentenceTransformers / OpenAI
│   ├── llm_provider.py   # Modular LLMs: OpenAI / Ollama (HTTP)
│   ├── ingest.py         # Parse (PDF/DOCX/TXT/MD) → chunk → embed → store
│   ├── rag.py            # Retrieve → grounded prompt → generate → citations/refusal
│   ├── auth.py           # HTTP Basic auth dependency for admin routes
│   └── rate_limit.py     # slowapi limiter (per-IP)
├── frontend/
│   └── app.py            # Streamlit Chat + Admin UI (thin client over the API)
├── tests/
│   ├── conftest.py             # Isolated temp store + fixtures
│   ├── test_chunking.py        # Pure chunking logic
│   ├── test_api.py             # Health, validation, auth, refusal flow
│   └── test_ingest_retrieval.py# Ingestion + retrieval + grounded answering
├── sample_docs/          # Example doc you can upload to try it out
├── data/                 # Uploaded files (git-ignored)
├── chroma_db/            # Persisted vectors (git-ignored)
├── .env.example          # Copy to .env and fill in
├── Dockerfile            # One image, runs API or frontend via SERVICE env
├── docker-entrypoint.sh  # Chooses api vs frontend at container start
├── docker-compose.yml    # Local API + frontend (+ optional Ollama)
├── render.yaml           # Render.com blueprint
├── fly.toml              # Fly.io config
├── Procfile              # Railway/Heroku-style start command
├── requirements.txt
└── README.md
```

---

## ⚙️ Prerequisites

- **Python 3.11 or 3.12** for local (non-Docker) runs.
  > ⚠️ Some dependencies (chromadb, tokenizers, pydantic-core) don't yet publish prebuilt
  > wheels for Python 3.13/3.14, so a local `pip install` on those versions tries to compile
  > from source and fails without a Rust/C++ toolchain. **Use 3.11/3.12 locally, or just use Docker** (which pins 3.11).
- (Optional) **Ollama** for free local LLM inference — https://ollama.com
- (Optional) **OpenAI API key** if you prefer the hosted model

---

## 🔑 Where to add your API keys

All secrets live in a **`.env`** file (never in code, never committed).

1. Copy the template:
   ```bash
   cp .env.example .env            # macOS/Linux
   Copy-Item .env.example .env     # Windows PowerShell
   ```
2. Edit `.env`:
   - **OpenAI (optional, paid):** set `OPENAI_API_KEY=sk-...`. The app auto-switches to OpenAI
     when a key is present (`LLM_PROVIDER=auto`). To force it: `LLM_PROVIDER=openai`.
   - **Ollama (default, free):** leave `OPENAI_API_KEY` empty and set
     `OLLAMA_MODEL=qwen2.5:3b` (or any model you've pulled).
   - **Admin login:** change `ADMIN_USERNAME` / `ADMIN_PASSWORD` (protects admin + delete routes).
   - **Embeddings:** default is local `sentence_transformers`; set `EMBEDDING_PROVIDER=openai` to use OpenAI.

> 🔒 In production (Render/Railway/Fly), set these as **dashboard secrets** instead of committing `.env`.

---

## ▶️ Run it locally

### Option A — Free, fully local (SentenceTransformers + Ollama)

```bash
# 1. Create a 3.11/3.12 virtual env and install deps
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configure
cp .env.example .env                 # leave OPENAI_API_KEY empty for the free path

# 3. (Free LLM) install + start Ollama, then pull a small model
ollama pull qwen2.5:3b               # ~2 GB; or try llama3.2:3b

# 4. Start the backend API (terminal 1)
uvicorn backend.main:app --reload --port 8000
#    -> API docs at http://localhost:8000/docs
#    -> health at  http://localhost:8000/health

# 5. Start the frontend (terminal 2)
streamlit run frontend/app.py
#    -> UI at http://localhost:8501
```

Then open the **Admin** page → upload `sample_docs/room204_projector.md` → go to **Chat** and ask:
> *"How do I connect a laptop to the projector in Room 204?"*

### Option B — OpenAI (paid)

Set `OPENAI_API_KEY=sk-...` in `.env`, then run steps 4–5 above (skip Ollama).

> 💡 First run downloads the local embedding model (~80 MB). Subsequent runs are fast.

---

## 🧪 How to test it

```bash
pip install -r requirements.txt      # on Python 3.11/3.12
pytest                               # runs all tests
pytest tests/test_api.py -v          # just the API tests
```

What the tests cover:
- **`test_chunking.py`** — chunk sizing/overlap edge cases (no models needed).
- **`test_api.py`** — `/health`, input validation (422), admin auth (401/200), unsupported file
  rejection, and the **refusal** behavior on an empty knowledge base.
- **`test_ingest_retrieval.py`** — real ingestion → retrieval → grounded answer with citations
  (uses a fake LLM so it's deterministic). These auto-skip if the embedding model can't be
  downloaded (e.g. offline CI).

Manual API smoke test (with the server running):
```bash
curl http://localhost:8000/health

# Upload + ingest
UP=$(curl -s -F "file=@sample_docs/room204_projector.md" http://localhost:8000/upload | python -c "import sys,json;print(json.load(sys.stdin)['upload_id'])")
curl -s -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d "{\"upload_id\":\"$UP\"}"

# Ask a question
curl -s -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"question":"How do I connect a laptop in Room 204?"}'

# Admin stats (Basic auth)
curl -s -u admin:change-me-please http://localhost:8000/admin/stats
```

---

## 🐳 Run with Docker

```bash
cp .env.example .env        # configure first

# Build & run API + frontend together
docker compose up --build
#  Frontend: http://localhost:8501
#  API docs: http://localhost:8000/docs
```

Single container (API only):
```bash
docker build -t tsrag .
docker run -p 8000:8000 --env-file .env \
  -v "$(pwd)/chroma_db:/app/chroma_db" -v "$(pwd)/data:/app/data" \
  -e SERVICE=api tsrag
```
Run the frontend image by setting `-e SERVICE=frontend -e PORT=8501 -e BACKEND_URL=http://host:8000`.

> To use Ollama with compose, uncomment the `ollama` service in `docker-compose.yml`
> and set `OLLAMA_BASE_URL=http://ollama:11434` in `.env`.

---

## 🚀 Deploy

The same `Dockerfile` deploys to most container platforms. Set secrets in the platform
dashboard (`OPENAI_API_KEY`, `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `CORS_ORIGINS`).

### Render (blueprint included)
1. Push this repo to GitHub.
2. Render → **New → Blueprint** → pick the repo (`render.yaml` is detected).
3. It provisions `tsrag-api` (with a 1 GB disk for the vector store) and `tsrag-frontend`.
4. Add secrets under **Environment**. Done — health checks hit `/health`.

### Railway
1. New Project → Deploy from GitHub repo (auto-detects the `Dockerfile`).
2. Add variables (`OPENAI_API_KEY`, `ADMIN_PASSWORD`, …). Railway injects `$PORT`.
3. Add a **Volume** mounted at `/app/chroma_db` to persist vectors.

### Fly.io (`fly.toml` included)
```bash
fly launch --no-deploy        # uses the Dockerfile
fly volumes create tsrag_data --size 1
fly secrets set OPENAI_API_KEY=sk-... ADMIN_PASSWORD=...
fly deploy
```

> **Note on hosting + LLM:** managed hosts usually can't run Ollama. In production, use
> **OpenAI** (set `OPENAI_API_KEY`) or point `OLLAMA_BASE_URL` at a separately-hosted Ollama.
> Embeddings stay free/local via SentenceTransformers either way.

---

## 🔌 API reference

| Method & path                 | Auth   | Description                                  |
|-------------------------------|--------|----------------------------------------------|
| `GET /health`                 | —      | Liveness + component status                  |
| `POST /upload`                | —      | Upload a file → returns `upload_id`          |
| `POST /ingest`                | —      | Parse/chunk/embed/store an uploaded file     |
| `POST /chat`                  | —      | Ask a question → answer + citations          |
| `GET /documents`              | —      | List ingested documents + chunk counts       |
| `DELETE /documents/{id}`      | admin  | Delete a document's chunks                   |
| `GET /admin/stats`            | admin  | Full system status (admin page data)         |

Interactive docs: **`http://localhost:8000/docs`**.

---

## 🔁 Switching providers (modularity)

- **LLM:** set `LLM_PROVIDER=openai|ollama|auto`. Add a new provider by subclassing
  `BaseLLM` in `backend/llm_provider.py` and registering it in `get_llm()`.
- **Embeddings:** set `EMBEDDING_PROVIDER=sentence_transformers|openai`. Add one by
  subclassing `BaseEmbedder` in `backend/embeddings.py`.
- **Vector DB:** implement `BaseVectorStore` in `backend/vector_store.py` (e.g. `PineconeVectorStore`)
  and return it from `get_vector_store()` based on `VECTOR_BACKEND`.

---

## 🧠 Key RAG concepts (this is also a portfolio project)

- **Chunking with overlap** keeps passages focused while preserving context across boundaries.
- **Embeddings** turn text into vectors so semantically similar text sits close together.
- **Cosine similarity + a minimum score threshold** decide whether retrieved context is
  good enough — this is what powers the **refuse-don't-hallucinate** behavior.
- **Grounded prompting**: the LLM is instructed to answer *only* from retrieved context and to
  cite numbered sources, which keeps answers verifiable.

---

## 📜 License

Provided as-is for internal Technical Services use and as a portfolio reference.
