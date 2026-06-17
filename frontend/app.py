"""
Streamlit frontend for the Technical Services RAG Assistant.

Two pages:
  * Chat   -> ask questions; answers show source citations; history kept in-session.
  * Admin  -> upload/ingest documents, view stats, delete documents (Basic-Auth).

The frontend is a thin client over the FastAPI backend. It holds NO secrets;
it only talks to the API at BACKEND_URL. Admin credentials are entered by the
user at runtime and sent as HTTP Basic auth to protected endpoints.

Run:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 180  # local LLMs (Ollama) can be slow on first token

st.set_page_config(
    page_title="Technical Services RAG Assistant",
    page_icon="🛠️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Small API helpers
# ---------------------------------------------------------------------------

def api_get(path: str, auth=None):
    return requests.get(f"{BACKEND_URL}{path}", auth=auth, timeout=REQUEST_TIMEOUT)


def api_post(path: str, json=None, files=None, auth=None):
    return requests.post(
        f"{BACKEND_URL}{path}", json=json, files=files, auth=auth, timeout=REQUEST_TIMEOUT
    )


def api_delete(path: str, auth=None):
    return requests.delete(f"{BACKEND_URL}{path}", auth=auth, timeout=REQUEST_TIMEOUT)


def backend_online() -> bool:
    try:
        return api_get("/health").ok
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    # Conversation history for the current browser session.
    st.session_state.messages = []  # list of {"role","content","citations"}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🛠️ Tech Services RAG")
    page = st.radio("Navigate", ["💬 Chat", "🗂️ Admin"], label_visibility="collapsed")

    st.divider()
    online = backend_online()
    st.markdown(f"**Backend:** {'🟢 online' if online else '🔴 offline'}")
    st.caption(BACKEND_URL)
    if not online:
        st.error("Cannot reach the backend API. Start it with:\n\n`uvicorn backend.main:app --port 8000`")

    if page == "💬 Chat":
        st.divider()
        if st.button("🧹 Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ---------------------------------------------------------------------------
# Chat page
# ---------------------------------------------------------------------------

def render_citations(citations):
    if not citations:
        return
    with st.expander(f"📎 Sources ({len(citations)})"):
        for c in citations:
            page = c.get("page")
            loc = c["filename"] + (f" — page {page}" if page else "")
            st.markdown(f"**[{c['index']}] {loc}**  ·  relevance `{c['score']}`")
            st.caption(c["preview"])


def chat_page():
    st.header("💬 Ask the Technical Services Assistant")
    st.caption(
        "Answers are grounded in your uploaded documents and include citations. "
        "If the answer isn't in the docs, the assistant will say so."
    )

    # Replay the conversation.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_citations(msg.get("citations"))

    prompt = st.chat_input("e.g. How do I connect a laptop to the projector in Room 204?")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            # Send prior turns (without citation payloads) so follow-ups keep context.
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            try:
                resp = api_post("/chat", json={"question": prompt, "history": history})
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
                return

        if resp.status_code == 429:
            st.warning("You're sending requests too quickly. Please wait a moment.")
            return
        if not resp.ok:
            st.error(f"Error {resp.status_code}: {resp.json().get('detail', resp.text)}")
            return

        data = resp.json()
        answer = data.get("answer", "")
        citations = data.get("citations", [])
        st.markdown(answer)
        render_citations(citations)
        if data.get("provider"):
            st.caption(f"Generated by {data['provider']} · {data.get('model','')}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "citations": citations}
    )


# ---------------------------------------------------------------------------
# Admin page
# ---------------------------------------------------------------------------

def admin_page():
    st.header("🗂️ Admin")
    st.caption("Upload documents and inspect system status. Admin actions require credentials.")

    with st.expander("🔐 Admin credentials (needed for stats & delete)", expanded=True):
        col1, col2 = st.columns(2)
        admin_user = col1.text_input("Username", value=st.session_state.get("admin_user", ""))
        admin_pass = col2.text_input("Password", type="password", value=st.session_state.get("admin_pass", ""))
        st.session_state.admin_user = admin_user
        st.session_state.admin_pass = admin_pass
    auth = (admin_user, admin_pass) if admin_user and admin_pass else None

    # ---- Upload + ingest ----
    st.subheader("📤 Upload & index a document")
    uploaded = st.file_uploader(
        "PDF, TXT, DOCX, or Markdown",
        type=["pdf", "txt", "docx", "md", "markdown"],
    )
    if uploaded is not None and st.button("Upload & index", type="primary"):
        with st.spinner("Uploading..."):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            up = api_post("/upload", files=files)
        if not up.ok:
            st.error(f"Upload failed: {up.json().get('detail', up.text)}")
        else:
            upload_id = up.json()["upload_id"]
            with st.spinner("Parsing, chunking, and embedding (first run downloads the model)..."):
                ing = api_post("/ingest", json={"upload_id": upload_id})
            if not ing.ok:
                st.error(f"Ingestion failed: {ing.json().get('detail', ing.text)}")
            else:
                r = ing.json()
                st.success(
                    f"Indexed **{r['filename']}** — {r['chunk_count']} chunks"
                    + (f", {r['pages']} pages" if r.get("pages") else "")
                )

    st.divider()

    # ---- Documents list ----
    st.subheader("📚 Indexed documents")
    try:
        docs_resp = api_get("/documents")
        docs = docs_resp.json().get("documents", []) if docs_resp.ok else []
    except requests.RequestException as exc:
        st.error(f"Could not load documents: {exc}")
        docs = []

    if not docs:
        st.info("No documents indexed yet.")
    else:
        for d in docs:
            c1, c2, c3 = st.columns([5, 2, 2])
            c1.markdown(f"**{d['filename']}**  \n`{d['document_id']}`")
            c2.metric("Chunks", d.get("chunk_count", 0))
            if c3.button("🗑️ Delete", key=f"del_{d['document_id']}"):
                if not auth:
                    st.warning("Enter admin credentials above to delete.")
                else:
                    dr = api_delete(f"/documents/{d['document_id']}", auth=auth)
                    if dr.ok:
                        st.success(f"Deleted {d['filename']}.")
                        st.rerun()
                    elif dr.status_code == 401:
                        st.error("Invalid admin credentials.")
                    else:
                        st.error(f"Delete failed: {dr.text}")

    st.divider()

    # ---- System stats ----
    st.subheader("📊 System status")
    if not auth:
        st.info("Enter admin credentials above to view detailed stats.")
        return
    stats_resp = api_get("/admin/stats", auth=auth)
    if stats_resp.status_code == 401:
        st.error("Invalid admin credentials.")
        return
    if not stats_resp.ok:
        st.error(f"Could not load stats: {stats_resp.text}")
        return

    s = stats_resp.json()
    m1, m2, m3 = st.columns(3)
    m1.metric("Documents", s.get("document_count", 0))
    m2.metric("Total chunks", s.get("total_chunks", 0))
    m3.metric("Pending uploads", s.get("pending_uploads", 0))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Embedding**")
        st.json(s.get("embedding", {}))
        st.markdown("**Retrieval settings**")
        st.json(s.get("retrieval", {}))
    with c2:
        st.markdown("**LLM provider**")
        st.json(s.get("llm", {}))
        st.markdown("**Vector store**")
        st.json(s.get("vector_store", {}))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if page == "💬 Chat":
    chat_page()
else:
    admin_page()
