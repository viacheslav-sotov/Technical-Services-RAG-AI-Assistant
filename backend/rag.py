"""
The RAG pipeline: retrieve relevant chunks, build a grounded prompt, generate an
answer, and return citations.

RAG concept (the whole flow)
----------------------------
    question
      -> embed question
      -> retrieve top-K nearest chunks from the vector store
      -> filter out weak matches (below MIN_RELEVANCE_SCORE)
      -> if nothing relevant: REFUSE (don't hallucinate)
      -> else: stuff the chunks into a system prompt as "context"
      -> ask the LLM to answer ONLY from that context, with citations

This grounding + refusal behavior is what makes the assistant trustworthy for
AV/event staff who rely on accurate equipment instructions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from backend.config import settings
from backend.embeddings import get_embedder
from backend.llm_provider import get_llm
from backend.utils import logger
from backend.vector_store import get_vector_store

REFUSAL_MESSAGE = "I could not find this in the Technical Services documents."

# The system prompt is the contract with the LLM. It is deliberately strict.
SYSTEM_PROMPT_TEMPLATE = """You are the Technical Services Assistant for AV and event staff.
Answer questions about AV equipment, rooms, troubleshooting steps, manuals, and \
internal documentation.

STRICT RULES:
1. Answer ONLY using the information in the CONTEXT below. Do not use outside \
knowledge and never invent equipment steps, model numbers, settings, or cable types.
2. If the CONTEXT does not contain the answer, reply with EXACTLY:
   "{refusal}"
3. Cite your sources inline using bracketed numbers like [1], [2] that correspond \
to the numbered context sources.
4. Be clear, practical, and concise. Prefer short numbered steps for troubleshooting.
5. If steps could be unsafe or depend on equipment not described, say so.

CONTEXT:
{context}
"""


@dataclass
class Citation:
    index: int
    filename: str
    page: int  # -1 means "no page" (txt/docx/md)
    score: float
    preview: str
    document_id: str

    def as_dict(self) -> Dict:
        return {
            "index": self.index,
            "filename": self.filename,
            "page": None if self.page == -1 else self.page,
            "score": round(self.score, 4),
            "preview": self.preview,
            "document_id": self.document_id,
        }


@dataclass
class RAGResponse:
    answer: str
    citations: List[Citation] = field(default_factory=list)
    used_context: bool = False
    provider: str = ""
    model: str = ""

    def as_dict(self) -> Dict:
        return {
            "answer": self.answer,
            "citations": [c.as_dict() for c in self.citations],
            "used_context": self.used_context,
            "provider": self.provider,
            "model": self.model,
        }


def _preview(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def retrieve(question: str, top_k: Optional[int] = None) -> List[Dict]:
    """Embed the question and return relevant chunks above the score threshold."""
    top_k = top_k or settings.RETRIEVAL_TOP_K
    embedder = get_embedder()
    query_vec = embedder.embed_query(question)
    store = get_vector_store()
    hits = store.query(query_vec, top_k=top_k)
    # Keep only sufficiently relevant chunks -> this powers the refusal behavior.
    relevant = [h for h in hits if h["score"] >= settings.MIN_RELEVANCE_SCORE]
    logger.info(
        "Retrieval: %d hits, %d above threshold %.2f",
        len(hits),
        len(relevant),
        settings.MIN_RELEVANCE_SCORE,
    )
    return relevant


def _build_context(hits: List[Dict]) -> str:
    """Render retrieved chunks into a numbered context block for the prompt."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        page = meta.get("page", -1)
        loc = f"{meta.get('filename', 'unknown')}"
        if page and page != -1:
            loc += f", page {page}"
        blocks.append(f"[{i}] (source: {loc})\n{hit['text']}")
    return "\n\n".join(blocks)


def answer_question(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> RAGResponse:
    """
    Main entry point used by the API.

    `history` is the prior conversation as [{"role","content"}, ...] and is passed
    to the LLM so follow-up questions keep context. Retrieval, however, is based on
    the current question.
    """
    history = history or []
    question = (question or "").strip()
    if not question:
        return RAGResponse(answer=REFUSAL_MESSAGE, used_context=False)

    llm = get_llm()
    hits = retrieve(question)

    # No relevant context -> refuse instead of hallucinating.
    if not hits:
        logger.info("No relevant context found; returning refusal.")
        return RAGResponse(
            answer=REFUSAL_MESSAGE,
            citations=[],
            used_context=False,
            provider=llm.name,
            model=llm.model,
        )

    citations = [
        Citation(
            index=i,
            filename=h["metadata"].get("filename", "unknown"),
            page=int(h["metadata"].get("page", -1)),
            score=h["score"],
            preview=_preview(h["text"]),
            document_id=h["metadata"].get("document_id", ""),
        )
        for i, h in enumerate(hits, start=1)
    ]

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        refusal=REFUSAL_MESSAGE, context=_build_context(hits)
    )

    # Cap history so the prompt stays within budget (last few turns are enough).
    trimmed_history = history[-6:]
    messages = [*trimmed_history, {"role": "user", "content": question}]

    answer = llm.generate(system_prompt, messages)
    if not answer:
        answer = REFUSAL_MESSAGE

    return RAGResponse(
        answer=answer,
        citations=citations,
        used_context=True,
        provider=llm.name,
        model=llm.model,
    )
