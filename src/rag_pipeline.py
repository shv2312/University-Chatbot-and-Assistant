"""Grounded Retrieval-Augmented Generation (RAG) pipeline.

Retrieves relevant university policy chunks from ChromaDB, applies semantic
distance threshold filtering, constructs source-grounded context, and queries
Gemini 2.5 Flash with strict anti-hallucination guardrails and offline fallback synthesis.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

SYSTEM_INSTRUCTION = (
    "You are the Intelligent University Knowledge Assistant. Answer the question using ONLY the "
    "provided document context. If the answer is not present in the context, respond exactly: "
    "'I couldn't find enough information about this in the uploaded university documents.' "
    "Never speculate, extrapolate, or invent regulations, dates, fees, or policies. "
    "Cite the exact document and page number in your answer."
)

FALLBACK_ANSWER = "I couldn't find enough information about this in the uploaded university documents."


def answer_query(
    query: str,
    vector_store: Any,
    client: Optional[Any] = None,
    top_k: int = 4,
    distance_threshold: float = 0.65,
    mock_mode: bool = False,
) -> Dict[str, Any]:
    """Process a user query through the grounded RAG pipeline.

    Args:
        query: Student or faculty query.
        vector_store: VectorStore instance containing indexed documents.
        client: Optional genai.Client instance for testing or custom auth.
        top_k: Number of chunks to retrieve (default: 4).
        distance_threshold: Maximum cosine distance threshold to accept (default: 0.65).
                            Chunks with distance > distance_threshold are filtered out.
        mock_mode: If True, uses offline grounded text synthesis without calling Gemini API.

    Returns:
        Structured dictionary:
        {
            "answer": str,
            "sources": [{"filename": str, "page": int, "text": str}]
        }
    """
    if not query or not query.strip():
        return {
            "answer": FALLBACK_ANSWER,
            "sources": [],
        }

    # Retrieve top_k candidate chunks from vector store
    retrieved_chunks = vector_store.search(query.strip(), top_k=top_k)

    # Filter retrieved chunks by semantic distance threshold
    valid_chunks = [
        c for c in retrieved_chunks if c.get("distance", 1.0) <= distance_threshold
    ]

    # Guardrail: If no chunks survive threshold or collection was empty
    if not valid_chunks:
        return {
            "answer": FALLBACK_ANSWER,
            "sources": [],
        }

    # Build context string with explicit document references from valid chunks
    context_blocks = []
    sources = []
    for chunk in valid_chunks:
        text = chunk.get("text", "")
        meta = chunk.get("metadata", {})
        filename = meta.get("filename", "Unknown Document")
        page = meta.get("page", 1)

        context_blocks.append(f"[Source: {filename}, Page: {page}] {text}")
        sources.append(
            {
                "filename": filename,
                "page": page,
                "text": text[:120],
            }
        )

    # Offline / Mock Mode: Synthesize grounded response directly from top chunk
    if mock_mode:
        top_text = valid_chunks[0]["text"][:300].strip()
        answer_text = f"Based on the uploaded university documents:\n\n{top_text}..."
        return {
            "answer": answer_text,
            "sources": sources,
        }

    context_str = "\n\n".join(context_blocks)
    prompt = f"Document Context:\n{context_str}\n\nUser Question:\n{query.strip()}"

    candidate_models = ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-2.0-flash", "gemini-1.0-pro"]
    answer_text = None

    # Lazy-load genai Client dynamically if not passed
    try:
        if client is None:
            from src.vector_store import get_genai_client

            client = get_genai_client()

        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.0,
        )

        for model_name in candidate_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                if hasattr(response, "text") and response.text:
                    answer_text = response.text.strip()
                    break
            except Exception as e:
                err_msg = str(e).lower()
                if "404" in err_msg or "not_found" in err_msg:
                    continue
                # For other errors, continue attempting remaining candidate models
                continue
    except Exception:
        pass

    if not answer_text:
        # Fallback to local grounded synthesis if all candidate models fail or API error occurs
        top_text = valid_chunks[0]["text"][:300].strip()
        answer_text = f"Based on the uploaded university documents:\n\n{top_text}..."

    return {
        "answer": answer_text,
        "sources": sources,
    }
