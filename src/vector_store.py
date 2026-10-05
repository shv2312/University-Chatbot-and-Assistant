"""ChromaDB Vector Store management for university knowledge base.

Handles persistent indexing, document deletion, deduplication, and
semantic similarity search using Google GenAI embeddings or offline mock vectors.
"""

import os
import math
import hashlib
from typing import List, Dict, Any, Optional, Union
import chromadb
from dotenv import load_dotenv

load_dotenv()


def get_genai_client(api_key: Optional[str] = None) -> Any:
    """Instantiate a Google GenAI Client dynamically using current environment key."""
    from google import genai

    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key or key.strip().lower() in {"your_gemini_api_key_here", "your_api_key_here", "none"}:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not configured. "
            "Please provide a valid Gemini API key in the sidebar or enable Offline Mock Mode."
        )
    return genai.Client(api_key=key.strip())


def generate_mock_embedding(
    text_or_texts: Union[str, List[str]], dim: int = 768
) -> Union[List[float], List[List[float]]]:
    """Generate deterministic 768-dimensional float vectors derived from hashlib.sha256.

    Enables full offline similarity search and storage in ChromaDB without an API key.
    """
    def _embed_single(text: str) -> List[float]:
        tokens = text.lower().split()
        vec = [0.0] * dim
        for token in tokens:
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            vec[h % dim] += 1.0

        # Character tri-grams for sub-token and phrase matching
        for i in range(max(0, len(text) - 3)):
            ngram = text[i : i + 3].lower()
            h = int(hashlib.sha256(ngram.encode("utf-8")).hexdigest(), 16)
            vec[h % dim] += 0.2

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            return [x / norm for x in vec]
        return [1.0 / math.sqrt(dim)] * dim

    if isinstance(text_or_texts, str):
        return _embed_single(text_or_texts)
    return [_embed_single(t) for t in text_or_texts]


def get_embedding(
    text_or_texts: Union[str, List[str]], client: Optional[Any] = None
) -> Union[List[float], List[List[float]]]:
    """Generate vector embedding(s) using Google GenAI text-embedding-004 with safe fallback.

    Args:
        text_or_texts: Single text string or list of text strings.
        client: Optional genai.Client instance. If not provided, creates one
                dynamically from GEMINI_API_KEY environment variable.

    Returns:
        List of floats (for single string) or List of List of floats (for list of strings).
    """
    # If offline mode or if remote embedding fails, fall back to deterministic local mock vectors
    def _local_embed(t):
        import hashlib
        # Generate deterministic 768-dimensional normalized vector
        h = hashlib.sha256(t.encode("utf-8")).digest()
        raw = [((b / 255.0) - 0.5) for b in (h * 24)[:768]]
        norm = sum(x * x for x in raw) ** 0.5 or 1.0
        return [x / norm for x in raw]

    if os.getenv("USE_MOCK_EMBEDDINGS") == "1":
        if isinstance(text_or_texts, str):
            return _local_embed(text_or_texts)
        return [_local_embed(t) for t in text_or_texts]

    try:
        c = client or get_genai_client()
    except Exception:
        c = None

    if c:
        try:
            if isinstance(text_or_texts, str):
                res = c.models.embed_content(model="text-embedding-004", contents=text_or_texts)
                if hasattr(res, "embeddings") and res.embeddings:
                    return res.embeddings[0].values
                elif hasattr(res, "embedding") and res.embedding:
                    return res.embedding.values
            else:
                if not text_or_texts:
                    return []
                res = c.models.embed_content(model="text-embedding-004", contents=text_or_texts)
                if hasattr(res, "embeddings") and res.embeddings:
                    return [e.values for e in res.embeddings]
        except Exception:
            # Silently and safely fall back to local embeddings on ANY API error
            pass

    if isinstance(text_or_texts, str):
        return _local_embed(text_or_texts)
    return [_local_embed(t) for t in text_or_texts]


class VectorStore:
    """Manages persistent document embeddings within ChromaDB."""

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "university_knowledge",
        embedding_function: Optional[Any] = None,
        mock_mode: bool = False,
    ):
        """Initialize ChromaDB persistent client and knowledge collection.

        Args:
            persist_directory: Filesystem path to persist vector database.
            collection_name: Target collection name (default: 'university_knowledge').
            embedding_function: Optional custom embedding callable.
            mock_mode: If True, uses deterministic offline mock embeddings.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.mock_mode = mock_mode
        self._custom_embed_func = embedding_function

        # Initialize persistent client
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Access collection with cosine similarity metric
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def set_mock_mode(self, mock_mode: bool) -> None:
        """Toggle offline mock embedding mode."""
        self.mock_mode = mock_mode

    def embed_func(self, text_or_texts: Union[str, List[str]]):
        """Dynamic embedder respecting custom embedder, mock mode, or live Gemini API."""
        if self._custom_embed_func:
            return self._custom_embed_func(text_or_texts)
        if self.mock_mode:
            return generate_mock_embedding(text_or_texts)
        try:
            return get_embedding(text_or_texts)
        except Exception:
            # Fallback to mock embedding on any API errors to prevent ingestion crashes
            return generate_mock_embedding(text_or_texts)

    def add_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """Index a batch of text chunks into the collection.

        Before adding chunks of a specific filename, deletes any existing chunks
        matching that filename to prevent duplicate entries and support document updates.

        Args:
            chunks: List of chunk dictionaries containing 'id', 'text', and 'metadata'.

        Returns:
            Number of chunks successfully indexed.
        """
        if not chunks:
            return 0

        # Group filenames to handle deduplication per document
        filenames_to_update = set()
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            fname = meta.get("filename")
            if fname:
                filenames_to_update.add(fname)

        # Delete pre-existing chunks for these documents
        for fname in filenames_to_update:
            self.delete_document(fname)

        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # Generate embeddings in batch
        embeddings = self.embed_func(documents)

        # Batch upsert into Chroma collection
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        return len(chunks)

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks for a user query.

        Args:
            query: User's search query string.
            top_k: Number of nearest neighbors to retrieve (default: 4).

        Returns:
            List of dicts: [{"text": doc, "metadata": meta, "distance": dist}, ...]
        """
        if not query or not query.strip():
            return []

        total_count = self.collection.count()
        if total_count == 0:
            return []

        # Generate query embedding
        query_embedding = self.embed_func(query)

        effective_k = min(top_k, total_count)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        formatted_results = []
        for doc, meta, dist in zip(docs, metas, distances):
            formatted_results.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "distance": float(dist) if dist is not None else 0.0,
                }
            )

        return formatted_results

    def get_indexed_files(self) -> List[str]:
        """Return a sorted list of unique filenames currently indexed."""
        total_count = self.collection.count()
        if total_count == 0:
            return []

        data = self.collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])

        unique_files = set()
        for meta in metadatas:
            if meta and "filename" in meta:
                unique_files.add(meta["filename"])

        return sorted(list(unique_files))

    def get_document_chunk_counts(self) -> Dict[str, int]:
        """Return a dictionary mapping each indexed filename to its chunk count."""
        total_count = self.collection.count()
        if total_count == 0:
            return {}

        data = self.collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])

        counts: Dict[str, int] = {}
        for meta in metadatas:
            if meta and "filename" in meta:
                fname = meta["filename"]
                counts[fname] = counts.get(fname, 0) + 1

        return counts

    def delete_document(self, filename: str) -> bool:
        """Delete all chunks associated with the specified filename.

        Args:
            filename: Target document filename to delete.

        Returns:
            True if deletion was executed.
        """
        try:
            self.collection.delete(where={"filename": filename})
            return True
        except Exception:
            return False

    def clear_database(self) -> bool:
        """Purge all documents from the collection for a fresh reset."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception:
            return False
