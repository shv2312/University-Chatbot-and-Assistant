"""Unit and integration tests for ChromaDB vector store and grounded RAG pipeline."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from src.vector_store import VectorStore
from src.rag_pipeline import answer_query, FALLBACK_ANSWER, SYSTEM_INSTRUCTION


def mock_embedder(text_or_texts):
    """Deterministic mock embedding function producing 32-dim unit vectors."""
    is_single = isinstance(text_or_texts, str)
    texts = [text_or_texts] if is_single else text_or_texts

    embeddings = []
    for text in texts:
        vec = [0.0] * 32
        for i, ch in enumerate(text.lower()[:32]):
            vec[i] = ord(ch) / 255.0
        norm = sum(x**2 for x in vec) ** 0.5 or 1.0
        embeddings.append([x / norm for x in vec])

    return embeddings[0] if is_single else embeddings


class TestRAGPipeline(unittest.TestCase):
    """Test suite covering vector indexing, document update/replacement, and RAG grounding."""

    def setUp(self):
        """Create an isolated temporary ChromaDB persistence directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = VectorStore(
            persist_directory=self.temp_dir,
            collection_name="test_knowledge",
            embedding_function=mock_embedder,
        )

    def tearDown(self):
        """Clean up the temporary vector database."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_chunk_indexing_and_retrieval(self):
        """Test 1: Validate chunk indexing, metadata persistence, and similarity search."""
        chunks = [
            {
                "id": "Academic_Regulations.txt_p1_c0",
                "text": "Minimum attendance required is 75% for appearing in semester exams.",
                "metadata": {"filename": "Academic_Regulations.txt", "page": 1},
            },
            {
                "id": "Academic_Regulations.txt_p1_c1",
                "text": "Attendance between 65% and 74% can be condoned with certified medical proof.",
                "metadata": {"filename": "Academic_Regulations.txt", "page": 1},
            },
            {
                "id": "Hostel_Guidelines.txt_p1_c0",
                "text": "Hostel curfew is strictly 8:30 PM on all days.",
                "metadata": {"filename": "Hostel_Guidelines.txt", "page": 1},
            },
        ]

        added_count = self.vector_store.add_documents(chunks)
        self.assertEqual(added_count, 3)

        # Verify indexed files
        indexed_files = self.vector_store.get_indexed_files()
        self.assertEqual(indexed_files, ["Academic_Regulations.txt", "Hostel_Guidelines.txt"])

        # Verify search returns proper structure
        results = self.vector_store.search("attendance requirements", top_k=2)
        self.assertGreaterEqual(len(results), 1)
        for r in results:
            self.assertIn("text", r)
            self.assertIn("metadata", r)
            self.assertIn("distance", r)
            self.assertIn("filename", r["metadata"])
            self.assertIn("page", r["metadata"])

    def test_document_update_replacement_logic(self):
        """Test 2: Verify old chunks are cleanly purged when re-indexing the same file."""
        initial_chunks = [
            {
                "id": "Academic_Regulations.txt_p1_c0",
                "text": "Initial Draft: 75% attendance mandatory.",
                "metadata": {"filename": "Academic_Regulations.txt", "page": 1},
            },
            {
                "id": "Academic_Regulations.txt_p1_c1",
                "text": "Initial Draft: 50 marks minimum passing mark.",
                "metadata": {"filename": "Academic_Regulations.txt", "page": 1},
            },
            {
                "id": "Academic_Regulations.txt_p1_c2",
                "text": "Initial Draft: Grade O is 90-100 marks.",
                "metadata": {"filename": "Academic_Regulations.txt", "page": 1},
            },
        ]

        self.vector_store.add_documents(initial_chunks)
        counts = self.vector_store.get_document_chunk_counts()
        self.assertEqual(counts["Academic_Regulations.txt"], 3)

        # Re-index with updated version containing only 1 chunk
        updated_chunks = [
            {
                "id": "Academic_Regulations.txt_p1_c0",
                "text": "Revised Policy: 75% attendance mandatory, condonation fee Rs. 1000.",
                "metadata": {"filename": "Academic_Regulations.txt", "page": 1},
            }
        ]

        self.vector_store.add_documents(updated_chunks)
        new_counts = self.vector_store.get_document_chunk_counts()
        self.assertEqual(new_counts["Academic_Regulations.txt"], 1)

        # Confirm content was updated
        search_res = self.vector_store.search("Revised Policy", top_k=1)
        self.assertIn("Revised Policy", search_res[0]["text"])

    def test_hallucination_guardrail_empty_context(self):
        """Test 3: Verify strict fallback answer when no relevant chunks exist."""
        # Querying an empty vector store
        res = answer_query("What is the hostel curfew time?", self.vector_store)

        self.assertEqual(res["answer"], FALLBACK_ANSWER)
        self.assertEqual(res["sources"], [])

    def test_delete_document_functionality(self):
        """Test 4: Verify complete removal of a specific document's chunks."""
        chunks = [
            {
                "id": "DocA.txt_p1_c0",
                "text": "Document A content.",
                "metadata": {"filename": "DocA.txt", "page": 1},
            },
            {
                "id": "DocB.txt_p1_c0",
                "text": "Document B content.",
                "metadata": {"filename": "DocB.txt", "page": 1},
            },
        ]
        self.vector_store.add_documents(chunks)
        self.assertEqual(len(self.vector_store.get_indexed_files()), 2)

        # Delete DocA
        deleted = self.vector_store.delete_document("DocA.txt")
        self.assertTrue(deleted)

        remaining_files = self.vector_store.get_indexed_files()
        self.assertNotIn("DocA.txt", remaining_files)
        self.assertIn("DocB.txt", remaining_files)
        self.assertEqual(len(remaining_files), 1)

    def test_grounded_rag_with_mocked_gemini(self):
        """Test 5: Verify prompt construction, system instructions, and citation extraction."""
        chunks = [
            {
                "id": "Hostel_Guidelines.txt_p1_c0",
                "text": "The hostel gate curfew is strictly 8:30 PM on all weekdays and weekends.",
                "metadata": {"filename": "Hostel_Guidelines.txt", "page": 1},
            }
        ]
        self.vector_store.add_documents(chunks)

        # Mock Gemini Client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = (
            "According to [Source: Hostel_Guidelines.txt, Page: 1], the hostel curfew is 8:30 PM."
        )
        mock_client.models.generate_content.return_value = mock_response

        result = answer_query(
            "What is the hostel curfew?",
            self.vector_store,
            client=mock_client,
        )

        # Verify model call
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gemini-1.5-flash")
        self.assertIn("[Source: Hostel_Guidelines.txt, Page: 1]", call_kwargs["contents"])
        self.assertEqual(
            call_kwargs["config"].system_instruction, SYSTEM_INSTRUCTION
        )

        # Verify returned structure
        self.assertEqual(result["answer"], mock_response.text)
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["filename"], "Hostel_Guidelines.txt")
        self.assertEqual(result["sources"][0]["page"], 1)

    def test_distance_threshold_filtering(self):
        """Test 6: Verify chunks exceeding distance_threshold are rejected and trigger fallback."""
        mock_vs = MagicMock()
        # Return a chunk with distance 0.85 (exceeding default 0.65 threshold)
        mock_vs.search.return_value = [
            {
                "text": "Completely irrelevant content from another department.",
                "metadata": {"filename": "Irrelevant.txt", "page": 1},
                "distance": 0.85,
            }
        ]

        result = answer_query(
            "What is the quantum computing curriculum?",
            mock_vs,
            distance_threshold=0.65,
        )

        self.assertEqual(result["answer"], FALLBACK_ANSWER)
        self.assertEqual(result["sources"], [])

    def test_gemini_exception_fallback(self):
        """Test 7: Verify fallback answer is returned gracefully if Gemini raises an error."""
        mock_vs = MagicMock()
        mock_vs.search.return_value = [
            {
                "text": "Relevant attendance text.",
                "metadata": {"filename": "Academic_Regulations.txt", "page": 1},
                "distance": 0.20,
            }
        ]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API Quota Exceeded")

        result = answer_query(
            "What is the attendance requirement?",
            mock_vs,
            client=mock_client,
            distance_threshold=0.65,
        )

        self.assertTrue(
            result["answer"].startswith("Based on the uploaded university documents:")
            or result["answer"].startswith("Based on the uploaded documents:")
        )
        self.assertEqual(len(result["sources"]), 1)

    def test_offline_mock_mode_query(self):
        """Test 8: Verify offline mock mode synthesizes answer from top chunk without calling Gemini."""
        mock_vs = MagicMock()
        mock_vs.search.return_value = [
            {
                "text": "The hostel gate curfew is strictly 8:30 PM on all days.",
                "metadata": {"filename": "Hostel_Guidelines.txt", "page": 1},
                "distance": 0.35,
            }
        ]

        result = answer_query(
            "What is the hostel curfew?",
            mock_vs,
            mock_mode=True,
        )

        self.assertTrue(
            result["answer"].startswith("Based on the uploaded university documents:")
            or result["answer"].startswith("Based on the uploaded documents:")
        )
        self.assertIn("8:30 PM", result["answer"])
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["filename"], "Hostel_Guidelines.txt")

    def test_vector_store_mock_mode_indexing_and_search(self):
        """Test 9: Verify VectorStore in mock mode indexes and retrieves using deterministic vectors."""
        mock_store = VectorStore(
            persist_directory=self.temp_dir,
            collection_name="mock_test_col",
            mock_mode=True,
        )
        chunks = [
            {
                "id": "Hostel_p1_c0",
                "text": "Hostel curfew is strictly 8:30 PM every night.",
                "metadata": {"filename": "Hostel.txt", "page": 1},
            }
        ]
        added = mock_store.add_documents(chunks)
        self.assertEqual(added, 1)

        search_res = mock_store.search("hostel curfew", top_k=1)
        self.assertEqual(len(search_res), 1)
        self.assertIn("Hostel curfew", search_res[0]["text"])
        self.assertLess(search_res[0]["distance"], 0.65)


if __name__ == "__main__":
    unittest.main()
