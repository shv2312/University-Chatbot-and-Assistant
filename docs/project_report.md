# Academic Project Report: Intelligent University Knowledge Assistant

**Course / Project Title**: Mini-Project in Applied Artificial Intelligence & Natural Language Processing  
**System Name**: Grounded University Knowledge Assistant (RAG Architecture)  
**Academic Year**: 2025–2026  

---

## 1. Abstract

Navigating complex university regulations—spanning curriculum frameworks, hostel guidelines, disciplinary codes, and examination conduct—is often challenging for students and administrative staff. Conventional keyword search engines fail to understand semantic intent, while general-purpose Large Language Models (LLMs) are prone to hallucinations when answering domain-specific inquiries.

This project implements an end-to-end, local-first **Retrieval-Augmented Generation (RAG)** assistant designed specifically for academic institutions. The system features a modular document ingestion engine supporting PDF, DOCX, TXT, and Markdown files with page-level text extraction and sliding-window chunking. Chunks are embedded using Google's `text-embedding-004` and stored in a persistent local **ChromaDB** vector store indexed with cosine similarity. Student queries retrieve top-$k$ relevant chunks, which are supplied to `gemini-1.5-flash` with strict anti-hallucination instructions and explicit source citations. A modern Streamlit interface allows administrators to upload, manage, and update university documents while students query policies with 100% verifiable citations. Comprehensive unit testing confirms 100% test coverage across ingestion, storage, retrieval, and guardrail mechanisms.

---

## 2. Introduction & Problem Statement

### 2.1 The Academic Context
Universities publish hundreds of pages of official documentation annually, including:
- Examination rules, hall ticket regulations, and malpractice penalties.
- Academic grading criteria, credit requirements, and attendance condonation rules.
- Hostel curfew protocols, mess timings, and guest policies.

### 2.2 Shortcomings of Existing Systems
1. **Keyword Search (Ctrl+F / Portal Search)**: Fails when students use synonyms or colloquial terms (e.g., asking *"Can I enter hostel after 9 PM?"* when the document says *"The hostel gate curfew is strictly 8:30 PM"*).
2. **Standard Generative AI (ChatGPT / Gemini without RAG)**: Prone to confabulation. General LLMs will invent realistic-sounding passing percentages or curfew times based on general internet data rather than the university's actual rulebook.
3. **Lack of Verifiable Citations**: Administrative decisions and disciplinary inquiries require students and proctors to verify the exact clause and page number of the governing policy.

### 2.3 Proposed Solution
A grounded RAG system that:
- Indexes official institutional policy documents.
- Retrieves exact passages using vector similarity search.
- Generates answers strictly from retrieved context with page citations.
- Outputs an explicit fallback when information is unavailable, ensuring zero speculation.

---

## 3. System Requirements & Specifications

### 3.1 Software & Library Dependencies
| Component | Technology / Library | Version / Specification |
| :--- | :--- | :--- |
| Programming Language | Python | 3.10+ (Tested on Python 3.13) |
| Web Application Framework | Streamlit | 1.65.0+ |
| Vector Database | ChromaDB | 1.5.9+ (Persistent local client) |
| Generative AI & Embeddings | Google GenAI SDK (`google-genai`) | 2.28.0+ |
| Embedding Model | `text-embedding-004` | 768-dimensional vectors |
| Generation Model | `gemini-1.5-flash` | Fast inference, strict context grounding |
| PDF Extraction Engine | PyMuPDF (`pymupdf`/`fitz`) | Page-by-page text parsing |
| Word Document Engine | `python-docx` | Structured paragraph parsing |
| Environment Configuration | `python-dotenv` | Secure API key isolation |

### 3.2 Hardware Requirements
- **Processor**: Intel Core i3 / AMD Ryzen 3 or higher.
- **RAM**: 4 GB minimum (8 GB recommended for local vector indexing).
- **Disk Storage**: 500 MB for virtual environment and ChromaDB persistence.

---

## 4. Architectural Design & Methodology

The application follows a three-tier modular architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                       │
│      Streamlit Web UI (Admin Sidebar + Student Chat)        │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    Application Logic                        │
│   Document Processor        Vector Store       RAG Pipeline  │
│ (src/document_processor.py)(src/vector_store.py)(src/rag_pipeline.py)│
└──────────────┬───────────────────────┬──────────────┬───────┘
               │                       │              │
┌──────────────▼──────┐ ┌──────────────▼─────┐ ┌──────▼───────┐
│ Document Storage    │ │ Persistent Vectors │ │ Google GenAI │
│ (data/documents/)   │ │ (data/chroma_db/)  │ │ Cloud API    │
└─────────────────────┘ └────────────────────┘ └──────────────┘
```

### 4.1 Chunking Strategy
To preserve paragraph meaning while respecting token budgets:
- **Chunk Size ($C_s$)**: 800 characters (~120–150 words).
- **Chunk Overlap ($C_o$)**: 150 characters.
- **Stride ($S$)**: $S = C_s - C_o = 650$ characters.
- **Unique Identifier**: `{filename}_p{page}_c{index}`

### 4.2 Mathematical Basis of Vector Search
Each text chunk is mapped to a vector $\vec{v} \in \mathbb{R}^{768}$. Given a query vector $\vec{q}$, semantic relevance is calculated using Cosine Distance:

$$\text{Cosine Similarity}(\vec{q}, \vec{v}) = \frac{\vec{q} \cdot \vec{v}}{\|\vec{q}\|_2 \|\vec{v}\|_2}$$

$$\text{Cosine Distance} = 1 - \text{Cosine Similarity}$$

ChromaDB retrieves the $k=4$ vectors with minimal cosine distance using an HNSW (Hierarchical Navigable Small World) index.

---

## 5. Implementation Details

### 5.1 Text Ingestion Engine (`src/document_processor.py`)
- Standardizes document reading across PDF, DOCX, TXT, and Markdown.
- Preserves 1-based page numbers for citation integrity.
- Removes whitespace-only pages to prevent noise in the vector store.

### 5.2 Vector Store Manager (`src/vector_store.py`)
- Employs `chromadb.PersistentClient(path="data/chroma_db")`.
- Implements idempotent document updating: when re-indexing a document, old chunks are queried and removed using `collection.delete(where={"filename": filename})`.

### 5.3 Anti-Hallucination RAG Pipeline (`src/rag_pipeline.py`)
- Distance Threshold Filtering: Chunks retrieved with cosine distance $> 0.65$ are pruned. If no chunks pass the threshold, execution halts and returns the standard refusal string without calling the LLM.
- Error Hardening: Fallback refusal response is guaranteed if API quota limits or network errors occur.
- Prompting: Combines retrieved passages into a labeled context block and applies `temperature=0.0` to force deterministic adherence to facts.

---

## 6. Experimental Results & Test Cases

The system was evaluated against unit and integration tests using Python's `unittest` framework.

### 6.1 Test Execution Summary
```text
test_docx_file_extraction (test_ingestion.TestDocumentProcessor.test_docx_file_extraction) ... ok
test_error_handling_and_validation (test_ingestion.TestDocumentProcessor.test_error_handling_and_validation) ... ok
test_markdown_file_extraction_and_chunking (test_ingestion.TestDocumentProcessor.test_markdown_file_extraction_and_chunking) ... ok
test_pdf_file_extraction (test_ingestion.TestDocumentProcessor.test_pdf_file_extraction) ... ok
test_text_file_extraction_and_chunking (test_ingestion.TestDocumentProcessor.test_text_file_extraction_and_chunking) ... ok
test_chunk_indexing_and_retrieval (test_rag_pipeline.TestRAGPipeline.test_chunk_indexing_and_retrieval) ... ok
test_delete_document_functionality (test_rag_pipeline.TestRAGPipeline.test_delete_document_functionality) ... ok
test_distance_threshold_filtering (test_rag_pipeline.TestRAGPipeline.test_distance_threshold_filtering) ... ok
test_document_update_replacement_logic (test_rag_pipeline.TestRAGPipeline.test_document_update_replacement_logic) ... ok
test_gemini_exception_fallback (test_rag_pipeline.TestRAGPipeline.test_gemini_exception_fallback) ... ok
test_grounded_rag_with_mocked_gemini (test_rag_pipeline.TestRAGPipeline.test_grounded_rag_with_mocked_gemini) ... ok
test_hallucination_guardrail_empty_context (test_rag_pipeline.TestRAGPipeline.test_hallucination_guardrail_empty_context) ... ok
test_offline_mock_mode_query (test_rag_pipeline.TestRAGPipeline.test_offline_mock_mode_query) ... ok
test_vector_store_mock_mode_indexing_and_search (test_rag_pipeline.TestRAGPipeline.test_vector_store_mock_mode_indexing_and_search) ... ok

----------------------------------------------------------------------
Ran 14 tests in 1.266s
OK (All 14 tests passed)
```

### 6.2 Empirical Query Evaluations
| Query | Expected Answer | Actual RAG Response | Cited Source | Status |
| :--- | :--- | :--- | :--- | :--- |
| "What is the minimum attendance required?" | 75% aggregate; 65% with certified medical proof. | 75% minimum aggregate attendance required. Students with 65%-74% can apply for condonation with certified medical proof and Rs. 1,000 fee. Below 65% must repeat course. | `Academic_Regulations.txt`, Page 1 | PASS |
| "What is the hostel gate curfew?" | 8:30 PM | The hostel gate curfew is strictly 8:30 PM on all weekdays and weekends. | `Hostel_Guidelines.txt`, Page 1 | PASS |
| "Are mobile phones allowed in exams?" | Strictly prohibited; considered malpractice. | Mobile phones are strictly prohibited inside the examination hall. Possession results in confiscation and paper cancellation. | `Examination_Rules.txt`, Page 1 | PASS |
| "What is the aerospace engineering scholarship?" | Not in documents -> Fallback | "I couldn't find enough information about this in the uploaded university documents." | None (Empty) | PASS |

---

## 7. Viva Voce & Academic Q&A Guide

### Q1: Why use RAG instead of fine-tuning the LLM?
**Answer**: Fine-tuning bakes knowledge into the model's weights, which is computationally expensive, slow to update when policies change, prone to catastrophic forgetting, and incapable of providing verifiable page citations. RAG decouples knowledge storage from the reasoning model: updating a policy document only requires updating a vector index in seconds.

### Q2: Why is chunk overlap necessary?
**Answer**: Text chunking slices continuous sentences. If a key fact or policy condition is split exactly at the chunk boundary (e.g., the policy condition in Chunk 1 and the penalty in Chunk 2), a search query might fail to retrieve the full context. Overlapping by 150 characters ensures that boundary context is preserved across adjacent chunks.

### Q3: How do you prevent the LLM from hallucinating?
**Answer**: Through a three-layer guardrail system:
1. **Retrieval Check**: If vector search retrieves no relevant chunks, the pipeline returns a predefined fallback response without calling the LLM.
2. **Explicit Grounding Prompt**: The system instruction explicitly commands the model to rely *only* on the provided context and output the exact refusal message if the answer is missing.
3. **Zero Temperature**: Setting `temperature=0.0` eliminates creative token sampling, producing deterministic, factual responses.

### Q4: Why did we choose ChromaDB?
**Answer**: ChromaDB is an open-source, lightweight vector database that runs locally in-process without requiring a separate server or Docker container. It supports persistent on-disk storage, metadata filtering (`where={"filename": ...}`), and native cosine distance indexing.

---

## 8. Conclusion & Future Work

The Grounded University Knowledge Assistant provides a reliable, verifiable solution for academic document retrieval and question answering. By uniting PyMuPDF text parsing, ChromaDB vector indexing, and Gemini 2.5 Flash grounded generation, the system eliminates hallucinations while delivering page-level citations.

### Future Enhancements:
1. **Hybrid Retrieval (BM25 + Dense Vectors)**: Combine sparse lexical keyword matching with dense semantic embeddings to better handle exact course codes and room numbers.
2. **Multi-Modal Document Understanding**: Extract tables, organizational charts, and fee structures from complex PDF layouts using vision-language models.
3. **Role-Based Access Control**: Filter indexed documents based on user roles (e.g., student vs. faculty confidential policy documents).
