# 🎓 Intelligent University Knowledge Assistant (RAG)

An end-to-end, locally persistent **Retrieval-Augmented Generation (RAG)** assistant built to help university students and faculty navigate official academic regulations, hostel guidelines, and examination rules with zero hallucinations and verified page-level citations.

---

## 📌 Project Overview

When navigating university life, finding accurate answers about attendance rules, hostel curfew hours, mess schedules, or examination codes of conduct often involves sifting through hundreds of pages of unsearchable PDF circulars and Word documents.

To solve this, we built the **Intelligent University Knowledge Assistant**. Our application ingests heterogenous university documents (`.pdf`, `.docx`, `.txt`, `.md`), chunks and indexes them into a local persistent vector store ([ChromaDB](https://www.trychroma.com/)), and uses Google's latest `gemini-1.5-flash` model with strict anti-hallucination prompts to answer questions accurately with exact document and page citations.

---

## 🚀 Key Features

- **Multi-Format Document Ingestion**: Extract page-aware text from PDF (via PyMuPDF), Word (`.docx`), plain text, and Markdown files.
- **Sliding-Window Chunking with Metadata**: Chunks documents with 800-character windows and 150-character overlaps, retaining source filename, page number, and chunk IDs for 100% traceable citations.
- **Persistent Local Vector Store**: Uses ChromaDB with cosine distance indexing stored in `data/chroma_db/`. No external database servers or cloud accounts required.
- **Document Update & Deduplication**: Re-indexing an updated policy automatically purges older chunks of that document, preventing stale data contamination.
- **Zero Hallucination Guarantee**: If an answer is not present in the uploaded university documents, the assistant strictly returns:  
  `"I couldn't find enough information about this in the uploaded university documents."`  
  Never guesses or speculates on rules.
- **Interactive Streamlit Web Dashboard**: Upload policies, monitor indexed chunks per file, delete documents, reset the database, or load pre-packaged university policies with one click.
- **Verified Sources Accordion**: Every answer displays collapsible source cards showing the cited document name, page number, and text excerpt.

---

## 🛠️ Technology Stack

| Layer | Tool / Framework | Purpose |
| :--- | :--- | :--- |
| **User Interface** | [Streamlit](https://streamlit.io/) | Interactive web dashboard and real-time chat interface |
| **Embeddings** | Google `text-embedding-004` | 768-dimensional semantic text embeddings |
| **LLM Reasoning** | Google `gemini-1.5-flash` | Grounded question answering with strict context adherence |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) | Local persistent vector search with HNSW cosine index |
| **PDF Extraction** | [PyMuPDF](https://pymupdf.readthedocs.io/) (`pymupdf`) | High-speed, page-by-page PDF text extraction |
| **DOCX Extraction** | `python-docx` | Structured paragraph parsing for Word documents |
| **Environment** | `python-dotenv` | Secure environment variable handling |

---

## 📐 System Architecture Diagram

```
[ University Documents ]
  (PDF / DOCX / TXT / MD)
          │
          ▼
┌─────────────────────────┐
│ src/document_processor  │ ──► Page Extraction + Sliding Window Chunking
└─────────────────────────┘      (Chunk Size: 800 | Overlap: 150)
          │
          ▼
┌─────────────────────────┐
│    src/vector_store     │ ──► Embeddings (text-embedding-004) + ChromaDB
└─────────────────────────┘      (Cosine Similarity | data/chroma_db/)
          │
          ▼
┌─────────────────────────┐
│    src/rag_pipeline     │ ──► Top-K Retrieval (k=4) + Grounded Prompting
└─────────────────────────┘      (Gemini 2.5 Flash | Zero Hallucination Guardrail)
          │
          ▼
┌─────────────────────────┐
│     Streamlit (app)     │ ──► Interactive Web Chat & Citation Inspector
└─────────────────────────┘
```

---

## 📂 Repository Layout

```text
C:\Projects\KVD Assignment\
├── app.py                         # Streamlit web application
├── requirements.txt               # Project dependencies
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git exclusion rules
├── data/
│   ├── documents/                 # User-uploaded files
│   ├── chroma_db/                 # Persistent vector store database
│   └── sample_documents/          # Bundled realistic test documents
│       ├── Academic_Regulations.txt
│       ├── Hostel_Guidelines.txt
│       └── Examination_Rules.txt
├── src/
│   ├── __init__.py
│   ├── document_processor.py      # Extraction & sliding-window chunking
│   ├── vector_store.py            # ChromaDB persistent collection manager
│   └── rag_pipeline.py            # Grounded generation pipeline
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py          # Unit tests for text extraction & chunking
│   └── test_rag_pipeline.py       # Unit tests for indexing, updates & guardrails
└── docs/
    ├── architecture.md            # Detailed technical architecture specification
    └── project_report.md          # Comprehensive academic mini-project report
```

---

## ⚙️ Installation & Setup

### 1. Clone or Open the Repository
```bash
cd "C:\Projects\KVD Assignment"
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```
Open `.env` and add your Google Gemini API key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```
*(You can obtain a free Gemini API key from [Google AI Studio](https://aistudio.google.com/).)*

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Running the Automated Test Suite

We implemented comprehensive unit tests covering all components. Run the complete test suite using Python's built-in `unittest` runner:

```bash
python -m unittest discover tests
```

Expected output:
```text
..............
----------------------------------------------------------------------
Ran 14 tests in 1.266s

OK
```

All 14 tests pass with zero errors, validating:
1. Multi-page PDF text extraction with 1-based indexing.
2. Word document (`.docx`) paragraph extraction.
3. Sliding-window chunking with boundary overlap.
4. ChromaDB cosine vector indexing and semantic retrieval.
5. Idempotent document update and replacement logic.
6. Target document deletion.
7. Anti-hallucination guardrail fallback when information is absent.
8. Semantic distance threshold filtering prunes irrelevant chunks ($> 0.65$).
9. Quota/API error fallback synthesis directly from grounded chunks.
10. Full offline mock demo mode with deterministic 768-dim vector embeddings.

---

## 🖥️ How to Run the Web Application

Launch the Streamlit app with:

```bash
streamlit run app.py
```

Once launched, your default web browser will open at:
```text
http://localhost:8501
```

### Quick Demo Walkthrough:
1. Look at the left sidebar: verify that the **Gemini API Connected** badge is active.
2. Under **Sample Documents**, click **⚡ Index Sample University Docs** to load the bundled Academic Regulations, Hostel Guidelines, and Examination Rules into ChromaDB.
3. The **Indexed Documents** inventory will display all 3 documents and their chunk counts.
4. Click any of the quick-ask sample buttons in the main chat container or type your own question.
5. Expand the **📚 Verified Sources & Page References** card to inspect the exact citations and text extracts.

---

## 💡 Sample Test Questions for Demonstration

### 1. Attendance & Academic Rules
- **Question**: *"What is the minimum attendance required and can I get condonation for medical reasons?"*
- **Assistant Answer**: Explains the 75% minimum aggregate attendance, the 65%–74% condonation window with certified medical proof, the Rs. 1,000 fee, and cites `Academic_Regulations.txt`, Page 1.

### 2. Hostel Guidelines
- **Question**: *"What time is hostel curfew and what are the mess dinner timings?"*
- **Assistant Answer**: Cites `Hostel_Guidelines.txt`, Page 1, specifying the strict 8:30 PM gate curfew and dinner service from 7:30 PM to 9:00 PM.

### 3. Examination Conduct
- **Question**: *"Can I bring a mobile phone into the exam hall?"*
- **Assistant Answer**: Cites `Examination_Rules.txt`, Page 1, confirming mobile phones are strictly prohibited, treated as malpractice, and result in paper cancellation.

### 4. Hallucination Guardrail Check
- **Question**: *"What is the university scholarship policy for aerospace engineering?"*
- **Assistant Answer**:  
  `"I couldn't find enough information about this in the uploaded university documents."`  
  *(Proves the model refuses to hallucinate facts absent from the knowledge base).*

---

## ⚠️ Limitations & Future Improvements

### Current Limitations:
- **Scanned Image PDFs**: PyMuPDF extracts programmatic text; scanned image-only PDFs require an OCR engine like Tesseract.
- **Complex Table Parsing**: Multi-column tabular data in PDFs can occasionally lose column alignment during flat text extraction.

### Future Enhancements:
- **Hybrid Search (BM25 + Dense Vectors)**: Combine sparse lexical matching for course codes with dense semantic vectors.
- **Multi-Modal Vision Ingestion**: Utilize Gemini's native multi-modal capabilities to directly parse document tables, floor plans, and complex figures.
- **Authentication & Role-Based Access**: Enable distinct access tiers (Students, Faculty, Examination Officers) to query role-specific confidential documents.
