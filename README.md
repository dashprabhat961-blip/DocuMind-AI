# DocuMind AI — Chat with Your PDFs Using RAG

**An AI-powered PDF question-answering application built with Retrieval-Augmented Generation (RAG), Sentence Transformers, FAISS, and Streamlit.**

DocuMind AI allows users to upload PDF documents, ask questions about their contents, and receive AI-generated answers supported by relevant document excerpts and source references. It also provides document summarization and key takeaways.

The project explores how semantic search and large language models can work together to make information inside documents easier to access.

## ✨ Features

- **PDF Question Answering:** Ask natural-language questions about uploaded documents.
- **Retrieval-Augmented Generation (RAG):** Retrieve relevant document chunks and use them as context for answer generation.
- **Semantic Search:** Generate text embeddings using Sentence Transformers and retrieve relevant chunks with FAISS.
- **Source References:** Display relevant excerpts and source information to help users verify answers.
- **Document Summarization:** Generate a summary of document content using the retrieval and generation pipeline.
- **Key Takeaways:** Extract important points from document content.
- **Text Preprocessing:** Normalize extracted text, repair line breaks conservatively, and preserve document structure where possible.
- **Configurable LLM Providers:** Support the configured Gemini and Groq integrations.
- **Automated Tests:** Test document processing, chunking, metadata preservation, and retrieval behavior.
- **Retrieval Evaluation:** Evaluate retrieval quality using a predefined question dataset and metrics such as Precision@3, Recall@3, and Top-1 Accuracy.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| User interface | Streamlit |
| Programming language | Python |
| PDF extraction | PyPDF |
| Text embeddings | Sentence Transformers |
| Embedding model | `all-MiniLM-L6-v2` |
| Vector similarity search | FAISS |
| AI response generation | Configured Gemini / Groq integration |
| Evaluation | Python, CSV, custom retrieval metrics |
| Testing | Python `unittest` |

## 🏗️ How It Works

DocuMind AI follows a Retrieval-Augmented Generation workflow.

1. **PDF Upload:** The user provides a PDF document.
2. **Text Extraction:** Text is extracted from the document's pages.
3. **Text Preprocessing:** Extracted text is normalized while preserving useful structure and page metadata.
4. **Text Chunking:** The document is divided into smaller, overlapping chunks to support retrieval.
5. **Embedding Generation:** Sentence Transformers converts chunks into numerical vector representations.
6. **Vector Indexing:** FAISS indexes the embeddings for similarity-based retrieval.
7. **Question Processing:** The user's question is embedded and compared against indexed document chunks.
8. **Context Retrieval:** Relevant chunks are selected and supplied to the configured language model.
9. **Answer Generation:** The model generates a response grounded in the retrieved context.
10. **Source Display:** Relevant excerpts and source references help users inspect the supporting content.

### Architecture

```text
                 ┌─────────────────────┐
                 │     Streamlit UI    │
                 │  PDF Upload & Chat  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   PDF Text Extract  │
                 │   & Preprocessing   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Text Chunking &    │
                 │  Metadata Tracking  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Sentence Transformers│
                 │  Embedding Model    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    FAISS Index      │
                 └──────────┬──────────┘
                            │
             User Question ─┤
                            ▼
                 ┌─────────────────────┐
                 │ Relevant Context    │
                 │ Retrieval (Top-k)   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Configured LLM      │
                 │ Gemini / Groq       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Answer, Excerpts    │
                 │ & Source References │
                 └─────────────────────┘
```

## 📂 Project Structure

```text
DocuMind_AI/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── test_rag_pipeline.py
├── evaluation_dataset.py
├── metrics.py
├── run_evaluation.py
│
├── generate_sample_pdf.py
├── sample_ai_lecture_notes.pdf
│
├── evaluation_results.csv
└── evaluation_summary.csv
```

**File descriptions**

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application and RAG workflow |
| `requirements.txt` | Project dependencies |
| `.env.example` | Example environment-variable configuration |
| `test_rag_pipeline.py` | Automated unit and regression tests |
| `evaluation_dataset.py` | Baseline evaluation questions and expected information |
| `metrics.py` | Retrieval evaluation metric calculations |
| `run_evaluation.py` | Runs the retrieval evaluation and exports CSV results |
| `generate_sample_pdf.py` | Generates a sample PDF for testing |
| `sample_ai_lecture_notes.pdf` | Sample document, if redistribution is permitted |
| `evaluation_results.csv` | Per-question evaluation results |
| `evaluation_summary.csv` | Aggregate evaluation metrics |

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or a compatible Python version
- pip
- A Gemini or Groq API key for the configured generation provider
- Internet access for initial dependency and embedding-model downloads

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DocuMind-AI.git
cd DocuMind-AI
```

Replace `YOUR_USERNAME` with your GitHub username and use your actual repository URL.

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks virtual-environment activation, use Command Prompt or configure your local execution policy appropriately.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file using `.env.example` as a reference.

Add the API key required by your selected provider, using the exact variable name expected by the application.

For example:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

Use your actual configured variable names if they differ. Keep `.env` private and never commit API keys to GitHub.

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

### 6. Use DocuMind AI

1. Upload a supported PDF document.
2. Wait for document processing and indexing to complete.
3. Ask a question about the document.
4. Review the generated answer and its supporting source excerpts.
5. Try the summary and key-takeaway features.

## 🧪 Testing

Run the automated test suite from the project root:

```bash
python -m unittest -v test_rag_pipeline.py
```

The current test suite contains six tests covering:

- Chunk-boundary integrity and forward progress
- Validation of chunking parameters
- Empty pages, blank text, and non-selectable PDFs
- Preservation of page and source metadata
- Retrieval regression against baseline questions
- Whitespace normalization and conservative line repair

**Latest local test result:** 6 tests passed.

## 📊 Retrieval Evaluation

The project's local evaluation run produced the following results:

| Metric | Result |
|---|---:|
| Precision@3 | 0.3333 |
| Recall@3 | 1.0000 |
| Top-1 Accuracy | 1.0000 |

Run the evaluation using:

```bash
python run_evaluation.py
```

The script generates:

- `evaluation_results.csv` — per-question results
- `evaluation_summary.csv` — aggregate metrics

### Interpreting the results

- **Precision@3 = 0.3333:** On average, one-third of the three retrieved results were relevant under the current evaluation's relevance criteria.
- **Recall@3 = 1.0000:** The evaluation retrieved all items considered relevant within the defined evaluation dataset at the selected cutoff.
- **Top-1 Accuracy = 1.0000:** The first retrieved result matched the evaluation's expected relevance criterion for every evaluated question.

These are results from the current local benchmark, not a guarantee of performance on arbitrary PDFs. They depend on the dataset, relevance labels, and metric implementation. The benchmark is small, so broader testing is needed before drawing general conclusions.

## ⚠️ Current Limitations

- Answer quality depends on the PDF's extracted text, retrieval quality, and the configured language model.
- Scanned PDFs without selectable text may require OCR, which is not guaranteed by the current pipeline.
- Retrieval-based summaries may not cover every section of a long document.
- The language model can still generate incorrect or incomplete answers; source verification is recommended.
- The current evaluation dataset is limited and should be expanded to cover more document types and question patterns.
- API-based generation requires valid provider credentials and may incur usage costs.

## 🔮 Future Improvements

- Improve long-document summarization through hierarchical or map-reduce summarization.
- Expand the evaluation dataset and include additional retrieval metrics.
- Compare embedding models and retrieval strategies.
- Add OCR support for scanned PDFs.
- Improve handling of tables, multi-column layouts, and complex document structures.
- Add conversation-level context management and more robust error handling.
- Deploy the application and add a public demo, subject to hosting and API limits.

## 🎯 Learning Outcomes

Through this project, I explored:

- Building a Retrieval-Augmented Generation pipeline
- Extracting and preprocessing PDF content
- Creating semantic embeddings
- Indexing and retrieving vectors using FAISS
- Connecting retrieval results to a language model
- Preserving source metadata for traceable answers
- Writing automated regression tests
- Evaluating retrieval quality using quantitative metrics

## 👨‍💻 Author

**Prabhat Shrinivas Dash**

B.Sc. Artificial Intelligence Student

Interested in Artificial Intelligence, Machine Learning, Generative AI, and practical AI applications.

- GitHub: [Your GitHub Profile](https://github.com/YOUR_USERNAME)
- LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/YOUR_LINKEDIN_USERNAME/)

Replace the profile placeholders with your actual links.

## 📄 License

Choose a license before distributing the project publicly. If you want others to reuse and modify your code, consider the MIT License after checking that all included code and assets can be licensed accordingly.

---

*DocuMind AI is a learning and portfolio project exploring document question answering with Retrieval-Augmented Generation.*
