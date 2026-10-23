# run_evaluation.py
"""Run the fixed evaluation benchmark for DocuMind AI.

This script:
1. Generates the in‑memory sample PDF used in the test suite.
2. Extracts text, chunks it, builds a FAISS index.
3. Retrieves relevant chunks for each query in `evaluation_dataset.EVALUATION_DATASET`.
4. Computes Precision@3, Recall@3 and Top‑1 page accuracy using `metrics.evaluate_queries`.
5. Writes per‑question results to `evaluation_results.csv` and aggregate metrics to `evaluation_summary.csv`.
"""
import csv
import io
import os
from typing import List

# Local imports from the project
from app import (
    extract_text_from_pdf,
    chunk_text,
    get_embedding_model,
    build_faiss_index,
    retrieve_relevant_chunks,
    DocumentChunk,
)
from metrics import evaluate_queries
from evaluation_dataset import EVALUATION_DATASET

# ---------------------------------------------------------------------------
# Helper: recreate the minimal in‑memory PDF generation used in the test suite.
# ---------------------------------------------------------------------------

def create_in_memory_pdf(pages_text: List[List[str]]) -> io.BytesIO:
    """Creates a tiny PDF (PDF‑1.4) directly in memory.

    The structure mirrors the one used in `test_rag_pipeline.py` so that the
    extracted pages match the expectations of the benchmark.
    """
    total_pages = len(pages_text)
    # Object IDs: 1 = Catalog, 2 = Pages, 3 = Font, then page objects start at 4.
    page_ids = [4 + i * 2 for i in range(total_pages)]
    pdf_objs = {}
    pdf_objs[1] = "<< /Type /Catalog /Pages 2 0 R >>"
    pdf_objs[2] = f"<< /Type /Pages /Kids [{' '.join(f'{pid} 0 R' for pid in page_ids)}] /Count {total_pages} >>"
    pdf_objs[3] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    for i, lines in enumerate(pages_text):
        p_obj_id = page_ids[i]
        c_obj_id = p_obj_id + 1
        pdf_objs[p_obj_id] = (
            f"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> >> /MediaBox [0 0 612 792] /Contents {c_obj_id} 0 R >>"
        )
        stream_lines = ["BT", "/F1 11 Tf", "50 740 Td", "16 TL"]
        for line in lines:
            safe_line = line.replace('\\', '\\\\\\').replace('(', '\\\(').replace(')', '\\\)')
            stream_lines.append(f"({safe_line}) Tj")
        stream_lines.append("ET")
        stream_str = "\n".join(stream_lines)
        pdf_objs[c_obj_id] = f"<< /Length {len(stream_str.encode('latin1'))} >>\nstream\n{stream_str}\nendstream"

    # Assemble final PDF
    out = ["%PDF-1.4\n"]
    xref = {}
    pos = len(out[0].encode('latin1'))
    for k in sorted(pdf_objs.keys()):
        xref[k] = pos
        block = f"{k} 0 obj\n{pdf_objs[k]}\nendobj\n"
        out.append(block)
        pos += len(block.encode('latin1'))
    xref_pos = pos
    out.append(f"xref\n0 {len(pdf_objs) + 1}\n0000000000 65535 f \n")
    for k in sorted(pdf_objs.keys()):
        out.append(f"{xref[k]:010d} 00000 n \n")
    out.append(
        f"trailer\n<< /Size {len(pdf_objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF"
    )
    return io.BytesIO("".join(out).encode('latin1'))

# ---------------------------------------------------------------------------
# Main evaluation workflow
# ---------------------------------------------------------------------------

def main():
    # 1. Build the sample PDF (identical to the one used in the test suite)
    pdf_stream = create_in_memory_pdf([
        [
            "CS 501: Advanced Artificial Intelligence & Deep Learning",
            "Instructor: Dr. Alan Turing | Semester: Fall 2026",
            "Lecture 1: Foundations of Representation Learning",
            "In representation learning, machine learning algorithms discover feature representations",
            "directly from raw data. Traditional machine learning relies heavily on manual feature engineering.",
        ],
        [
            "Lecture 2: The Transformer Architecture and Self-Attention",
            "Self-Attention Formula:",
            "Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V",
            "Where Q is Queries, K is Keys, V is Values, and d_k is the dimension of the keys.",
            "The scaling factor sqrt(d_k) prevents the dot products from growing excessively large.",
        ],
        [
            "Lecture 3: Retrieval-Augmented Generation (RAG) Systems",
            "Vector Search with FAISS provides optimized nearest neighbor search.",
            "Final Exam Date: The comprehensive final exam will take place on December 18th, from 9:00 AM to 12:00 PM in Hall B.",
        ],
    ])

    # 2. Extract pages
    pages, err = extract_text_from_pdf(pdf_stream)
    if err:
        raise RuntimeError(f"PDF extraction error: {err}")

    # 3. Chunk the text (same hyper‑parameters used in the UI defaults)
    chunks: List[DocumentChunk] = chunk_text(
        pages,
        source_name="sample_ai_lecture_notes.pdf",
        chunk_size=700,
        chunk_overlap=150,
    )

    # 4. Build FAISS index
    embedding_model = get_embedding_model()
    if embedding_model is None:
        raise RuntimeError("Failed to load the embedding model")
    index = build_faiss_index(chunks, embedding_model)

    # 5. Retrieval function compatible with `metrics.evaluate_queries`
    def retrieve_fn(query: str):
        return retrieve_relevant_chunks(query, index, chunks, embedding_model, top_k=3)

    # 6. Assemble query/relevant pairs for the evaluator
    queries = [(entry["query"], entry["relevant_pages"]) for entry in EVALUATION_DATASET]

    # 7. Compute metrics
    results = evaluate_queries(queries, retrieve_fn, k=3)

    # 8. Write per‑question CSV
    per_path = os.path.join(os.getcwd(), "evaluation_results.csv")
    with open(per_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "precision@3", "recall@3", "top1_accuracy"])
        for q, p, r, t1 in results:
            writer.writerow([q, f"{p:.4f}", f"{r:.4f}", t1])

    # 9. Compute aggregate statistics (simple macro average)
    avg_precision = sum(p for _, p, _, _ in results) / len(results)
    avg_recall = sum(r for _, _, r, _ in results) / len(results)
    avg_top1 = sum(t1 for _, _, _, t1 in results) / len(results)

    agg_path = os.path.join(os.getcwd(), "evaluation_summary.csv")
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["average_precision@3", f"{avg_precision:.4f}"])
        writer.writerow(["average_recall@3", f"{avg_recall:.4f}"])
        writer.writerow(["average_top1_accuracy", f"{avg_top1:.4f}"])

    print("=== Evaluation completed ===")
    print(f"Per-question CSV written to: {per_path}")
    print(f"Aggregate summary CSV written to: {agg_path}")
    print(f"Average Precision@3: {avg_precision:.4f}")
    print(f"Average Recall@3   : {avg_recall:.4f}")
    print(f"Average Top-1 Acc  : {avg_top1:.4f}")

if __name__ == "__main__":
    main()
