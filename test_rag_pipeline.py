"""
Test suite for DocuMind AI RAG Pipeline.
Covers whitespace normalization, chunk-boundary integrity, overlap validation,
metadata preservation, empty/scanned PDF handling, and retrieval regression.
"""

import io
import sys
import unittest
from typing import List

from app import (
    normalize_extracted_text,
    extract_text_from_pdf,
    chunk_text,
    get_embedding_model,
    build_faiss_index,
    retrieve_relevant_chunks,
    DocumentChunk,
)


def create_in_memory_pdf(pages_text: List[List[str]]) -> io.BytesIO:
    """Creates a minimal in-memory PDF 1.4 for testing without external PDF tools."""
    pdf_objs = {}
    total_pages = len(pages_text)

    # 1: Catalog, 2: Pages, 3: Font
    page_ids = [4 + (i * 2) for i in range(total_pages)]
    pdf_objs[1] = "<< /Type /Catalog /Pages 2 0 R >>"
    pdf_objs[2] = f"<< /Type /Pages /Kids [{' '.join(f'{pid} 0 R' for pid in page_ids)}] /Count {total_pages} >>"
    pdf_objs[3] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    for i, lines in enumerate(pages_text):
        p_obj_id = page_ids[i]
        c_obj_id = p_obj_id + 1
        pdf_objs[p_obj_id] = (
            f"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> >> "
            f"/MediaBox [0 0 612 792] /Contents {c_obj_id} 0 R >>"
        )
        stream_lines = ["BT", "/F1 11 Tf", "50 740 Td", "16 TL"]
        for line in lines:
            safe_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream_lines.append(f"({safe_line}) '")
        stream_lines.append("ET")
        stream_str = "\n".join(stream_lines)
        pdf_objs[c_obj_id] = f"<< /Length {len(stream_str.encode('latin1'))} >>\nstream\n{stream_str}\nendstream"

    # Assemble complete PDF file
    out = ["%PDF-1.4\n"]
    xref = {}
    pos = len(out[0].encode("latin1"))
    all_keys = sorted(pdf_objs.keys())
    for k in all_keys:
        xref[k] = pos
        block = f"{k} 0 obj\n{pdf_objs[k]}\nendobj\n"
        out.append(block)
        pos += len(block.encode("latin1"))

    xref_pos = pos
    out.append(f"xref\n0 {len(all_keys) + 1}\n0000000000 65535 f \n")
    for k in all_keys:
        out.append(f"{xref[k]:010d} 00000 n \n")

    out.append(f"trailer\n<< /Size {len(all_keys) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF")
    return io.BytesIO("".join(out).encode("latin1"))


class TestDocuMindRAGPipeline(unittest.TestCase):
    """Automated unit test cases for DocuMind AI pipeline."""

    def test_whitespace_normalization_and_line_repair(self):
        """Tests that whitespace normalization conservatively repairs lines while preserving structures."""
        # 1. Heading preservation
        heading_text = "Module 3: Advanced Applications\nSection 3.1 Medical Imaging\nDetailed description here."
        norm_h = normalize_extracted_text(heading_text)
        self.assertIn("Module 3: Advanced Applications\nSection 3.1 Medical Imaging", norm_h)

        # 2. Bullet list preservation
        bullet_text = "Introduction:\n● Brain tumor detection\n● Stroke diagnosis\n➢ CT Scan analysis"
        norm_b = normalize_extracted_text(bullet_text)
        self.assertIn("● Brain tumor detection\n● Stroke diagnosis\n➢ CT Scan analysis", norm_b)

        # 3. Equation preservation
        eq_text = (
            "Self-Attention Formula:\n"
            "Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V\n"
            "Where Q is Queries, K is Keys, V is Values."
        )
        norm_eq = normalize_extracted_text(eq_text)
        self.assertIn("Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V", norm_eq)

        # 4. Soft hyphen and wrapped words repair
        hyphen_text = "In repre-\nsentation learning, algorithms discover\nfeatures directly."
        norm_hyphen = normalize_extracted_text(hyphen_text)
        self.assertIn("representation learning, algorithms discover features directly.", norm_hyphen)

        # 5. Compound prefix preservation
        compound_text = "MRI is a non-\ninvasive procedure."
        norm_compound = normalize_extracted_text(compound_text)
        self.assertIn("non-invasive procedure.", norm_compound)

        # 6. Non-breaking spaces and interleaved newline spaces
        messy_text = "Word1\u00a0Word2\u200b.\n \n \n \nWord3   Word4."
        norm_messy = normalize_extracted_text(messy_text)
        self.assertEqual(norm_messy, "Word1 Word2.\n\nWord3 Word4.")

    def test_chunk_overlap_validation(self):
        """Tests that invalid chunking parameters raise ValueError."""
        dummy_pages = [{"page_number": 1, "text": "Valid document content."}]

        # chunk_size must be positive
        with self.assertRaises(ValueError):
            chunk_text(dummy_pages, "test.pdf", chunk_size=0, chunk_overlap=0)
        with self.assertRaises(ValueError):
            chunk_text(dummy_pages, "test.pdf", chunk_size=-100, chunk_overlap=10)

        # chunk_overlap must be non-negative
        with self.assertRaises(ValueError):
            chunk_text(dummy_pages, "test.pdf", chunk_size=500, chunk_overlap=-10)

        # chunk_overlap must be strictly less than chunk_size
        with self.assertRaises(ValueError):
            chunk_text(dummy_pages, "test.pdf", chunk_size=500, chunk_overlap=500)
        with self.assertRaises(ValueError):
            chunk_text(dummy_pages, "test.pdf", chunk_size=500, chunk_overlap=600)

        # Valid boundary: chunk_overlap = 0 works cleanly
        chunks_zero_overlap = chunk_text(dummy_pages, "test.pdf", chunk_size=500, chunk_overlap=0)
        self.assertEqual(len(chunks_zero_overlap), 1)

    def test_chunk_boundary_integrity(self):
        """Tests that chunking never cuts words in half and guarantees forward progress."""
        # Repeating sentence with known distinct words
        sentence = ("The quick brown fox jumps over the lazy dog. " * 30).strip()
        chunks = chunk_text([{"page_number": 1, "text": sentence}], "test.pdf", chunk_size=160, chunk_overlap=40)

        self.assertGreater(len(chunks), 1, "Expected multiple chunks for long sentence")
        valid_words = {"The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"}

        for i, c in enumerate(chunks):
            first_word = c.text.split()[0].rstrip(".,!?")
            last_word = c.text.split()[-1].rstrip(".,!?")
            self.assertIn(
                first_word,
                valid_words,
                f"Chunk {i} started with a severed word: '{first_word}' in '{c.text[:30]}'",
            )
            self.assertIn(
                last_word,
                valid_words,
                f"Chunk {i} ended with a severed word: '{last_word}' in '{c.text[-30:]}'",
            )

    def test_page_metadata_preservation(self):
        """Tests that page numbers and source names are accurately tracked, without dropping short text."""
        pages = [
            {"page_number": 1, "text": "Short Title"},
            {"page_number": 2, "text": "Paragraph on page two with some important discussion."},
            {"page_number": 3, "text": "Conclusion."},
        ]
        chunks = chunk_text(pages, "lecture_notes.pdf", chunk_size=500, chunk_overlap=50)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].page_number, 1)
        self.assertEqual(chunks[0].text, "Short Title")
        self.assertEqual(chunks[0].source_name, "lecture_notes.pdf")

        self.assertEqual(chunks[1].page_number, 2)
        self.assertEqual(chunks[2].page_number, 3)
        self.assertEqual(chunks[2].text, "Conclusion.")

    def test_empty_and_scanned_pdf(self):
        """Tests handling of empty pages, blank text, and non-selectable PDFs."""
        # 1. Empty pages data in chunk_text
        empty_chunks = chunk_text([], "test.pdf")
        self.assertEqual(len(empty_chunks), 0)

        whitespace_chunks = chunk_text(
            [{"page_number": 1, "text": ""}, {"page_number": 2, "text": "   \n\n  "}],
            "test.pdf",
        )
        self.assertEqual(len(whitespace_chunks), 0)

        # 2. In-memory PDF with no text stream (simulating scanned image / blank PDF)
        blank_pdf_stream = create_in_memory_pdf([[""]])
        pages, err = extract_text_from_pdf(blank_pdf_stream)
        self.assertEqual(pages, [])
        self.assertIn("No selectable text could be extracted", err)

    def test_retrieval_regression(self):
        """Runs the fixed baseline evaluation questions against sample lecture notes."""
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

        pages, err = extract_text_from_pdf(pdf_stream)
        self.assertFalse(err)
        self.assertEqual(len(pages), 3)

        chunks = chunk_text(pages, "sample_ai_lecture_notes.pdf", chunk_size=700, chunk_overlap=150)
        model = get_embedding_model()
        self.assertIsNotNone(model)

        index = build_faiss_index(chunks, model)
        self.assertEqual(index.ntotal, len(chunks))

        # Query 1: Final exam
        results1 = retrieve_relevant_chunks(
            "When is the comprehensive final exam?", index, chunks, model, top_k=2
        )
        self.assertGreater(len(results1), 0)
        retrieved_pages1 = [c.page_number for c, _ in results1]
        self.assertIn(3, retrieved_pages1)
        top_c1, score1 = results1[0]
        print(f"\n[Retrieval Benchmark] Query 1 Top Match: Page {top_c1.page_number} | Score: {score1:.4f}")

        # Query 2: Self-attention formula
        results2 = retrieve_relevant_chunks(
            "What is the self attention formula?", index, chunks, model, top_k=2
        )
        self.assertGreater(len(results2), 0)
        top_c2, score2 = results2[0]
        print(f"[Retrieval Benchmark] Query 2 Top Match: Page {top_c2.page_number} | Score: {score2:.4f}")
        self.assertEqual(top_c2.page_number, 2)


def run_all_tests():
    """Custom runner providing clear pass/fail counts and status reporting."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDocuMindRAGPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print("DOCUMIND AI TEST RUN SUMMARY")
    print("=" * 60)
    print(f"Total Tests Run : {result.testsRun}")
    print(f"Passed          : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures        : {len(result.failures)}")
    print(f"Errors          : {len(result.errors)}")
    print("=" * 60)

    if result.wasSuccessful():
        print("[SUCCESS] All pipeline regression and integrity tests PASSED!")
        return 0
    else:
        print("[FAILURE] Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
