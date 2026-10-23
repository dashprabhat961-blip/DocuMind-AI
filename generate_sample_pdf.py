"""
Generate a sample 3-page PDF with lecture notes for testing DocuMind AI.
Uses pure Python without external PDF creation dependencies.
"""

def generate_sample_pdf(filename: str = "sample_ai_lecture_notes.pdf"):
    # PDF 1.4 specification objects
    pages_text = [
        # Page 1: Course Overview & Foundations
        [
            "CS 501: Advanced Artificial Intelligence & Deep Learning",
            "Instructor: Dr. Alan Turing | Semester: Fall 2026",
            "",
            "Lecture 1: Foundations of Representation Learning",
            "In representation learning, machine learning algorithms discover feature representations",
            "directly from raw data. Traditional machine learning relies heavily on manual feature engineering,",
            "whereas deep neural networks construct hierarchical representations across multiple layers.",
            "",
            "Key Topic: Word Embeddings",
            "Word2Vec and GloVe introduced dense vector representations of words where spatial proximity",
            "reflects semantic similarity. In 384 or 768 dimensional vector spaces, words sharing similar",
            "contexts cluster together. This enables algebraic analogies such as vector('King') - vector('Man')",
            "+ vector('Woman') approx vector('Queen').",
            "",
            "Grading Policy: Midterm Exam accounts for 30%, Weekly Quizzes account for 20%, and the Final",
            "Capstone Project accounts for 50% of the overall course grade."
        ],
        # Page 2: Transformers & Attention Mechanisms
        [
            "Lecture 2: The Transformer Architecture and Self-Attention",
            "",
            "Self-Attention Formula:",
            "Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V",
            "Where Q is Queries, K is Keys, V is Values, and d_k is the dimension of the keys.",
            "The scaling factor sqrt(d_k) prevents the dot products from growing excessively large,",
            "which would push softmax into regions with tiny gradients (gradient vanishing).",
            "",
            "Multi-Head Attention:",
            "Multi-Head Attention enables the model to jointly attend to information from different representation",
            "subspaces at different positions. Rather than calculating a single attention distribution,",
            "it projects queries, keys, and values h times with learned linear projections.",
            "",
            "Important Date: Capstone Project Proposals are strictly due on Friday, November 14th at 11:59 PM.",
            "Late submissions incur a penalty of 10% per calendar day."
        ],
        # Page 3: Retrieval-Augmented Generation (RAG) & Vector Databases
        [
            "Lecture 3: Retrieval-Augmented Generation (RAG) Systems",
            "",
            "Limitations of Pure Parametric LLMs:",
            "Standard Large Language Models suffer from knowledge cutoff dates, lack of private domain data,",
            "and hallucinations. Retrieval-Augmented Generation addresses these drawbacks by decoupling",
            "parametric memory (the neural network weights) from non-parametric external memory.",
            "",
            "Vector Search with FAISS:",
            "Facebook AI Similarity Search (FAISS) provides optimized C++ implementations of nearest neighbor",
            "search. Using IndexFlatIP with L2-normalized vector embeddings calculates exact Cosine Similarity.",
            "Chunking strategy is critical: typical chunks range between 500 to 1000 characters with 10% to 20%",
            "sliding overlap to prevent fragmentation at semantic boundaries.",
            "",
            "Final Exam Date: The comprehensive final exam will take place on December 18th, from 9:00 AM to 12:00 PM",
            "in Hall B. Bring a valid student ID card and a scientific calculator."
        ]
    ]

    objects = []
    
    def add_object(content):
        objects.append(content)
        return len(objects)

    # 1: Catalog (will set later)
    # 2: Pages (will set later)
    # 3: Font
    font_id = 3
    page_ids = []
    content_ids = []

    # We will assemble cleanly:
    pdf_objs = {}
    pdf_objs[1] = "<< /Type /Catalog /Pages 2 0 R >>"
    pdf_objs[2] = "<< /Type /Pages /Kids [4 0 R 6 0 R 8 0 R] /Count 3 >>"
    pdf_objs[3] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    for i in range(3):
        p_obj_id = 4 + (i * 2)
        c_obj_id = p_obj_id + 1
        pdf_objs[p_obj_id] = f"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> >> /MediaBox [0 0 612 792] /Contents {c_obj_id} 0 R >>"
        
        # Stream content
        stream_lines = ["BT", "/F1 11 Tf", "50 740 Td", "16 TL"]
        for line in pages_text[i]:
            safe_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream_lines.append(f"({safe_line}) '")
        stream_lines.append("ET")
        stream_str = "\n".join(stream_lines)
        pdf_objs[c_obj_id] = f"<< /Length {len(stream_str.encode('latin1'))} >>\nstream\n{stream_str}\nendstream"

    # Assemble complete PDF file with xref table
    out = ["%PDF-1.4\n"]
    xref_offsets = {}
    
    # Calculate byte offsets
    current_offset = len(out[0].encode("latin1"))
    
    for obj_id in sorted(pdf_objs.keys()):
        xref_offsets[obj_id] = current_offset
        obj_text = f"{obj_id} 0 obj\n{pdf_objs[obj_id]}\nendobj\n"
        out.append(obj_text)
        current_offset += len(obj_text.encode("latin1"))

    # Xref table
    xref_start = current_offset
    total_objs = max(pdf_objs.keys()) + 1
    out.append(f"xref\n0 {total_objs}\n")
    out.append("0000000000 65535 f \n")
    for obj_id in range(1, total_objs):
        offset = xref_offsets[obj_id]
        out.append(f"{offset:010d} 00000 n \n")

    # Trailer
    out.append(f"trailer\n<< /Size {total_objs} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n")

    final_content = "".join(out)
    with open(filename, "wb") as f:
        f.write(final_content.encode("latin1"))
    print(f"Generated test PDF '{filename}' ({len(final_content)} bytes, 3 pages)")

if __name__ == "__main__":
    generate_sample_pdf()
