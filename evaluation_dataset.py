# Fixed evaluation dataset for DocuMind AI
# Each entry contains a natural language query and the set of relevant page numbers.
# The page numbers correspond to the PDF used in the baseline tests (sample_ai_lecture_notes.pdf).
# This dataset is deliberately small and deterministic for regression comparison.

EVALUATION_DATASET = [
    {
        "query": "When is the comprehensive final exam?",
        "relevant_pages": {3},
    },
    {
        "query": "What is the self attention formula?",
        "relevant_pages": {2},
    },
    {
        "query": "Who is the instructor for the AI lecture?",
        "relevant_pages": {1},
    },
    {
        "query": "What does FAISS provide?",
        "relevant_pages": {3},
    },
]
