'''Metrics utilities for evaluation benchmarks.

Provides functions to compute Precision@k, Recall@k, and Top‑1 evidence‑page accuracy.
All functions expect lists of retrieved items (each item can be any hashable, e.g., page numbers)
and a set/list of ground‑truth relevant items.
Edge‑case handling:
* If k is 0, precision and recall return 0.0.
* If there are no relevant items, recall is defined as 1.0 (nothing to miss) and precision follows normal definition.
* Top‑1 accuracy returns 0 when retrieved list is empty.
''' 

from typing import List, Set, Tuple

def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Compute Precision@k.

    Args:
        retrieved: Ordered list of retrieved identifiers (e.g., page numbers).
        relevant: Set of ground‑truth relevant identifiers.
        k: Number of top results to consider.
    Returns:
        Precision value in [0, 1].
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hit_count = sum(1 for r in top_k if r in relevant)
    return hit_count / min(k, len(top_k))

def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Compute Recall@k.

    Recall is defined as the proportion of *all* relevant items that appear in the top‑k list.
    If there are no relevant items, recall is defined as 1.0 (nothing to miss).
    """
    if not relevant:
        return 1.0
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hit_count = sum(1 for r in top_k if r in relevant)
    return hit_count / len(relevant)

def top1_page_accuracy(retrieved: List[int], relevant: Set[int]) -> int:
    """Return 1 if the first retrieved page is relevant, else 0.
    An empty retrieved list yields 0.
    """
    if not retrieved:
        return 0
    return 1 if retrieved[0] in relevant else 0

def evaluate_queries(
    queries: List[Tuple[str, Set[int]]],
    retrieve_fn,
    k: int = 3,
) -> List[Tuple[str, float, float, int]]:
    """Run evaluation on a list of (query, relevant_pages) tuples.

    retrieve_fn should accept a query string and return a list of DocumentChunk objects.
    The function extracts the ``page_number`` attribute for metric calculation.
    Returns a list of (query, precision, recall, top1) tuples.
    """
    results = []
    for query, relevant_pages in queries:
        chunks = retrieve_fn(query)
        retrieved_pages = [c.page_number for c, _ in chunks]
        p = precision_at_k(retrieved_pages, relevant_pages, k)
        r = recall_at_k(retrieved_pages, relevant_pages, k)
        t1 = top1_page_accuracy(retrieved_pages, relevant_pages)
        results.append((query, p, r, t1))
    return results
