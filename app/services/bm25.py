import math
import re


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_DEFAULT_K1 = 1.5
_DEFAULT_B = 0.75


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score - min_score <= 1e-9:
        return [0.0 for _ in scores]
    return [(s - min_score) / (max_score - min_score) for s in scores]


def compute_bm25_scores(
    query: str,
    documents: list[str],
    k1: float = _DEFAULT_K1,
    b: float = _DEFAULT_B,
) -> list[float]:
    if not documents:
        return []

    query_terms = _tokenize(query)
    if not query_terms:
        return [0.0 for _ in documents]

    doc_tokens = [_tokenize(doc) for doc in documents]
    doc_lengths = [len(tokens) for tokens in doc_tokens]
    avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    if avgdl <= 0:
        return [0.0 for _ in documents]

    unique_terms = set(query_terms)
    doc_freq: dict[str, int] = {term: 0 for term in unique_terms}

    for tokens in doc_tokens:
        token_set = set(tokens)
        for term in unique_terms:
            if term in token_set:
                doc_freq[term] += 1

    scores: list[float] = []
    total_docs = len(documents)
    for tokens, doc_len in zip(doc_tokens, doc_lengths, strict=True):
        score = 0.0
        term_counts: dict[str, int] = {}
        for token in tokens:
            if token in unique_terms:
                term_counts[token] = term_counts.get(token, 0) + 1

        for term in unique_terms:
            df = doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            tf = term_counts.get(term, 0)
            if tf == 0:
                continue
            denom = tf + k1 * (1 - b + b * (doc_len / avgdl))
            score += idf * (tf * (k1 + 1)) / denom
        scores.append(score)

    return scores


def apply_hybrid_scores(
    query: str,
    rows: list[dict[str, object]],
    alpha: float,
) -> list[dict[str, object]]:
    if not rows:
        return rows

    documents = [str(row.get("text", "")) for row in rows]
    bm25_scores = compute_bm25_scores(query, documents)
    dense_scores = [float(row.get("score", 0.0) or 0.0) for row in rows]

    dense_norm = _normalize_scores(dense_scores)
    bm25_norm = _normalize_scores(bm25_scores)

    fused: list[dict[str, object]] = []
    for row, bm25, d_norm, b_norm in zip(rows, bm25_scores, dense_norm, bm25_norm, strict=True):
        hybrid_score = alpha * d_norm + (1.0 - alpha) * b_norm
        row_copy = dict(row)
        row_copy["bm25_score"] = float(bm25)
        row_copy["hybrid_score"] = float(hybrid_score)
        fused.append(row_copy)

    fused.sort(key=lambda r: r.get("hybrid_score", float("-inf")), reverse=True)
    return fused
