import logging
import threading

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_model: CrossEncoder | None = None
_model_name: str | None = None


def _get_model(model_name: str) -> CrossEncoder | None:
    global _model, _model_name
    with _model_lock:
        if _model is not None and _model_name == model_name:
            return _model

        try:
            _model = CrossEncoder(model_name)
            _model_name = model_name
            return _model
        except Exception as exc:
            logger.warning("Cross-encoder load failed for %s: %s", model_name, exc)
            return None


def rerank_rows(
    question: str,
    rows: list[dict[str, object]],
    top_n: int,
    model_name: str,
) -> list[dict[str, object]]:
    if not rows or top_n <= 0:
        return rows

    model = _get_model(model_name)
    if model is None:
        return rows

    capped = min(top_n, len(rows))
    pairs = [[question, str(row.get("text", ""))] for row in rows[:capped]]

    try:
        scores = model.predict(pairs)
    except Exception as exc:
        logger.warning("Cross-encoder rerank failed: %s", exc)
        return rows

    scored = []
    for row, score in zip(rows[:capped], scores):
        row_copy = dict(row)
        row_copy["rerank_score"] = float(score)
        scored.append(row_copy)

    scored.sort(key=lambda r: r.get("rerank_score", float("-inf")), reverse=True)
    return scored + rows[capped:]
