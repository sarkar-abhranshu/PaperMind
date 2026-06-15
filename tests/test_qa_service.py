from app.core.state import state
from app.services.embedding_engine import embed_texts
from app.config import settings
from app.services.qa_service import answer_question_with_sections


def setup_function() -> None:
    state.clear()


def test_qa_insufficient_context_when_filter_excludes_sections() -> None:
    embedding = embed_texts(["Results describe higher accuracy."], settings.embedding_dim)[0]
    state.vdb.upsert(
        [
            {
                "chunk_id": "c1",
                "paper_id": "p1",
                "section": "results",
                "chunk_index": 0,
                "text": "Results describe higher accuracy.",
                "embedding": embedding,
            }
        ]
    )

    resp = answer_question_with_sections("What is the method?", ["p1"], ["method"])
    assert resp["grounded"] is False
    assert resp["error"]["code"] == "E005"
