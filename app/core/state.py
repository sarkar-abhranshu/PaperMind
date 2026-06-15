from typing import Optional

from app.core.user_context import get_current_user_id
from app.core.user_state import UserStateStore
from app.models.schemas import Chunk, IngestedPaper
from app.services.vector_db import ChromaVectorDB


class AppState:
    def __init__(self, db_path: str = "./data/user_state.db") -> None:
        self._stores: dict[str, UserStateStore] = {}
        self._vdb = ChromaVectorDB()
        self._db_path = db_path

    def _get_store(self, user_id: str) -> UserStateStore:
        if user_id not in self._stores:
            self._stores[user_id] = UserStateStore(user_id, db_path=self._db_path)
        return self._stores[user_id]

    @property
    def papers(self) -> dict[str, IngestedPaper]:
        return self._get_store(get_current_user_id()).get_all_papers()

    def get_paper(self, paper_id: str, user_id: Optional[str] = None) -> Optional[IngestedPaper]:
        uid = user_id or get_current_user_id()
        return self._get_store(uid).get_paper(paper_id)

    def add_paper(self, paper_id: str, paper: IngestedPaper, user_id: Optional[str] = None) -> None:
        uid = user_id or get_current_user_id()
        self._get_store(uid).add_paper(paper_id, paper)

    def remove_paper(self, paper_id: str, user_id: Optional[str] = None) -> None:
        uid = user_id or get_current_user_id()
        self._get_store(uid).remove_paper(paper_id)

    @property
    def sections(self) -> dict[str, dict[str, str]]:
        return self._get_store(get_current_user_id()).get_all_sections()

    def get_sections(self, paper_id: str, user_id: Optional[str] = None) -> Optional[dict[str, str]]:
        uid = user_id or get_current_user_id()
        return self._get_store(uid).get_sections(paper_id)

    def add_sections(self, paper_id: str, sections: dict[str, str], user_id: Optional[str] = None) -> None:
        uid = user_id or get_current_user_id()
        self._get_store(uid).add_sections(paper_id, sections)

    @property
    def chunks(self) -> dict[str, list[Chunk]]:
        return self._get_store(get_current_user_id()).get_all_chunks()

    def get_chunks(self, paper_id: str, user_id: Optional[str] = None) -> Optional[list[Chunk]]:
        uid = user_id or get_current_user_id()
        return self._get_store(uid).get_chunks(paper_id)

    def add_chunks(self, paper_id: str, chunks: list[Chunk], user_id: Optional[str] = None) -> None:
        uid = user_id or get_current_user_id()
        self._get_store(uid).add_chunks(paper_id, chunks)

    @property
    def paper_embeddings(self) -> dict[str, list[float]]:
        return self._get_store(get_current_user_id()).get_all_embeddings()

    def add_embedding(self, paper_id: str, embedding: list[float], user_id: Optional[str] = None) -> None:
        uid = user_id or get_current_user_id()
        self._get_store(uid).add_embedding(paper_id, embedding)

    @property
    def vdb(self) -> ChromaVectorDB:
        return self._vdb

    @property
    def selected_papers(self) -> set[str]:
        return self._get_store(get_current_user_id()).get_selected_papers()

    def add_selected_paper(self, paper_id: str, user_id: Optional[str] = None) -> None:
        uid = user_id or get_current_user_id()
        self._get_store(uid).add_selected_paper(paper_id)

    def remove_selected_paper(self, paper_id: str, user_id: Optional[str] = None) -> None:
        uid = user_id or get_current_user_id()
        self._get_store(uid).remove_selected_paper(paper_id)

    def toggle_paper_selection(self, paper_id: str, user_id: Optional[str] = None) -> bool:
        uid = user_id or get_current_user_id()
        return self._get_store(uid).toggle_paper_selection(paper_id)

    def set_selected_papers(self, paper_ids: list[str], user_id: Optional[str] = None) -> None:
        uid = user_id or get_current_user_id()
        self._get_store(uid).set_selected_papers(paper_ids)

    def get_conversation(self, conversation_id: str, user_id: Optional[str] = None) -> list[dict]:
        uid = user_id or get_current_user_id()
        return self._get_store(uid).get_conversation(conversation_id)

    def append_to_conversation(
        self, conversation_id: str, role: str, content: str, user_id: Optional[str] = None
    ) -> None:
        uid = user_id or get_current_user_id()
        self._get_store(uid).append_to_conversation(conversation_id, role, content)

    def clear_conversation(self, conversation_id: str, user_id: Optional[str] = None) -> None:
        uid = user_id or get_current_user_id()
        self._get_store(uid).clear_conversation(conversation_id)

    def clear(self, user_id: Optional[str] = None) -> None:
        uid = user_id or get_current_user_id()
        self._get_store(uid).clear()
        self._vdb.reset()


state = AppState()
