import json
import sqlite3
import threading
from typing import Optional

from app.models.schemas import Chunk, IngestedPaper

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    user_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    PRIMARY KEY (user_id, paper_id)
);

CREATE TABLE IF NOT EXISTS sections (
    user_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    section_name TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (user_id, paper_id, section_name)
);

CREATE TABLE IF NOT EXISTS chunks (
    user_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    section TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding TEXT DEFAULT '[]',
    PRIMARY KEY (user_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS paper_embeddings (
    user_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    embedding TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (user_id, paper_id)
);

CREATE TABLE IF NOT EXISTS selected_papers (
    user_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    PRIMARY KEY (user_id, paper_id)
);

CREATE TABLE IF NOT EXISTS conversations (
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    messages TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (user_id, conversation_id)
);
"""


class UserStateStore:
    def __init__(self, user_id: str, db_path: str = "./data/user_state.db") -> None:
        self._user_id = user_id
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # ---- papers ----

    def get_all_papers(self) -> dict[str, IngestedPaper]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT paper_id, filename, raw_text FROM papers WHERE user_id = ?",
                (self._user_id,),
            )
            return {
                row["paper_id"]: IngestedPaper(
                    paper_id=row["paper_id"],
                    filename=row["filename"],
                    raw_text=row["raw_text"],
                )
                for row in cursor.fetchall()
            }

    def get_paper(self, paper_id: str) -> Optional[IngestedPaper]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT paper_id, filename, raw_text FROM papers WHERE user_id = ? AND paper_id = ?",
                (self._user_id, paper_id),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return IngestedPaper(
                paper_id=row["paper_id"],
                filename=row["filename"],
                raw_text=row["raw_text"],
            )

    def add_paper(self, paper_id: str, paper: IngestedPaper) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO papers (user_id, paper_id, filename, raw_text) VALUES (?, ?, ?, ?)",
                (self._user_id, paper_id, paper.filename, paper.raw_text),
            )
            self._conn.commit()

    def remove_paper(self, paper_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM papers WHERE user_id = ? AND paper_id = ?",
                (self._user_id, paper_id),
            )
            self._conn.commit()

    # ---- sections ----

    def get_all_sections(self) -> dict[str, dict[str, str]]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT paper_id, section_name, content FROM sections WHERE user_id = ?",
                (self._user_id,),
            )
            result: dict[str, dict[str, str]] = {}
            for row in cursor.fetchall():
                pid = row["paper_id"]
                if pid not in result:
                    result[pid] = {}
                result[pid][row["section_name"]] = row["content"]
            return result

    def get_sections(self, paper_id: str) -> Optional[dict[str, str]]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT section_name, content FROM sections WHERE user_id = ? AND paper_id = ?",
                (self._user_id, paper_id),
            )
            rows = cursor.fetchall()
            if not rows:
                return None
            return {row["section_name"]: row["content"] for row in rows}

    def add_sections(self, paper_id: str, sections: dict[str, str]) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM sections WHERE user_id = ? AND paper_id = ?",
                (self._user_id, paper_id),
            )
            for section_name, content in sections.items():
                self._conn.execute(
                    "INSERT INTO sections (user_id, paper_id, section_name, content) VALUES (?, ?, ?, ?)",
                    (self._user_id, paper_id, section_name, content),
                )
            self._conn.commit()

    # ---- chunks ----

    def get_all_chunks(self) -> dict[str, list[Chunk]]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT chunk_id, paper_id, section, chunk_index, text, embedding FROM chunks WHERE user_id = ?",
                (self._user_id,),
            )
            result: dict[str, list[Chunk]] = {}
            for row in cursor.fetchall():
                pid = row["paper_id"]
                if pid not in result:
                    result[pid] = []
                emb: list[float] = json.loads(row["embedding"]) if row["embedding"] else []
                result[pid].append(
                    Chunk(
                        chunk_id=row["chunk_id"],
                        paper_id=pid,
                        section=row["section"],
                        chunk_index=row["chunk_index"],
                        text=row["text"],
                        embedding=emb,
                    )
                )
            return result

    def get_chunks(self, paper_id: str) -> Optional[list[Chunk]]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT chunk_id, paper_id, section, chunk_index, text, embedding FROM chunks WHERE user_id = ? AND paper_id = ?",
                (self._user_id, paper_id),
            )
            rows = cursor.fetchall()
            if not rows:
                return None
            result: list[Chunk] = []
            for row in rows:
                emb: list[float] = json.loads(row["embedding"]) if row["embedding"] else []
                result.append(
                    Chunk(
                        chunk_id=row["chunk_id"],
                        paper_id=row["paper_id"],
                        section=row["section"],
                        chunk_index=row["chunk_index"],
                        text=row["text"],
                        embedding=emb,
                    )
                )
            return result

    def add_chunks(self, paper_id: str, chunks: list[Chunk]) -> None:
        with self._lock:
            for chunk in chunks:
                self._conn.execute(
                    "INSERT OR REPLACE INTO chunks (user_id, chunk_id, paper_id, section, chunk_index, text, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        self._user_id,
                        chunk.chunk_id,
                        paper_id,
                        chunk.section,
                        chunk.chunk_index,
                        chunk.text,
                        json.dumps(chunk.embedding),
                    ),
                )
            self._conn.commit()

    # ---- paper embeddings ----

    def get_all_embeddings(self) -> dict[str, list[float]]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT paper_id, embedding FROM paper_embeddings WHERE user_id = ?",
                (self._user_id,),
            )
            return {
                row["paper_id"]: json.loads(row["embedding"]) if row["embedding"] else []
                for row in cursor.fetchall()
            }

    def add_embedding(self, paper_id: str, embedding: list[float]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO paper_embeddings (user_id, paper_id, embedding) VALUES (?, ?, ?)",
                (self._user_id, paper_id, json.dumps(embedding)),
            )
            self._conn.commit()

    # ---- selected papers ----

    def get_selected_papers(self) -> set[str]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT paper_id FROM selected_papers WHERE user_id = ?",
                (self._user_id,),
            )
            return {row["paper_id"] for row in cursor.fetchall()}

    def add_selected_paper(self, paper_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO selected_papers (user_id, paper_id) VALUES (?, ?)",
                (self._user_id, paper_id),
            )
            self._conn.commit()

    def remove_selected_paper(self, paper_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM selected_papers WHERE user_id = ? AND paper_id = ?",
                (self._user_id, paper_id),
            )
            self._conn.commit()

    def toggle_paper_selection(self, paper_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT 1 FROM selected_papers WHERE user_id = ? AND paper_id = ?",
                (self._user_id, paper_id),
            )
            if cursor.fetchone():
                self._conn.execute(
                    "DELETE FROM selected_papers WHERE user_id = ? AND paper_id = ?",
                    (self._user_id, paper_id),
                )
                self._conn.commit()
                return False
            else:
                self._conn.execute(
                    "INSERT INTO selected_papers (user_id, paper_id) VALUES (?, ?)",
                    (self._user_id, paper_id),
                )
                self._conn.commit()
                return True

    def set_selected_papers(self, paper_ids: list[str]) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM selected_papers WHERE user_id = ?",
                (self._user_id,),
            )
            for pid in paper_ids:
                self._conn.execute(
                    "INSERT INTO selected_papers (user_id, paper_id) VALUES (?, ?)",
                    (self._user_id, pid),
                )
            self._conn.commit()

    # ---- conversations ----

    def get_conversation(self, conversation_id: str) -> list[dict]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT messages FROM conversations WHERE user_id = ? AND conversation_id = ?",
                (self._user_id, conversation_id),
            )
            row = cursor.fetchone()
            if row is None:
                return []
            return json.loads(row["messages"])

    def append_to_conversation(self, conversation_id: str, role: str, content: str) -> None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT messages FROM conversations WHERE user_id = ? AND conversation_id = ?",
                (self._user_id, conversation_id),
            )
            row = cursor.fetchone()
            messages: list[dict] = json.loads(row["messages"]) if row else []
            messages.append({"role": role, "content": content})
            self._conn.execute(
                "INSERT OR REPLACE INTO conversations (user_id, conversation_id, messages) VALUES (?, ?, ?)",
                (self._user_id, conversation_id, json.dumps(messages)),
            )
            self._conn.commit()

    def clear_conversation(self, conversation_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM conversations WHERE user_id = ? AND conversation_id = ?",
                (self._user_id, conversation_id),
            )
            self._conn.commit()

    # ---- clear ----

    def clear(self) -> None:
        with self._lock:
            for table in ("papers", "sections", "chunks", "paper_embeddings", "selected_papers", "conversations"):
                self._conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (self._user_id,))
            self._conn.commit()
