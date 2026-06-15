from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.user_context import get_current_user_id


class ChromaVectorDB:
    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        collection_name: str = "papermind_chunks",
    ) -> None:
        base_dir = Path(persist_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(base_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, rows: list[dict]) -> None:
        if not rows:
            return

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict] = []
        documents: list[str] = []

        user_id = get_current_user_id()

        for row in rows:
            ids.append(str(row["chunk_id"]))
            embeddings.append(list(row["embedding"]))
            metadatas.append(
                {
                    "paper_id": row["paper_id"],
                    "section": row["section"],
                    "chunk_index": row["chunk_index"],
                    "user_id": user_id,
                }
            )
            documents.append(str(row["text"]))

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    def search(
        self,
        query_vector: list[float],
        k: int,
        paper_ids: Optional[list[str]] = None,
        sections: Optional[list[str]] = None,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        if k <= 0:
            return []

        uid = user_id or get_current_user_id()
        conditions: list[dict] = [{"user_id": uid}]
        if paper_ids:
            conditions.append({"paper_id": {"$in": paper_ids}})
        if sections:
            conditions.append({"section": {"$in": sections}})

        where_clause: Optional[dict] = {"$and": conditions} if len(conditions) > 1 else conditions[0]

        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=where_clause,
            include=["metadatas", "documents", "distances"],
        )

        ids = (results.get("ids") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        rows: list[dict] = []
        for idx, chunk_id in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            distance = distances[idx] if idx < len(distances) else 0.0
            score = float(1.0 - distance)
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "paper_id": metadata.get("paper_id"),
                    "section": metadata.get("section"),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "text": documents[idx] if idx < len(documents) else "",
                    "score": score,
                }
            )

        return rows

    def clear_user_data(self, user_id: Optional[str] = None) -> None:
        uid = user_id or get_current_user_id()
        self._collection.delete(where={"user_id": uid})

    def reset(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(name=name)
