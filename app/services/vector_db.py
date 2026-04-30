from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings


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

    def upsert(self, rows: list[dict[str, object]]) -> None:
        if not rows:
            return

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, object]] = []
        documents: list[str] = []

        for row in rows:
            ids.append(str(row["chunk_id"]))
            embeddings.append(list(row["embedding"]))
            metadatas.append(
                {
                    "paper_id": row["paper_id"],
                    "section": row["section"],
                    "chunk_index": row["chunk_index"],
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
        paper_ids: list[str] | None = None,
        sections: list[str] | None = None,
    ) -> list[dict[str, object]]:
        if k <= 0:
            return []

        where: dict[str, object] = {}
        if paper_ids:
            where["paper_id"] = {"$in": paper_ids}
        if sections:
            where["section"] = {"$in": sections}

        where_clause = where if where else None

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
        embeddings: list[list[float]] = []

        rows: list[dict[str, object]] = []
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

    def reset(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(name=name)
