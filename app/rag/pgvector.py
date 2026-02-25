# app/rag/vectordb_pg.py
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import psycopg2
from psycopg2.extras import Json

from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

log = logging.getLogger("rag.vectordb_pg")

def _to_pgvector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"

@dataclass
class PgVectorStore:
    dsn: str
    table: str
    embed_model: str

    def upsert_chunks(self, doc_id: str, chunks: List[Document]) -> int:
        return self.upsert_documents(docs=chunks, doc_id=doc_id)

    def delete_doc(self, doc_id: str) -> int:
        return self.delete_by_doc_id(doc_id)

    def _connect(self):
        return psycopg2.connect(self.dsn)

    def ensure_schema(self, dim: int):
        # Если ты делаешь миграции Alembic — этот метод можно не использовать.
        sql = f"""
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE IF NOT EXISTS {self.table} (
          id           TEXT PRIMARY KEY,
          doc_id       TEXT NOT NULL,
          chunk_index  INT  NOT NULL,
          content      TEXT NOT NULL,
          metadata     JSONB NOT NULL DEFAULT '{{}}',
          embedding    vector({dim}) NOT NULL,
          created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS {self.table}_doc_id_idx ON {self.table}(doc_id);
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()

    def upsert_documents(
        self,
        docs: List[Document],
        doc_id: str,
        ids: Optional[List[str]] = None,
    ) -> int:
        """
        Upsert chunks into Postgres. `ids` should be stable (doc_id:chunk_index, etc).
        """
        embeddings_client = OllamaEmbeddings(model=self.embed_model)

        texts = [d.page_content for d in docs]
        vectors = embeddings_client.embed_documents(texts)  # List[List[float]]

        if not vectors:
            return 0

        dim = len(vectors[0])
        self.ensure_schema(dim=dim)

        if ids is None:
            ids = [f"{doc_id}:{i}" for i in range(len(docs))]

        rows = []
        for i, d in enumerate(docs):
            md = dict(d.metadata or {})
            rows.append(
                (
                    ids[i],
                    doc_id,
                    i,
                    d.page_content,
                    Json(md),
                    _to_pgvector_literal(vectors[i]),
                )
            )

        sql = f"""
        INSERT INTO {self.table} (id, doc_id, chunk_index, content, metadata, embedding)
        VALUES (%s, %s, %s, %s, %s, %s::vector)
        ON CONFLICT (id)
        DO UPDATE SET
          content = EXCLUDED.content,
          metadata = EXCLUDED.metadata,
          embedding = EXCLUDED.embedding,
          updated_at = now();
        """

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
            conn.commit()

        log.info("Upserted %d chunks into %s (doc_id=%s)", len(rows), self.table, doc_id)
        return len(rows)

    def delete_by_doc_id(self, doc_id: str) -> int:
        sql = f"DELETE FROM {self.table} WHERE doc_id = %s"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (doc_id,))
                n = cur.rowcount
            conn.commit()
        return n

    def similarity_search(
        self,
        query: str,
        k: int = 6,
        filters: Optional[Dict[str, Any]] = None,
        metric: str = "cosine",
    ) -> List[Document]:
        """
        metric:
          - "cosine" uses <=> (cosine distance)
          - "l2" uses <-> (euclidean distance)
        filters: SQL-level constraints via metadata/doc_id/etc.
        """
        embeddings_client = OllamaEmbeddings(model=self.embed_model)
        qv = embeddings_client.embed_query(query)
        qv_lit = _to_pgvector_literal(qv)
        dim = len(qv)
        self.ensure_schema(dim=dim)

        if metric == "cosine":
            dist_op = "<=>"
        elif metric == "l2":
            dist_op = "<->"
        else:
            raise ValueError("metric must be 'cosine' or 'l2'")

        where = []
        params: List[Any] = []

        # Пример фильтра по doc_id
        if filters and "doc_id" in filters:
            where.append("doc_id = %s")
            params.append(filters["doc_id"])

        # Пример фильтра по metadata (JSONB): metadata->>'file_path' = '...'
        if filters and "file_path" in filters:
            where.append("metadata->>'file_path' = %s")
            params.append(filters["file_path"])

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        # Важно: psycopg2 умеет писать list[float] в pgvector как array-like,
        # но иногда нужен адаптер. В большинстве случаев работает "as is".
        sql = f"""
        SELECT content, metadata
        FROM {self.table}
        {where_sql}
        ORDER BY embedding {dist_op} %s::vector
        LIMIT %s;
        """
        params.append(qv_lit)
        params.append(k)

        out: List[Document] = []
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                for content, md in cur.fetchall():
                    out.append(Document(page_content=content, metadata=md or {}))
        return out
