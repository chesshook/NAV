"""Disk-backed knowledge store with optional SQLite FTS5 indexing.

JSONL imports are streamed in batches: a multi-gigabyte source file is never
read into RAM as one object. The database stores source/licence metadata.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Iterable


class KnowledgeStore:
    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        if path != ":memory:":
            Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-8192")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.fts_enabled = False
        self._create_schema()

    def _create_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                doc_id TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                answer TEXT NOT NULL,
                domain TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                licence TEXT NOT NULL DEFAULT '',
                keywords TEXT NOT NULL DEFAULT '[]',
                intents TEXT NOT NULL DEFAULT '[]',
                tags TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents(domain)")
        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts "
                "USING fts5(title, text, answer, keywords, tags)"
            )
            self.fts_enabled = True
        except sqlite3.OperationalError:
            # Some Python distributions ship SQLite without FTS5. A LIKE fallback
            # keeps the prototype usable, albeit slower on a large store.
            self.fts_enabled = False
        self.conn.commit()

    @staticmethod
    def _clean_record(record: dict) -> dict:
        required = ("id", "title")
        for field in required:
            if not str(record.get(field, "")).strip():
                raise ValueError(f"Knowledge record is missing required field: {field}")
        def clean_list(value):
            if value is None:
                return []
            if isinstance(value, str):
                return [value]
            if not isinstance(value, (list, tuple)):
                raise ValueError("keywords, intents and tags must be arrays or strings")
            return [str(item).strip() for item in value if str(item).strip()]
        return {
            "doc_id": str(record["id"]).strip(),
            "title": str(record["title"]).strip(),
            "text": str(record.get("text", "")).strip(),
            "answer": str(record.get("answer", "")).strip(),
            "domain": str(record.get("domain", "")).strip(),
            "source": str(record.get("source", "")).strip(),
            "licence": str(record.get("licence", record.get("license", ""))).strip(),
            "keywords": clean_list(record.get("keywords")),
            "intents": clean_list(record.get("intents")),
            "tags": clean_list(record.get("tags")),
        }

    def upsert_records(self, records: Iterable[dict]) -> int:
        count = 0
        with self.conn:
            for raw in records:
                rec = self._clean_record(raw)
                values = (
                    rec["doc_id"], rec["title"], rec["text"], rec["answer"],
                    rec["domain"], rec["source"], rec["licence"],
                    json.dumps(rec["keywords"], ensure_ascii=False),
                    json.dumps(rec["intents"], ensure_ascii=False),
                    json.dumps(rec["tags"], ensure_ascii=False),
                )
                self.conn.execute(
                    """
                    INSERT INTO documents
                    (doc_id,title,text,answer,domain,source,licence,keywords,intents,tags)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(doc_id) DO UPDATE SET
                      title=excluded.title, text=excluded.text, answer=excluded.answer,
                      domain=excluded.domain, source=excluded.source, licence=excluded.licence,
                      keywords=excluded.keywords, intents=excluded.intents, tags=excluded.tags
                    """,
                    values,
                )
                row = self.conn.execute("SELECT id FROM documents WHERE doc_id=?", (rec["doc_id"],)).fetchone()
                rowid = int(row["id"])
                if self.fts_enabled:
                    self.conn.execute("DELETE FROM documents_fts WHERE rowid=?", (rowid,))
                    self.conn.execute(
                        "INSERT INTO documents_fts(rowid,title,text,answer,keywords,tags) VALUES (?,?,?,?,?,?)",
                        (rowid, rec["title"], rec["text"], rec["answer"],
                         " ".join(rec["keywords"]), " ".join(rec["tags"])),
                    )
                count += 1
        return count

    def import_jsonl(self, filename: str, batch_size: int = 500) -> int:
        """Stream JSONL into SQLite. Memory use is bounded by batch_size records."""
        imported = 0
        batch: list[dict] = []
        with open(filename, "r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError("each JSONL line must contain an object")
                    batch.append(item)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"{filename}:{line_number}: {exc}") from exc
                if len(batch) >= batch_size:
                    imported += self.upsert_records(batch)
                    batch.clear()
        if batch:
            imported += self.upsert_records(batch)
        return imported

    def search(self, query: str, limit: int = 30) -> list[dict]:
        terms = []
        seen = set()
        for term in query.lower().split():
            term = "".join(ch for ch in term if ch.isalnum() or ch in "-'")
            if len(term) > 1 and term not in seen:
                terms.append(term)
                seen.add(term)
        if not terms:
            return []

        rows = []
        if self.fts_enabled:
            expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
            try:
                rows = self.conn.execute(
                    "SELECT d.* FROM documents_fts f "
                    "JOIN documents d ON d.id=f.rowid "
                    "WHERE documents_fts MATCH ? "
                    "ORDER BY bm25(documents_fts) LIMIT ?",
                    (expression, int(limit)),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []

        if not rows:
            clauses = []
            params: list[str | int] = []
            for term in terms:
                clauses.append("(lower(title) LIKE ? OR lower(text) LIKE ? OR lower(answer) LIKE ? OR lower(keywords) LIKE ?)")
                pattern = f"%{term}%"
                params.extend([pattern, pattern, pattern, pattern])
            sql = "SELECT * FROM documents WHERE " + " OR ".join(clauses) + " LIMIT ?"
            params.append(int(limit))
            rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> dict:
        result = dict(row)
        for key in ("keywords", "intents", "tags"):
            try:
                result[key] = json.loads(result[key])
            except (TypeError, json.JSONDecodeError):
                result[key] = []
        return result

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def close(self) -> None:
        self.conn.close()
