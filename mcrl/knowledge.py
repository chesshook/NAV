"""Disk-backed documents and provenance-aware semantic facts.

Imports are streamed in batches. Facts are atomic subject-predicate-object units,
so the answer composer can select and qualify individual claims rather than quote
an entire stored paragraph.
"""
from __future__ import annotations

import json
import re
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
        self.fact_fts_enabled = False
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
            self.fts_enabled = False

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                aliases TEXT NOT NULL DEFAULT '[]',
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                epistemic_status TEXT NOT NULL DEFAULT 'established',
                priority INTEGER NOT NULL DEFAULT 50,
                source TEXT NOT NULL,
                source_url TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject)")
        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts "
                "USING fts5(fact_id UNINDEXED, subject, aliases, predicate, object, tags)"
            )
            self.fact_fts_enabled = True
        except sqlite3.OperationalError:
            self.fact_fts_enabled = False
        self.conn.commit()

    @staticmethod
    def _clean_list(value, field_name: str) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{field_name} must be an array or string")
        return [str(item).strip() for item in value if str(item).strip()]

    @classmethod
    def _clean_record(cls, record: dict) -> dict:
        for field in ("id", "title"):
            if not str(record.get(field, "")).strip():
                raise ValueError(f"Knowledge record is missing required field: {field}")
        return {
            "doc_id": str(record["id"]).strip(),
            "title": str(record["title"]).strip(),
            "text": str(record.get("text", "")).strip(),
            "answer": str(record.get("answer", "")).strip(),
            "domain": str(record.get("domain", "")).strip(),
            "source": str(record.get("source", "")).strip(),
            "licence": str(record.get("licence", record.get("license", ""))).strip(),
            "keywords": cls._clean_list(record.get("keywords"), "keywords"),
            "intents": cls._clean_list(record.get("intents"), "intents"),
            "tags": cls._clean_list(record.get("tags"), "tags"),
        }

    @classmethod
    def _clean_fact(cls, fact: dict) -> dict:
        required = ("id", "subject", "predicate", "object", "source", "source_url")
        for field in required:
            if not str(fact.get(field, "")).strip():
                raise ValueError(f"Fact is missing required field: {field}")
        confidence = float(fact.get("confidence", 1.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("fact confidence must be between 0 and 1")
        status = str(fact.get("epistemic_status", "established")).strip().lower()
        if status not in {"established", "qualified", "tentative"}:
            raise ValueError("epistemic_status must be established, qualified, or tentative")
        return {
            "fact_id": str(fact["id"]).strip(),
            "subject": str(fact["subject"]).strip(),
            "aliases": cls._clean_list(fact.get("aliases"), "aliases"),
            "predicate": str(fact["predicate"]).strip(),
            "object": str(fact["object"]).strip(),
            "confidence": confidence,
            "epistemic_status": status,
            "priority": int(fact.get("priority", 50)),
            "source": str(fact["source"]).strip(),
            "source_url": str(fact["source_url"]).strip(),
            "tags": cls._clean_list(fact.get("tags"), "tags"),
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

    def upsert_facts(self, facts: Iterable[dict]) -> int:
        count = 0
        with self.conn:
            for raw in facts:
                fact = self._clean_fact(raw)
                values = (
                    fact["fact_id"], fact["subject"],
                    json.dumps(fact["aliases"], ensure_ascii=False),
                    fact["predicate"], fact["object"], fact["confidence"],
                    fact["epistemic_status"], fact["priority"], fact["source"],
                    fact["source_url"], json.dumps(fact["tags"], ensure_ascii=False),
                )
                self.conn.execute(
                    """
                    INSERT INTO facts
                    (fact_id,subject,aliases,predicate,object,confidence,
                     epistemic_status,priority,source,source_url,tags)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(fact_id) DO UPDATE SET
                      subject=excluded.subject, aliases=excluded.aliases,
                      predicate=excluded.predicate, object=excluded.object,
                      confidence=excluded.confidence, epistemic_status=excluded.epistemic_status,
                      priority=excluded.priority, source=excluded.source,
                      source_url=excluded.source_url, tags=excluded.tags
                    """,
                    values,
                )
                if self.fact_fts_enabled:
                    self.conn.execute("DELETE FROM facts_fts WHERE fact_id=?", (fact["fact_id"],))
                    self.conn.execute(
                        "INSERT INTO facts_fts(fact_id,subject,aliases,predicate,object,tags) VALUES (?,?,?,?,?,?)",
                        (
                            fact["fact_id"], fact["subject"], " ".join(fact["aliases"]),
                            fact["predicate"], fact["object"], " ".join(fact["tags"]),
                        ),
                    )
                count += 1
        return count

    @staticmethod
    def _stream_jsonl(filename: str, batch_size: int):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
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
                    yield batch
                    batch = []
        if batch:
            yield batch

    def import_jsonl(self, filename: str, batch_size: int = 500) -> int:
        """Stream document records into SQLite with bounded batch memory."""
        imported = 0
        for batch in self._stream_jsonl(filename, batch_size):
            imported += self.upsert_records(batch)
        return imported

    def import_facts_jsonl(self, filename: str, batch_size: int = 500) -> int:
        """Stream atomic, source-attributed semantic facts into SQLite."""
        imported = 0
        for batch in self._stream_jsonl(filename, batch_size):
            imported += self.upsert_facts(batch)
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

    def search_facts(self, query: str | Iterable[str], limit: int = 20) -> list[dict]:
        """Find facts by subject, aliases, and tagged topic—not by answer prose alone."""
        raw_terms = query.split() if isinstance(query, str) else list(query)
        terms: list[str] = []
        for raw in raw_terms:
            for term in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", str(raw).lower()):
                if term not in terms and len(term) > 1:
                    terms.append(term)
        if not terms:
            return []

        ids: list[str] = []
        if self.fact_fts_enabled:
            expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
            try:
                rows = self.conn.execute(
                    "SELECT fact_id FROM facts_fts WHERE facts_fts MATCH ? ORDER BY bm25(facts_fts) LIMIT ?",
                    (expression, int(limit)),
                ).fetchall()
                ids = [str(row["fact_id"]) for row in rows]
            except sqlite3.OperationalError:
                ids = []

        if not ids:
            clauses = []
            params: list[str | int] = []
            for term in terms:
                pattern = f"%{term}%"
                clauses.append("(lower(subject) LIKE ? OR lower(aliases) LIKE ? OR lower(predicate) LIKE ? OR lower(tags) LIKE ?)")
                params.extend([pattern, pattern, pattern, pattern])
            params.append(int(limit))
            rows = self.conn.execute(
                "SELECT fact_id FROM facts WHERE " + " OR ".join(clauses)
                + " ORDER BY priority ASC, confidence DESC LIMIT ?",
                params,
            ).fetchall()
            ids = [str(row["fact_id"]) for row in rows]

        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        fetched = self.conn.execute(
            f"SELECT * FROM facts WHERE fact_id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id = {str(row["fact_id"]): self._row_to_fact(row) for row in fetched}
        results = [by_id[fact_id] for fact_id in ids if fact_id in by_id]
        # Priority is hand-authored metadata; FTS supplies candidates, not truth.
        results.sort(key=lambda item: (item["priority"], -item["confidence"], item["fact_id"]))
        return results[:limit]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> dict:
        result = dict(row)
        for key in ("keywords", "intents", "tags"):
            try:
                result[key] = json.loads(result[key])
            except (TypeError, json.JSONDecodeError):
                result[key] = []
        return result

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> dict:
        result = dict(row)
        for key in ("aliases", "tags"):
            try:
                result[key] = json.loads(result[key])
            except (TypeError, json.JSONDecodeError):
                result[key] = []
        return result

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def fact_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])

    def close(self) -> None:
        self.conn.close()
