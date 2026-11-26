"""Single-file SQLite store with dual-track (fact + bound evidence) CJK 2-gram FTS5 search."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterator

from .contracts import Segment
from .expiry import (
    ADAPTIVE_MIN_LIMIT,
    ADAPTIVE_SCORE_RATIO,
    fact_matches_occurred_range,
    normalize_occurred_range,
    partition_expired,
    partition_superseded,
    resolve_default_limit,
    select_adaptive_hits,
)

# SQLite file format application_id & schema format signature
APP_FORMAT_ID = 0x494D4C4C
SCHEMA_SIG = "42e7ab99494de6a6b4e88eb2"

_ASCII_WORD_RE = re.compile(r"[a-z0-9][a-z0-9._:/-]*")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")

SCHEMA_SQL = f"""
PRAGMA application_id = {APP_FORMAT_ID};
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    source_id TEXT,
    mentioned_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_docs_ns_sha
    ON documents(namespace, content_sha256);

CREATE TABLE IF NOT EXISTS segments (
    doc_id TEXT NOT NULL,
    seg_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    path_json TEXT NOT NULL,
    role TEXT,
    text TEXT NOT NULL,
    mentioned_at TEXT NOT NULL,
    PRIMARY KEY (doc_id, seg_id)
);

CREATE TABLE IF NOT EXISTS facts (
    fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    text TEXT NOT NULL,
    type TEXT NOT NULL,
    occurred_start TEXT,
    occurred_end TEXT,
    where_loc TEXT,
    entities_json TEXT NOT NULL,
    evidence_ids_json TEXT NOT NULL,
    action TEXT NOT NULL,
    mentioned_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_ns_time
    ON facts(namespace, mentioned_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    fact_tokens,
    evidence_tokens,
    namespace UNINDEXED,
    fact_id UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS fact_supersessions (
    new_fact_id INTEGER NOT NULL,
    old_fact_id INTEGER NOT NULL,
    namespace TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (new_fact_id, old_fact_id)
);
CREATE INDEX IF NOT EXISTS idx_fact_supersessions_old
    ON fact_supersessions(old_fact_id);
"""


def _has_supersessions_table(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'fact_supersessions'"
    ).fetchone()
    return row is not None


def _extract_supersedes_ids(item: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    raw_single = item.get("supersedes_fact_id")
    raw_many = item.get("supersedes_fact_ids")
    candidates: list[Any] = []
    if raw_single is not None:
        candidates.append(raw_single)
    if isinstance(raw_many, list):
        candidates.extend(raw_many)
    for val in candidates:
        if isinstance(val, bool):
            continue
        parsed: int | None = None
        if isinstance(val, int):
            parsed = val
        elif isinstance(val, str) and val.strip().isdigit():
            parsed = int(val.strip())
        if parsed is not None and parsed > 0 and parsed not in ids:
            ids.append(parsed)
    return ids


def tokenize_for_fts(text: str) -> str:
    """Zero-dependency bilingual tokenizer (ASCII identifiers + CJK 2-grams)."""
    low = (text or "").lower()
    tokens: list[str] = []

    for m in _ASCII_WORD_RE.finditer(low):
        w = m.group(0).strip("._:/-")
        if w:
            tokens.append(w)
            parts = [p for p in re.split(r"[._:/-]+", w) if p and p != w]
            tokens.extend(parts)

    for m in _CJK_RUN_RE.finditer(low):
        run = m.group(0)
        if len(run) == 1:
            tokens.append(run)
        else:
            for i in range(len(run) - 1):
                tokens.append(run[i : i + 2])
            if len(run) <= 4:
                tokens.append(run)

    return " ".join(tokens)


def _fts_quote(tok: str) -> str:
    safe = tok.replace('"', "")
    return f'"{safe}"'


class SQLiteMemoryStore:
    """Single-file SQLite store (segments + facts + dual-track facts_fts)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
                ("schema_version", "1"),
            )
            conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
                ("schema_sig", SCHEMA_SIG),
            )

    def has_document(self, namespace: str, content_sha256: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT doc_id FROM documents WHERE namespace = ? AND content_sha256 = ?",
                (namespace, content_sha256),
            ).fetchone()
            return str(row["doc_id"]) if row else None

    def save_document_and_facts(
        self,
        *,
        doc_id: str,
        namespace: str,
        content_sha256: str,
        source_id: str | None,
        mentioned_at: str,
        created_at: str,
        segments: list[Segment],
        facts: list[dict[str, Any]],
    ) -> list[int]:
        seg_text_map = {s.id: s.text for s in segments}
        saved_ids: list[int] = []

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents(doc_id, namespace, content_sha256, source_id, mentioned_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, namespace, content_sha256, source_id, mentioned_at, created_at),
            )
            conn.executemany(
                """
                INSERT INTO segments(doc_id, seg_id, namespace, path_json, role, text, mentioned_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        doc_id,
                        s.id,
                        namespace,
                        json.dumps(s.path, ensure_ascii=False),
                        s.role,
                        s.text,
                        mentioned_at,
                    )
                    for s in segments
                ],
            )
            for item in facts:
                entities = item.get("entities") or []
                ev_ids = item.get("evidence") or []
                cur = conn.execute(
                    """
                    INSERT INTO facts(
                        doc_id, namespace, text, type, occurred_start, occurred_end,
                        where_loc, entities_json, evidence_ids_json, action, mentioned_at, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        namespace,
                        item["text"],
                        item["type"],
                        item.get("occurred_start"),
                        item.get("occurred_end"),
                        item.get("where"),
                        json.dumps(entities, ensure_ascii=False),
                        json.dumps(ev_ids, ensure_ascii=False),
                        item.get("action", "pass"),
                        mentioned_at,
                        created_at,
                    ),
                )
                fact_id = int(cur.lastrowid)
                saved_ids.append(fact_id)

                # Index fact tokens and bound evidence tokens in separate weighted columns
                fact_blob = " ".join(
                    [
                        item["text"],
                        " ".join(str(e) for e in entities),
                        str(item.get("occurred_start") or ""),
                        str(item.get("where") or ""),
                    ]
                )
                bound_evidence = " ".join(seg_text_map.get(eid, "") for eid in ev_ids)
                conn.execute(
                    "INSERT INTO facts_fts(fact_tokens, evidence_tokens, namespace, fact_id) VALUES (?, ?, ?, ?)",
                    (tokenize_for_fts(fact_blob), tokenize_for_fts(bound_evidence), namespace, fact_id),
                )

                for old_fid in _extract_supersedes_ids(item):
                    if old_fid == fact_id:
                        continue
                    old_row = conn.execute(
                        "SELECT fact_id FROM facts WHERE fact_id = ? AND namespace = ?",
                        (old_fid, namespace),
                    ).fetchone()
                    if old_row is not None:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO fact_supersessions(new_fact_id, old_fact_id, namespace, created_at)
                            VALUES (?, ?, ?, ?)
                            """,
                            (fact_id, old_fid, namespace, created_at),
                        )
        return saved_ids

    def search(
        self,
        query: str,
        *,
        namespace: str | None = None,
        limit: int | None = None,
        adaptive: bool = True,
        min_limit: int = ADAPTIVE_MIN_LIMIT,
        score_ratio: float = ADAPTIVE_SCORE_RATIO,
        as_of: str | date | None = None,
        occurred_from: str | date | None = None,
        occurred_to: str | date | None = None,
        include_superseded: bool = True,
    ) -> list[dict[str, Any]]:
        from_date, to_date = normalize_occurred_range(occurred_from, occurred_to)
        has_time_filter = from_date is not None or to_date is not None
        cap = resolve_default_limit(limit, adaptive=adaptive)
        fetch_limit = max(cap * 4, 48) if (has_time_filter or not include_superseded) else max(cap * 3, 24)
        q_tokens = tokenize_for_fts(query).split()

        with self._connect() as conn:
            if not q_tokens:
                rows = self._recent_rows(
                    conn,
                    namespace=namespace,
                    limit=fetch_limit,
                    from_date=from_date,
                    to_date=to_date,
                )
                hydrated = self._hydrate_ranked(conn, [(r, 0.0) for r in rows])
                if has_time_filter:
                    hydrated = [
                        r
                        for r in hydrated
                        if fact_matches_occurred_range(
                            r.get("occurred_start"),
                            r.get("occurred_end"),
                            from_date=from_date,
                            to_date=to_date,
                        )
                    ]
                if not include_superseded:
                    hydrated = [r for r in hydrated if not r.get("superseded_by")]
                if not has_time_filter:
                    hydrated, _ = partition_superseded(hydrated)
                partitioned, _ = partition_expired(hydrated, today=as_of)
                return partitioned[:cap]

            uniq_tokens: list[str] = []
            for t in q_tokens:
                if t not in uniq_tokens:
                    uniq_tokens.append(t)
                if len(uniq_tokens) >= 24:
                    break

            quoted = [_fts_quote(t) for t in uniq_tokens]
            seen_ids: set[int] = set()
            ranked: list[tuple[sqlite3.Row, float]] = []

            if 1 <= len(quoted) <= 6:
                and_expr = " AND ".join(quoted)
                for row, score in self._fts_query(
                    conn,
                    and_expr,
                    namespace=namespace,
                    limit=fetch_limit,
                    from_date=from_date,
                    to_date=to_date,
                ):
                    fid = int(row["fact_id"])
                    if fid not in seen_ids:
                        seen_ids.add(fid)
                        ranked.append((row, score + 1.0))

            if len(ranked) < fetch_limit:
                or_expr = " OR ".join(quoted)
                for row, score in self._fts_query(
                    conn,
                    or_expr,
                    namespace=namespace,
                    limit=fetch_limit * 2,
                    from_date=from_date,
                    to_date=to_date,
                ):
                    fid = int(row["fact_id"])
                    if fid not in seen_ids:
                        seen_ids.add(fid)
                        ranked.append((row, score))
                    if len(ranked) >= fetch_limit:
                        break

            hydrated = self._hydrate_ranked(conn, ranked[:fetch_limit])
            if has_time_filter:
                hydrated = [
                    r
                    for r in hydrated
                    if fact_matches_occurred_range(
                        r.get("occurred_start"),
                        r.get("occurred_end"),
                        from_date=from_date,
                        to_date=to_date,
                    )
                ]
            if not include_superseded:
                hydrated = [r for r in hydrated if not r.get("superseded_by")]
            if not has_time_filter:
                hydrated, _ = partition_superseded(hydrated)
            partitioned, _ = partition_expired(hydrated, today=as_of)
            return select_adaptive_hits(
                partitioned,
                limit=cap,
                min_limit=min_limit,
                score_ratio=score_ratio,
                adaptive=adaptive,
            )

    @staticmethod
    def _append_occurred_sql_prefilter(
        sql: str,
        params: list[Any],
        *,
        alias: str,
        from_date: date | None,
        to_date: date | None,
    ) -> str:
        if from_date is None and to_date is None:
            return sql
        col_start = f"{alias}.occurred_start" if alias else "occurred_start"
        col_end = f"{alias}.occurred_end" if alias else "occurred_end"
        sql += f" AND {col_start} IS NOT NULL AND {col_start} != ''"
        if to_date is not None:
            sql += f" AND substr({col_start}, 1, 10) <= ?"
            params.append(to_date.isoformat())
        if from_date is not None:
            end_expr = f"COALESCE(NULLIF({col_end}, ''), {col_start})"
            sql += f" AND (substr({end_expr}, 1, 10) >= ? OR length({end_expr}) < 10)"
            params.append(from_date.isoformat())
        return sql

    def _fts_query(
        self,
        conn: sqlite3.Connection,
        match_expr: str,
        *,
        namespace: str | None,
        limit: int,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> list[tuple[sqlite3.Row, float]]:
        sql = """
            SELECT f.*, bm25(facts_fts, 2.0, 1.0) AS rank_score
            FROM facts_fts
            JOIN facts f ON f.fact_id = facts_fts.fact_id
            WHERE facts_fts MATCH ?
        """
        params: list[Any] = [match_expr]
        if namespace:
            sql += " AND f.namespace = ?"
            params.append(namespace)
        sql = self._append_occurred_sql_prefilter(
            sql, params, alias="f", from_date=from_date, to_date=to_date
        )
        sql += " ORDER BY rank_score ASC, f.mentioned_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [(r, round(-float(r["rank_score"]), 4)) for r in rows]

    def _recent_rows(
        self,
        conn: sqlite3.Connection,
        *,
        namespace: str | None,
        limit: int,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM facts WHERE 1=1"
        params: list[Any] = []
        if namespace:
            sql += " AND namespace = ?"
            params.append(namespace)
        sql = self._append_occurred_sql_prefilter(
            sql, params, alias="", from_date=from_date, to_date=to_date
        )
        if from_date is not None or to_date is not None:
            sql += " ORDER BY occurred_start DESC, mentioned_at DESC, fact_id DESC LIMIT ?"
        else:
            sql += " ORDER BY mentioned_at DESC, fact_id DESC LIMIT ?"
        params.append(limit)
        return conn.execute(sql, params).fetchall()

    def _hydrate_ranked(
        self,
        conn: sqlite3.Connection,
        ranked: list[tuple[sqlite3.Row, float]],
    ) -> list[dict[str, Any]]:
        supersedes_map: dict[int, list[int]] = {}
        superseded_by_map: dict[int, list[int]] = {}
        if ranked and _has_supersessions_table(conn):
            fids = [int(row["fact_id"]) for row, _ in ranked]
            placeholders = ",".join("?" for _ in fids)
            for srow in conn.execute(
                f"SELECT new_fact_id, old_fact_id FROM fact_supersessions "
                f"WHERE new_fact_id IN ({placeholders}) ORDER BY old_fact_id ASC",
                fids,
            ).fetchall():
                supersedes_map.setdefault(int(srow["new_fact_id"]), []).append(int(srow["old_fact_id"]))
            for srow in conn.execute(
                f"SELECT fs.old_fact_id, fs.new_fact_id FROM fact_supersessions fs "
                f"JOIN facts nf ON nf.fact_id = fs.new_fact_id "
                f"WHERE fs.old_fact_id IN ({placeholders}) ORDER BY fs.new_fact_id ASC",
                fids,
            ).fetchall():
                superseded_by_map.setdefault(int(srow["old_fact_id"]), []).append(int(srow["new_fact_id"]))

        results: list[dict[str, Any]] = []
        for row, score in ranked:
            fid = int(row["fact_id"])
            doc_id = str(row["doc_id"])
            ev_ids = json.loads(row["evidence_ids_json"] or "[]")
            ev_texts: list[dict[str, Any]] = []
            if ev_ids:
                placeholders = ",".join("?" for _ in ev_ids)
                seg_rows = conn.execute(
                    f"SELECT seg_id, role, text FROM segments WHERE doc_id = ? AND seg_id IN ({placeholders})",
                    [doc_id, *ev_ids],
                ).fetchall()
                seg_lookup = {str(sr["seg_id"]): sr for sr in seg_rows}
                for sid in ev_ids:
                    sr = seg_lookup.get(sid)
                    if sr:
                        ev_texts.append(
                            {
                                "id": sid,
                                "role": sr["role"],
                                "text": sr["text"],
                            }
                        )
            item: dict[str, Any] = {
                "fact_id": fid,
                "doc_id": doc_id,
                "namespace": str(row["namespace"]),
                "text": str(row["text"]),
                "type": str(row["type"]),
                "occurred_start": row["occurred_start"],
                "occurred_end": row["occurred_end"],
                "where": row["where_loc"],
                "entities": json.loads(row["entities_json"] or "[]"),
                "evidence_ids": ev_ids,
                "evidence": ev_texts,
                "action": str(row["action"]),
                "mentioned_at": str(row["mentioned_at"]),
                "score": score,
            }
            if fid in supersedes_map:
                item["supersedes_fact_id"] = supersedes_map[fid][0]
            if fid in superseded_by_map:
                item["superseded_by"] = superseded_by_map[fid]
            results.append(item)
        return results

    def stats(self, namespace: str | None = None) -> dict[str, Any]:
        with self._connect() as conn:
            has_sup = _has_supersessions_table(conn)
            if namespace:
                docs = conn.execute(
                    "SELECT COUNT(*) FROM documents WHERE namespace = ?", (namespace,)
                ).fetchone()[0]
                segs = conn.execute(
                    "SELECT COUNT(*) FROM segments WHERE namespace = ?", (namespace,)
                ).fetchone()[0]
                facts = conn.execute(
                    "SELECT COUNT(*) FROM facts WHERE namespace = ?", (namespace,)
                ).fetchone()[0]
                revised = conn.execute(
                    "SELECT COUNT(*) FROM facts WHERE namespace = ? AND action = 'revise'",
                    (namespace,),
                ).fetchone()[0]
                supersessions = (
                    conn.execute(
                        "SELECT COUNT(*) FROM fact_supersessions fs "
                        "JOIN facts nf ON nf.fact_id = fs.new_fact_id "
                        "WHERE fs.namespace = ?",
                        (namespace,),
                    ).fetchone()[0]
                    if has_sup
                    else 0
                )
                namespaces = [namespace]
            else:
                docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                segs = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
                facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
                revised = conn.execute(
                    "SELECT COUNT(*) FROM facts WHERE action = 'revise'"
                ).fetchone()[0]
                supersessions = (
                    conn.execute(
                        "SELECT COUNT(*) FROM fact_supersessions fs "
                        "JOIN facts nf ON nf.fact_id = fs.new_fact_id"
                    ).fetchone()[0]
                    if has_sup
                    else 0
                )
                namespaces = [
                    str(r[0])
                    for r in conn.execute("SELECT DISTINCT namespace FROM documents ORDER BY namespace").fetchall()
                ]
        return {
            "db_path": str(self.db_path),
            "documents": int(docs),
            "segments": int(segs),
            "facts": int(facts),
            "revised_facts": int(revised),
            "supersessions": int(supersessions),
            "namespaces": namespaces,
        }

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Fetch document metadata, ordered immutable segments, and associated fact IDs."""
        with self._connect() as conn:
            doc = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
            if not doc:
                return None
            seg_rows = conn.execute(
                "SELECT seg_id, path_json, role, text FROM segments WHERE doc_id = ?",
                (doc_id,),
            ).fetchall()
            # Natural sort s0, s1, ..., s10
            seg_rows = sorted(
                seg_rows,
                key=lambda r: int(str(r["seg_id"])[1:]) if str(r["seg_id"])[1:].isdigit() else 0,
            )
            fact_rows = conn.execute(
                "SELECT fact_id, evidence_ids_json FROM facts WHERE doc_id = ? ORDER BY fact_id ASC",
                (doc_id,),
            ).fetchall()
        return {
            "doc_id": str(doc["doc_id"]),
            "namespace": str(doc["namespace"]),
            "source_id": doc["source_id"],
            "mentioned_at": str(doc["mentioned_at"]),
            "created_at": str(doc["created_at"]),
            "segments": [
                {
                    "id": str(r["seg_id"]),
                    "path": json.loads(r["path_json"] or "[]"),
                    "role": r["role"],
                    "text": str(r["text"]),
                }
                for r in seg_rows
            ],
            "fact_count": len(fact_rows),
        }

    def delete_fact(self, fact_id: int) -> bool:
        """Delete a single fact and remove its FTS5 entry and supersession edges."""
        with self._connect() as conn:
            fid = int(fact_id)
            cur = conn.execute("DELETE FROM facts WHERE fact_id = ?", (fid,))
            conn.execute("DELETE FROM facts_fts WHERE fact_id = ?", (fid,))
            if _has_supersessions_table(conn):
                conn.execute(
                    "DELETE FROM fact_supersessions WHERE new_fact_id = ? OR old_fact_id = ?",
                    (fid, fid),
                )
            return cur.rowcount > 0

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document along with all of its segments, facts, FTS5 entries, and supersession edges."""
        with self._connect() as conn:
            fids = [
                int(r[0])
                for r in conn.execute("SELECT fact_id FROM facts WHERE doc_id = ?", (doc_id,)).fetchall()
            ]
            has_sup = _has_supersessions_table(conn)
            for fid in fids:
                conn.execute("DELETE FROM facts_fts WHERE fact_id = ?", (fid,))
                if has_sup:
                    conn.execute(
                        "DELETE FROM fact_supersessions WHERE new_fact_id = ? OR old_fact_id = ?",
                        (fid, fid),
                    )
            conn.execute("DELETE FROM facts WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM segments WHERE doc_id = ?", (doc_id,))
            cur = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            return cur.rowcount > 0
