"""Fail-closed pre-retraction archive ledger and same-DB lossless soft-eviction for ZeroMem.

Design invariants (zero silent data loss, dry-run by default):
1. Pre-retraction archive + `retractions.jsonl` ledger (`retract_record`):
   - Requires explicit `--archive-root` or env `ZEROMEM_REMOVED_ROOT` / `MEMORY_REMOVED_ROOT`;
     refuses if missing or placed inside agent/config/DB directories.
   - Strict sequence: snapshot full record -> write exclusive (`O_EXCL`) fsynced archive JSON
     -> re-read and verify raw bytes & canonical snapshot SHA-256 -> append 1 body-free metadata
     row to `retractions.jsonl` (fsynced) -> ONLY THEN delete from active SQLite tables.
2. Same-DB lossless soft-eviction side table (`soft_evict` / `restore_evicted`):
   - Creates `evicted_records` side table without altering `meta.schema_version = '1'` or
     existing table definitions.
   - Moves facts or full documents into `evicted_records` inside a single `BEGIN IMMEDIATE`
     transaction with hard conservation assertions (`active_after + moved == active_before`)
     and `PRAGMA foreign_key_check`.
   - Refuses to empty 100% of active facts/documents in scope unless `force=True`.
   - `restore_evicted` verifies `content_sha256` of every row in the batch (and optional
     manifest) BEFORE restoring a single row; any tampering aborts fail-closed with 0 changes,
     and atomically restores both base tables and dual-track `facts_fts`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any

from .expiry import parse_until_date, resolve_as_of_date
from .store import SQLiteMemoryStore, tokenize_for_fts

EVICTED_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS evicted_records (
    batch_id TEXT NOT NULL,
    record_kind TEXT NOT NULL CHECK(record_kind IN ('fact', 'document')),
    record_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    reason TEXT NOT NULL,
    evicted_at TEXT NOT NULL,
    PRIMARY KEY (batch_id, record_kind, record_id)
);
CREATE INDEX IF NOT EXISTS idx_evicted_records_batch
    ON evicted_records(batch_id);
"""

_DISALLOWED_HOME_SUBDIRS = (
    ".zeromem",
    ".v2-memory",
    ".dsh",
    ".codex",
    ".gemini",
    ".claude",
    ".cursor",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_subpath(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def resolve_archive_root(
    explicit_root: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    db_path: str | Path | None = None,
) -> tuple[Path, str]:
    """Resolve and validate the pre-retraction archive root (fail-closed if unset or unsafe)."""
    source_env = os.environ if env is None else env
    if explicit_root is not None and str(explicit_root).strip():
        raw, source = str(explicit_root).strip(), "cli:--archive-root"
    elif source_env.get("ZEROMEM_REMOVED_ROOT", "").strip():
        raw, source = source_env["ZEROMEM_REMOVED_ROOT"].strip(), "env:ZEROMEM_REMOVED_ROOT"
    elif source_env.get("MEMORY_REMOVED_ROOT", "").strip():
        raw, source = source_env["MEMORY_REMOVED_ROOT"].strip(), "env:MEMORY_REMOVED_ROOT"
    else:
        raise ValueError(
            "archive-root-required: supply --archive-root or set ZEROMEM_REMOVED_ROOT "
            "(fail-closed: refusing to retract without an external archive destination)"
        )

    if "\x00" in raw:
        raise ValueError("invalid-archive-root: null byte not allowed")
    root = Path(raw).expanduser().resolve()
    home = Path.home().resolve()
    for blocked_name in _DISALLOWED_HOME_SUBDIRS:
        blocked = home / blocked_name
        if _is_subpath(root, blocked):
            raise ValueError(
                f"unsafe-archive-root: archive root must not live inside {blocked}"
            )
    if db_path is not None:
        db_dir = Path(db_path).expanduser().resolve().parent
        if _is_subpath(root, db_dir):
            raise ValueError(
                f"unsafe-archive-root: archive root must not live inside database directory {db_dir}"
            )
    return root, source


def _next_ledger_seq(ledger_path: Path) -> int:
    if not ledger_path.is_file():
        return 1
    seq = 0
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if isinstance(row.get("seq"), int):
                seq = max(seq, row["seq"])
        except ValueError:
            seq += 1
    return seq + 1


def _write_exclusive_fsync(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(str(target), flags, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def archive_and_append_ledger(
    *,
    archive_root: Path,
    archive_root_source: str,
    ledger_dir: Path,
    action: str,
    target_kind: str,
    target_id: str,
    namespace: str,
    snapshot_payload: dict[str, Any],
    text_length: int,
    reason: str,
) -> dict[str, Any]:
    """Write immutable full-body archive, verify round-trip SHA-256, then append body-free ledger."""
    started = time.monotonic()
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason-required: supply a non-empty retraction reason")
    clean_reason = re.sub(r"\s+", " ", reason.strip())[:240]

    canonical_snapshot = canonical_json(snapshot_payload)
    content_sha = sha256_text(canonical_snapshot)
    now_dt = datetime.now(timezone.utc)
    month_bucket = now_dt.strftime("%Y-%m")
    stamp = now_dt.strftime("%Y%m%dT%H%M%S%fZ")
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(target_id))[:64] or "record"

    archive_doc = {
        "schema_version": 1,
        "archived_at": now_dt.isoformat(),
        "action": action,
        "store_kind": "zeromem_sqlite",
        "target_kind": target_kind,
        "target_id": str(target_id),
        "namespace": namespace,
        "reason": clean_reason,
        "content_sha256": content_sha,
        "snapshot": snapshot_payload,
    }
    archive_bytes = (json.dumps(archive_doc, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    archive_file = archive_root / month_bucket / f"{target_kind}-{safe_id}-{stamp}-{content_sha[:12]}.json"
    _write_exclusive_fsync(archive_file, archive_bytes)

    # Round-trip disk verification before writing ledger or permitting deletion
    disk_bytes = archive_file.read_bytes()
    if sha256_bytes(disk_bytes) != sha256_bytes(archive_bytes):
        raise ValueError("archive-verification-failed: written archive bytes mismatch")
    reloaded = json.loads(disk_bytes.decode("utf-8"))
    if (
        reloaded.get("content_sha256") != content_sha
        or canonical_json(reloaded.get("snapshot")) != canonical_snapshot
    ):
        raise ValueError("archive-verification-failed: archived snapshot hash mismatch")

    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_dir / "retractions.jsonl"
    seq = _next_ledger_seq(ledger_path)
    elapsed_ms = round((time.monotonic() - started) * 1000, 2)
    ledger_entry = {
        "seq": seq,
        "ts": now_dt.isoformat(),
        "action": action,
        "store_kind": "zeromem_sqlite",
        "target_kind": target_kind,
        "target_id": str(target_id),
        "namespace": namespace,
        "content_sha256": content_sha,
        "before_bytes": len(archive_bytes),
        "text_length": int(text_length),
        "reason": clean_reason,
        "archive_file": str(archive_file),
        "archive_root_source": archive_root_source,
        "ms": elapsed_ms,
    }
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(ledger_entry, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    return {
        "seq": seq,
        "content_sha256": content_sha,
        "before_bytes": len(archive_bytes),
        "text_length": int(text_length),
        "archive_file": str(archive_file),
        "ledger_file": str(ledger_path),
        "archive_root_source": archive_root_source,
        "ms": elapsed_ms,
    }


def _has_table(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _snapshot_fact(conn: sqlite3.Connection, fact_id: int) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM facts WHERE fact_id = ?", (int(fact_id),)).fetchone()
    if not row:
        raise ValueError(f"fact-not-found: {fact_id}")
    fact = dict(row)
    ev_ids = json.loads(fact.get("evidence_ids_json") or "[]")
    segs: list[dict[str, Any]] = []
    if ev_ids:
        placeholders = ",".join("?" for _ in ev_ids)
        seg_rows = conn.execute(
            f"SELECT * FROM segments WHERE doc_id = ? AND seg_id IN ({placeholders}) ORDER BY seg_id",
            [fact["doc_id"], *ev_ids],
        ).fetchall()
        segs = [dict(r) for r in seg_rows]
    snap: dict[str, Any] = {"fact": fact, "bound_segments": segs}
    if _has_table(conn, "fact_supersessions"):
        sup_rows = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM fact_supersessions WHERE new_fact_id = ? ORDER BY old_fact_id ASC",
                (int(fact_id),),
            ).fetchall()
        ]
        if sup_rows:
            snap["supersessions"] = sup_rows
    return snap


def _snapshot_document(conn: sqlite3.Connection, doc_id: str) -> dict[str, Any]:
    doc = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    if not doc:
        raise ValueError(f"document-not-found: {doc_id}")
    segs = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM segments WHERE doc_id = ? ORDER BY seg_id", (doc_id,)
        ).fetchall()
    ]
    facts = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM facts WHERE doc_id = ? ORDER BY fact_id ASC", (doc_id,)
        ).fetchall()
    ]
    snap: dict[str, Any] = {"document": dict(doc), "segments": segs, "facts": facts}
    if facts and _has_table(conn, "fact_supersessions"):
        fids = [int(f["fact_id"]) for f in facts]
        placeholders = ",".join("?" for _ in fids)
        sup_rows = [
            dict(r)
            for r in conn.execute(
                f"SELECT * FROM fact_supersessions WHERE new_fact_id IN ({placeholders}) "
                "ORDER BY new_fact_id ASC, old_fact_id ASC",
                fids,
            ).fetchall()
        ]
        if sup_rows:
            snap["supersessions"] = sup_rows
    return snap


def _open_ro(db_file: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_file.as_uri() + "?mode=ro", uri=True, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def retract_record(
    *,
    db_path: str | Path,
    reason: str,
    fact_id: int | None = None,
    doc_id: str | None = None,
    archive_root: str | Path | None = None,
    ledger_dir: str | Path | None = None,
    apply: bool = False,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fail-closed retraction: archive full snapshot -> append ledger -> verify -> delete."""
    if (fact_id is None) == (not doc_id):
        raise ValueError("target-required: provide exactly one of fact_id or doc_id")
    db_file = Path(db_path).expanduser().resolve()
    if not db_file.is_file():
        raise ValueError(f"db-not-found: {db_file}")

    resolved_archive_root, root_source = resolve_archive_root(archive_root, env=env, db_path=db_file)
    resolved_ledger_dir = (
        Path(ledger_dir).expanduser().resolve()
        if ledger_dir
        else (db_file.parent / "governance")
    )

    conn = _open_ro(db_file)
    try:
        if fact_id is not None:
            target_kind = "fact"
            target_id = str(int(fact_id))
            snap = _snapshot_fact(conn, int(fact_id))
            namespace = str(snap["fact"]["namespace"])
            text_len = len(str(snap["fact"].get("text") or ""))
            fact_count = 1
            seg_count = len(snap["bound_segments"])
        else:
            target_kind = "document"
            target_id = str(doc_id)
            snap = _snapshot_document(conn, target_id)
            namespace = str(snap["document"]["namespace"])
            text_len = sum(len(str(s.get("text") or "")) for s in snap["segments"])
            fact_count = len(snap["facts"])
            seg_count = len(snap["segments"])
    finally:
        conn.close()

    plan = {
        "mode": "apply" if apply else "dry_run",
        "action": f"retract_{target_kind}",
        "db_path": str(db_file),
        "target_kind": target_kind,
        "target_id": target_id,
        "namespace": namespace,
        "fact_count": fact_count,
        "segment_count": seg_count,
        "text_length": text_len,
        "archive_root": str(resolved_archive_root),
        "archive_root_source": root_source,
        "ledger_dir": str(resolved_ledger_dir),
    }
    if not apply:
        return {**plan, "mutated": False}

    receipt = archive_and_append_ledger(
        archive_root=resolved_archive_root,
        archive_root_source=root_source,
        ledger_dir=resolved_ledger_dir,
        action="retract",
        target_kind=target_kind,
        target_id=target_id,
        namespace=namespace,
        snapshot_payload=snap,
        text_length=text_len,
        reason=reason,
    )
    store = SQLiteMemoryStore(db_file)
    deleted = (
        store.delete_fact(int(fact_id))
        if fact_id is not None
        else store.delete_document(str(doc_id))
    )
    return {**plan, "mutated": bool(deleted), "deleted": bool(deleted), "ledger_receipt": receipt}


def _select_evict_candidates(
    conn: sqlite3.Connection,
    *,
    fact_ids: Sequence[int] | None = None,
    doc_ids: Sequence[str] | None = None,
    expired_only: bool = False,
    namespace: str | None = None,
    as_of: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    ns_clause = " WHERE namespace = ?" if namespace else ""
    ns_params: list[Any] = [namespace] if namespace else []

    active_facts = [
        dict(r)
        for r in conn.execute(
            f"SELECT * FROM facts{ns_clause} ORDER BY fact_id ASC", ns_params
        ).fetchall()
    ]
    active_docs = [
        dict(r)
        for r in conn.execute(
            f"SELECT * FROM documents{ns_clause} ORDER BY doc_id ASC", ns_params
        ).fetchall()
    ]

    wanted_docs = set(str(d) for d in (doc_ids or []))
    chosen_docs = [d for d in active_docs if d["doc_id"] in wanted_docs]
    chosen_doc_ids = {d["doc_id"] for d in chosen_docs}

    wanted_facts = {int(fid) for fid in (fact_ids or [])}
    today = resolve_as_of_date(as_of)

    chosen_facts: list[dict[str, Any]] = []
    for row in active_facts:
        # Facts belonging to a selected document are evicted as part of that document
        if row["doc_id"] in chosen_doc_ids:
            continue
        matched = int(row["fact_id"]) in wanted_facts
        if not matched and expired_only:
            entities = json.loads(row.get("entities_json") or "[]")
            ev_ids = json.loads(row.get("evidence_ids_json") or "[]")
            segs: list[dict[str, Any]] = []
            if ev_ids:
                placeholders = ",".join("?" for _ in ev_ids)
                segs = [
                    dict(sr)
                    for sr in conn.execute(
                        f"SELECT seg_id, text FROM segments WHERE doc_id = ? AND seg_id IN ({placeholders})",
                        [row["doc_id"], *ev_ids],
                    ).fetchall()
                ]
            until_d = parse_until_date(entities, text=str(row.get("text") or ""), evidence=segs)
            if until_d is not None and (today - until_d).days > 0:
                matched = True
        if matched:
            chosen_facts.append(row)

    # Total facts removed = standalone chosen_facts + facts belonging to chosen_docs
    doc_fact_count = sum(1 for row in active_facts if row["doc_id"] in chosen_doc_ids)
    totals = {
        "active_facts_before": len(active_facts),
        "active_documents_before": len(active_docs),
        "doc_contained_facts": doc_fact_count,
    }
    return chosen_facts, chosen_docs, totals


def soft_evict(
    *,
    db_path: str | Path,
    fact_ids: Sequence[int] | None = None,
    doc_ids: Sequence[str] | None = None,
    expired_only: bool = False,
    namespace: str | None = None,
    as_of: str | None = None,
    reason: str = "soft-evict",
    batch_id: str | None = None,
    dump_manifest: str | Path | None = None,
    ledger_dir: str | Path | None = None,
    force: bool = False,
    apply: bool = False,
) -> dict[str, Any]:
    """Lossless same-DB soft eviction into `evicted_records` with conservation & SHA-256."""
    db_file = Path(db_path).expanduser().resolve()
    if not db_file.is_file():
        raise ValueError(f"db-not-found: {db_file}")
    if not (fact_ids or doc_ids or expired_only):
        raise ValueError("evict-selector-required: pass --expired, --fact-id, or --doc-id")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason-required")

    now_iso = _utc_now()
    resolved_batch = batch_id or ("evict-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))

    if not apply:
        conn = _open_ro(db_file)
        try:
            chosen_facts, chosen_docs, totals = _select_evict_candidates(
                conn,
                fact_ids=fact_ids,
                doc_ids=doc_ids,
                expired_only=expired_only,
                namespace=namespace,
                as_of=as_of,
            )
            total_facts_removed = len(chosen_facts) + totals["doc_contained_facts"]
            would_empty_facts = (
                totals["active_facts_before"] > 0
                and total_facts_removed == totals["active_facts_before"]
            )
            would_empty_docs = (
                totals["active_documents_before"] > 0
                and len(chosen_docs) == totals["active_documents_before"]
            )
            return {
                "mode": "dry_run",
                "mutated": False,
                "batch_id": resolved_batch,
                "db_path": str(db_file),
                "namespace": namespace,
                **totals,
                "candidate_fact_ids": [int(r["fact_id"]) for r in chosen_facts],
                "candidate_doc_ids": [str(r["doc_id"]) for r in chosen_docs],
                "would_empty_active_table": would_empty_facts or would_empty_docs,
                "requires_force": (would_empty_facts or would_empty_docs) and not force,
            }
        finally:
            conn.close()

    conn = sqlite3.connect(str(db_file), timeout=20.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 20000")
        conn.executescript(EVICTED_TABLE_DDL)
        conn.execute("BEGIN IMMEDIATE")
        try:
            schema_row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if not schema_row or str(schema_row[0]) != "1":
                raise ValueError("unsupported-zeromem-schema")

            chosen_facts, chosen_docs, totals = _select_evict_candidates(
                conn,
                fact_ids=fact_ids,
                doc_ids=doc_ids,
                expired_only=expired_only,
                namespace=namespace,
                as_of=as_of,
            )
            if not chosen_facts and not chosen_docs:
                conn.rollback()
                return {
                    "mode": "apply",
                    "mutated": False,
                    "batch_id": resolved_batch,
                    "moved_facts": 0,
                    "moved_documents": 0,
                    **totals,
                    "note": "no matching active records to evict",
                }

            total_facts_removed = len(chosen_facts) + totals["doc_contained_facts"]
            if not force:
                if totals["active_facts_before"] > 0 and total_facts_removed == totals["active_facts_before"]:
                    raise ValueError("refuse-empty-active-facts: pass --force to evict all active facts")
                if totals["active_documents_before"] > 0 and len(chosen_docs) == totals["active_documents_before"]:
                    raise ValueError("refuse-empty-active-documents: pass --force to evict all active documents")

            fingerprints: list[dict[str, Any]] = []

            has_sup = _has_table(conn, "fact_supersessions")
            for frow in chosen_facts:
                fid = int(frow["fact_id"])
                snap = _snapshot_fact(conn, fid)
                payload_str = canonical_json(snap)
                fp = sha256_text(payload_str)
                conn.execute(
                    """INSERT INTO evicted_records
                       (batch_id, record_kind, record_id, namespace, payload_json, content_sha256, reason, evicted_at)
                       VALUES (?, 'fact', ?, ?, ?, ?, ?, ?)""",
                    (resolved_batch, str(fid), str(frow["namespace"]), payload_str, fp, reason.strip(), now_iso),
                )
                cur = conn.execute("DELETE FROM facts WHERE fact_id = ?", (fid,))
                conn.execute("DELETE FROM facts_fts WHERE fact_id = ?", (fid,))
                if has_sup:
                    conn.execute("DELETE FROM fact_supersessions WHERE new_fact_id = ?", (fid,))
                if cur.rowcount != 1:
                    raise ValueError(f"evict-rowcount-mismatch for fact {fid}")
                fingerprints.append(
                    {
                        "record_kind": "fact",
                        "record_id": str(fid),
                        "namespace": str(frow["namespace"]),
                        "content_sha256": fp,
                    }
                )

            for drow in chosen_docs:
                did = str(drow["doc_id"])
                snap = _snapshot_document(conn, did)
                payload_str = canonical_json(snap)
                fp = sha256_text(payload_str)
                conn.execute(
                    """INSERT INTO evicted_records
                       (batch_id, record_kind, record_id, namespace, payload_json, content_sha256, reason, evicted_at)
                       VALUES (?, 'document', ?, ?, ?, ?, ?, ?)""",
                    (resolved_batch, did, str(drow["namespace"]), payload_str, fp, reason.strip(), now_iso),
                )
                for f in snap["facts"]:
                    conn.execute("DELETE FROM facts_fts WHERE fact_id = ?", (int(f["fact_id"]),))
                    if has_sup:
                        conn.execute("DELETE FROM fact_supersessions WHERE new_fact_id = ?", (int(f["fact_id"]),))
                conn.execute("DELETE FROM facts WHERE doc_id = ?", (did,))
                conn.execute("DELETE FROM segments WHERE doc_id = ?", (did,))
                cur = conn.execute("DELETE FROM documents WHERE doc_id = ?", (did,))
                if cur.rowcount != 1:
                    raise ValueError(f"evict-rowcount-mismatch for document {did}")
                fingerprints.append(
                    {
                        "record_kind": "document",
                        "record_id": did,
                        "namespace": str(drow["namespace"]),
                        "content_sha256": fp,
                    }
                )

            ns_clause = " WHERE namespace = ?" if namespace else ""
            ns_params: list[Any] = [namespace] if namespace else []
            facts_after = int(conn.execute(f"SELECT COUNT(*) FROM facts{ns_clause}", ns_params).fetchone()[0])
            docs_after = int(conn.execute(f"SELECT COUNT(*) FROM documents{ns_clause}", ns_params).fetchone()[0])

            # Hard conservation invariants inside the transaction
            if facts_after + total_facts_removed != totals["active_facts_before"]:
                raise ValueError("conservation-invariant-violated: facts")
            if docs_after + len(chosen_docs) != totals["active_documents_before"]:
                raise ValueError("conservation-invariant-violated: documents")
            fk_issues = conn.execute("PRAGMA foreign_key_check").fetchall()
            if fk_issues:
                raise ValueError(f"foreign-key-check-failed: {len(fk_issues)} violations")

            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    finally:
        conn.close()

    manifest_path = None
    if dump_manifest:
        manifest_path = Path(dump_manifest).expanduser().resolve()
        manifest_doc = {
            "schema_version": 1,
            "batch_id": resolved_batch,
            "db_path": str(db_file),
            "evicted_at": now_iso,
            "reason": reason.strip(),
            "fingerprints": fingerprints,
        }
        _write_exclusive_fsync(
            manifest_path,
            (json.dumps(manifest_doc, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )

    if ledger_dir:
        ldir = Path(ledger_dir).expanduser().resolve()
        ldir.mkdir(parents=True, exist_ok=True)
        lpath = ldir / "retractions.jsonl"
        row = {
            "seq": _next_ledger_seq(lpath),
            "ts": now_iso,
            "action": "soft_evict",
            "store_kind": "zeromem_sqlite",
            "batch_id": resolved_batch,
            "moved_facts": len(chosen_facts),
            "moved_documents": len(chosen_docs),
            "total_facts_removed": total_facts_removed,
            "reason": reason.strip()[:240],
            "manifest_file": str(manifest_path) if manifest_path else None,
        }
        with lpath.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    return {
        "mode": "apply",
        "mutated": True,
        "batch_id": resolved_batch,
        "db_path": str(db_file),
        "namespace": namespace,
        **totals,
        "active_facts_after": facts_after,
        "active_documents_after": docs_after,
        "moved_facts": len(chosen_facts),
        "moved_documents": len(chosen_docs),
        "total_facts_removed": total_facts_removed,
        "conservation_verified": True,
        "fingerprints": fingerprints,
        "manifest_file": str(manifest_path) if manifest_path else None,
    }


def _reinsert_fact_and_fts(
    conn: sqlite3.Connection,
    fact: Mapping[str, Any],
    seg_text_map: Mapping[str, str],
) -> None:
    conn.execute(
        """INSERT INTO facts(
               fact_id, doc_id, namespace, text, type, occurred_start, occurred_end,
               where_loc, entities_json, evidence_ids_json, action, mentioned_at, created_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            int(fact["fact_id"]),
            fact["doc_id"],
            fact["namespace"],
            fact["text"],
            fact["type"],
            fact.get("occurred_start"),
            fact.get("occurred_end"),
            fact.get("where_loc"),
            fact["entities_json"],
            fact["evidence_ids_json"],
            fact["action"],
            fact["mentioned_at"],
            fact["created_at"],
        ),
    )
    entities = json.loads(fact.get("entities_json") or "[]")
    ev_ids = json.loads(fact.get("evidence_ids_json") or "[]")
    fact_blob = " ".join(
        [
            str(fact["text"]),
            " ".join(str(e) for e in entities),
            str(fact.get("occurred_start") or ""),
            str(fact.get("where_loc") or ""),
        ]
    )
    bound_evidence = " ".join(seg_text_map.get(str(eid), "") for eid in ev_ids)
    conn.execute(
        "INSERT INTO facts_fts(fact_tokens, evidence_tokens, namespace, fact_id) VALUES (?, ?, ?, ?)",
        (
            tokenize_for_fts(fact_blob),
            tokenize_for_fts(bound_evidence),
            fact["namespace"],
            int(fact["fact_id"]),
        ),
    )


def restore_evicted(
    *,
    db_path: str | Path,
    batch_id: str,
    verify_manifest: str | Path | None = None,
    ledger_dir: str | Path | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Verify every evicted row's SHA-256 fingerprint before atomically restoring a batch."""
    db_file = Path(db_path).expanduser().resolve()
    if not db_file.is_file():
        raise ValueError(f"db-not-found: {db_file}")

    expected_map: dict[tuple[str, str], str] | None = None
    if verify_manifest:
        manifest = json.loads(Path(verify_manifest).expanduser().resolve().read_text(encoding="utf-8"))
        if manifest.get("batch_id") != batch_id:
            raise ValueError(
                f"manifest-batch-mismatch: expected {batch_id}, got {manifest.get('batch_id')}"
            )
        expected_map = {
            (str(row["record_kind"]), str(row["record_id"])): str(row["content_sha256"])
            for row in manifest.get("fingerprints") or []
        }

    def verify_rows(rows: Sequence[sqlite3.Row]) -> list[dict[str, Any]]:
        if not rows:
            raise ValueError(f"evicted-batch-not-found: {batch_id}")
        verified: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()
        for r in rows:
            actual_sha = sha256_text(str(r["payload_json"]))
            if actual_sha != str(r["content_sha256"]):
                raise ValueError(
                    f"evicted-fingerprint-mismatch: {r['record_kind']}:{r['record_id']} "
                    f"(stored={str(r['content_sha256'])[:12]}, actual={actual_sha[:12]})"
                )
            key = (str(r["record_kind"]), str(r["record_id"]))
            seen_keys.add(key)
            if expected_map is not None:
                if key not in expected_map or expected_map[key] != actual_sha:
                    raise ValueError(f"manifest-fingerprint-mismatch: {key}")
            verified.append(dict(r))
        if expected_map is not None and seen_keys != set(expected_map.keys()):
            raise ValueError("manifest-row-set-mismatch: batch rows differ from manifest")
        return verified

    if not apply:
        conn = _open_ro(db_file)
        try:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'evicted_records'"
            ).fetchone()
            if not has_table:
                raise ValueError("evicted_records table does not exist")
            rows = conn.execute(
                "SELECT * FROM evicted_records WHERE batch_id = ? ORDER BY record_kind, record_id",
                (batch_id,),
            ).fetchall()
            verified = verify_rows(rows)
            return {
                "mode": "dry_run",
                "mutated": False,
                "batch_id": batch_id,
                "verified_rows": len(verified),
                "fingerprint_verified": True,
                "record_ids": [f"{r['record_kind']}:{r['record_id']}" for r in verified],
            }
        finally:
            conn.close()

    conn = sqlite3.connect(str(db_file), timeout=20.0)
    conn.row_factory = sqlite3.Row
    now_iso = _utc_now()
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 20000")
        conn.execute("BEGIN IMMEDIATE")
        try:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'evicted_records'"
            ).fetchone()
            if not has_table:
                raise ValueError("evicted_records table does not exist")
            rows = conn.execute(
                "SELECT * FROM evicted_records WHERE batch_id = ? ORDER BY record_kind, record_id",
                (batch_id,),
            ).fetchall()
            verified = verify_rows(rows)

            facts_before = int(conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
            docs_before = int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

            restored_facts = 0
            restored_docs = 0

            # Restore documents first in case standalone facts refer to them
            for r in sorted(verified, key=lambda x: (0 if x["record_kind"] == "document" else 1, x["record_id"])):
                snap = json.loads(r["payload_json"])
                if r["record_kind"] == "document":
                    doc = snap["document"]
                    conn.execute(
                        """INSERT INTO documents(doc_id, namespace, content_sha256, source_id, mentioned_at, created_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            doc["doc_id"],
                            doc["namespace"],
                            doc["content_sha256"],
                            doc.get("source_id"),
                            doc["mentioned_at"],
                            doc["created_at"],
                        ),
                    )
                    seg_map: dict[str, str] = {}
                    for seg in snap.get("segments") or []:
                        conn.execute(
                            """INSERT INTO segments(doc_id, seg_id, namespace, path_json, role, text, mentioned_at)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (
                                seg["doc_id"],
                                seg["seg_id"],
                                seg["namespace"],
                                seg["path_json"],
                                seg.get("role"),
                                seg["text"],
                                seg["mentioned_at"],
                            ),
                        )
                        seg_map[str(seg["seg_id"])] = str(seg["text"])
                    for f in snap.get("facts") or []:
                        _reinsert_fact_and_fts(conn, f, seg_map)
                        restored_facts += 1
                    if snap.get("supersessions") and _has_table(conn, "fact_supersessions"):
                        for sup in snap["supersessions"]:
                            conn.execute(
                                """INSERT OR IGNORE INTO fact_supersessions(new_fact_id, old_fact_id, namespace, created_at)
                                   VALUES (?, ?, ?, ?)""",
                                (
                                    int(sup["new_fact_id"]),
                                    int(sup["old_fact_id"]),
                                    str(sup["namespace"]),
                                    str(sup["created_at"]),
                                ),
                            )
                    restored_docs += 1
                elif r["record_kind"] == "fact":
                    fact = snap["fact"]
                    seg_map = {
                        str(s["seg_id"]): str(s["text"])
                        for s in (snap.get("bound_segments") or [])
                    }
                    # Also fall back to existing segments table if present
                    for sr in conn.execute(
                        "SELECT seg_id, text FROM segments WHERE doc_id = ?", (fact["doc_id"],)
                    ).fetchall():
                        seg_map.setdefault(str(sr["seg_id"]), str(sr["text"]))
                    _reinsert_fact_and_fts(conn, fact, seg_map)
                    if snap.get("supersessions") and _has_table(conn, "fact_supersessions"):
                        for sup in snap["supersessions"]:
                            conn.execute(
                                """INSERT OR IGNORE INTO fact_supersessions(new_fact_id, old_fact_id, namespace, created_at)
                                   VALUES (?, ?, ?, ?)""",
                                (
                                    int(sup["new_fact_id"]),
                                    int(sup["old_fact_id"]),
                                    str(sup["namespace"]),
                                    str(sup["created_at"]),
                                ),
                            )
                    restored_facts += 1

            conn.execute("DELETE FROM evicted_records WHERE batch_id = ?", (batch_id,))
            facts_after = int(conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
            docs_after = int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

            if facts_after != facts_before + restored_facts:
                raise ValueError("restore-conservation-violated: facts")
            if docs_after != docs_before + restored_docs:
                raise ValueError("restore-conservation-violated: documents")
            fk_issues = conn.execute("PRAGMA foreign_key_check").fetchall()
            if fk_issues:
                raise ValueError(f"foreign-key-check-failed: {len(fk_issues)} violations")

            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    finally:
        conn.close()

    if ledger_dir:
        ldir = Path(ledger_dir).expanduser().resolve()
        ldir.mkdir(parents=True, exist_ok=True)
        lpath = ldir / "retractions.jsonl"
        row = {
            "seq": _next_ledger_seq(lpath),
            "ts": now_iso,
            "action": "soft_restore",
            "store_kind": "zeromem_sqlite",
            "batch_id": batch_id,
            "restored_facts": restored_facts,
            "restored_documents": restored_docs,
        }
        with lpath.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    return {
        "mode": "apply",
        "mutated": True,
        "batch_id": batch_id,
        "restored_facts": restored_facts,
        "restored_documents": restored_docs,
        "fingerprint_verified": True,
        "conservation_verified": True,
    }
