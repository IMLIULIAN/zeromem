"""Read-only Markdown export (Obsidian/Git friendly) and SQLite Online Backup snapshots.

Design invariants:
1. `export_markdown`:
   - 100% read-only: opens SQLite via `mode=ro` + `PRAGMA query_only = ON`, never mutates DB.
   - Zero LLM calls, zero third-party dependencies.
   - Preserves bound original evidence quotes, timeline fields (`occurred_start` / `occurred_end`),
     supersession lineage (`supersedes_fact_id` / `superseded_by`), and `until:` expiry status.
   - Defense-in-depth credential redaction via `redact_credentials`.
   - Supports stdout / single Markdown file (`output_path`) or per-namespace Obsidian/Git vault
     directory (`output_dir` with `INDEX.md`).
2. `backup_database`:
   - Uses SQLite Online Backup API (`sqlite3.Connection.backup`) so live WAL readers/writers are
     never blocked or corrupted.
   - Converts the destination snapshot to `PRAGMA journal_mode = DELETE`, guaranteeing a clean
     single-file snapshot with zero `-wal`/`-shm` sidecars (safe for OneDrive/BaiduSyncdisk/iCloud).
   - Verifies `PRAGMA integrity_check == 'ok'`, computes SHA-256, and publishes atomically via
     temp file + `os.replace`.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any

from .contracts import redact_credentials
from .expiry import (
    evaluate_expiry,
    fact_matches_occurred_range,
    normalize_occurred_range,
    parse_until_date,
    resolve_as_of_date,
)
from .store import APP_FORMAT_ID

_UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _has_table(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _open_ro(db_file: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_file.as_uri() + "?mode=ro", uri=True, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _atomic_write_text(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp.{os.getpid()}.{int(time.time() * 1000)}")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _safe_namespace_stem(namespace: str, used: set[str]) -> str:
    cleaned = _UNSAFE_FILENAME_RE.sub("_", namespace.strip()).strip(" .")
    if not cleaned or cleaned.upper() in {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
        "INDEX",
    }:
        cleaned = f"ns_{cleaned or 'default'}"
    cleaned = cleaned[:80]
    candidate = cleaned
    if candidate.casefold() in used:
        digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:6]
        candidate = f"{cleaned}_{digest}"
    used.add(candidate.casefold())
    return candidate


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    s = str(value)
    return json.dumps(s, ensure_ascii=False)


def _format_blockquote(prefix: str, text: str) -> str:
    safe_text = redact_credentials(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = safe_text.splitlines() or [""]
    out = [f"> {prefix}{lines[0]}"]
    for extra in lines[1:]:
        out.append(f"> {extra}" if extra else ">")
    return "\n".join(out)


def _render_fact_markdown(fact: dict[str, Any]) -> str:
    fid = fact["fact_id"]
    badges: list[str] = [f"`{fact['type']}`"]
    if fact["action"] == "revise":
        badges.append("`已纠错`")
    if fact.get("supersedes_fact_id") is not None:
        badges.append(f"`替代 #{fact['supersedes_fact_id']}`")
    if fact.get("superseded_by"):
        sup_refs = ", ".join(f"#{x}" for x in fact["superseded_by"])
        badges.append(f"`已被 {sup_refs} 替代`")
    if fact.get("until_expiry"):
        badges.append(f"`{fact['until_expiry']['label']}`")

    lines = [f"### Fact #{fid} " + " ".join(badges), ""]
    lines.append(redact_credentials(fact["text"]))
    lines.append("")

    meta_parts: list[str] = []
    if fact.get("occurred_start"):
        occ = str(fact["occurred_start"])
        if fact.get("occurred_end") and str(fact["occurred_end"]) != occ:
            occ = f"{occ} ~ {fact['occurred_end']}"
        meta_parts.append(f"**发生时间**: `{occ}`")
    meta_parts.append(f"**记录时间**: `{fact['mentioned_at']}`")
    if fact.get("where"):
        meta_parts.append(f"**地点/环境**: {redact_credentials(str(fact['where']))}")
    src_ref = f"`{fact['doc_id']}`"
    if fact.get("source_id"):
        src_ref += f" ({redact_credentials(str(fact['source_id']))})"
    meta_parts.append(f"**来源**: {src_ref}")
    lines.append("- " + " | ".join(meta_parts))

    entities = fact.get("entities") or []
    if entities:
        ent_str = " ".join(f"`{redact_credentials(str(e))}`" for e in entities)
        lines.append(f"- **实体/标签**: {ent_str}")

    if fact.get("supersedes_fact_id") is not None:
        lines.append(f"- **演化关系**: 替代旧事实 `Fact #{fact['supersedes_fact_id']}`")
    if fact.get("superseded_by"):
        newer = ", ".join(f"`Fact #{x}`" for x in fact["superseded_by"])
        lines.append(f"- **历史状态**: 已被更新事实 {newer} 替代")

    evidence = fact.get("evidence") or []
    if evidence:
        lines.append("")
        for ev in evidence:
            role_part = f" · `{ev['role']}`" if ev.get("role") else ""
            lines.append(_format_blockquote(f"**原文证据 (`{ev['id']}`{role_part})**: ", str(ev["text"])))

    lines.append("")
    return "\n".join(lines)


def _load_export_data(
    db_file: Path,
    *,
    namespace: str | None = None,
    from_date: date | None = None,
    to_date: date | None = None,
    as_of_date: date,
    include_superseded: bool = True,
    include_evicted: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, int]]:
    conn = _open_ro(db_file)
    try:
        if namespace:
            fact_rows = conn.execute(
                "SELECT * FROM facts WHERE namespace = ? ORDER BY namespace ASC, "
                "COALESCE(occurred_start, mentioned_at) DESC, fact_id ASC",
                (namespace,),
            ).fetchall()
            doc_rows = conn.execute(
                "SELECT doc_id, source_id FROM documents WHERE namespace = ?",
                (namespace,),
            ).fetchall()
        else:
            fact_rows = conn.execute(
                "SELECT * FROM facts ORDER BY namespace ASC, "
                "COALESCE(occurred_start, mentioned_at) DESC, fact_id ASC"
            ).fetchall()
            doc_rows = conn.execute("SELECT doc_id, source_id FROM documents").fetchall()

        doc_source_map = {str(r["doc_id"]): r["source_id"] for r in doc_rows}

        seg_cache: dict[str, dict[str, dict[str, Any]]] = {}
        doc_ids = sorted({str(r["doc_id"]) for r in fact_rows})
        for did in doc_ids:
            seg_cache[did] = {
                str(sr["seg_id"]): {
                    "id": str(sr["seg_id"]),
                    "role": sr["role"],
                    "text": str(sr["text"]),
                }
                for sr in conn.execute(
                    "SELECT seg_id, role, text FROM segments WHERE doc_id = ?", (did,)
                ).fetchall()
            }

        supersedes_map: dict[int, list[int]] = {}
        superseded_by_map: dict[int, list[int]] = {}
        if fact_rows and _has_table(conn, "fact_supersessions"):
            for srow in conn.execute(
                "SELECT new_fact_id, old_fact_id FROM fact_supersessions ORDER BY old_fact_id ASC"
            ).fetchall():
                supersedes_map.setdefault(int(srow["new_fact_id"]), []).append(int(srow["old_fact_id"]))
            for srow in conn.execute(
                "SELECT fs.old_fact_id, fs.new_fact_id FROM fact_supersessions fs "
                "JOIN facts nf ON nf.fact_id = fs.new_fact_id ORDER BY fs.new_fact_id ASC"
            ).fetchall():
                superseded_by_map.setdefault(int(srow["old_fact_id"]), []).append(int(srow["new_fact_id"]))

        evicted_batches: list[dict[str, Any]] = []
        if include_evicted and _has_table(conn, "evicted_records"):
            if namespace:
                erows = conn.execute(
                    "SELECT batch_id, record_kind, record_id, namespace, reason, evicted_at, content_sha256 "
                    "FROM evicted_records WHERE namespace = ? ORDER BY evicted_at DESC, batch_id ASC",
                    (namespace,),
                ).fetchall()
            else:
                erows = conn.execute(
                    "SELECT batch_id, record_kind, record_id, namespace, reason, evicted_at, content_sha256 "
                    "FROM evicted_records ORDER BY evicted_at DESC, batch_id ASC"
                ).fetchall()
            evicted_batches = [dict(r) for r in erows]
    finally:
        conn.close()

    by_ns: dict[str, list[dict[str, Any]]] = {}
    referenced_docs: set[str] = set()
    superseded_cnt = 0
    expired_cnt = 0
    total_facts = 0

    for row in fact_rows:
        if not fact_matches_occurred_range(
            row["occurred_start"],
            row["occurred_end"],
            from_date=from_date,
            to_date=to_date,
        ):
            continue
        fid = int(row["fact_id"])
        sup_by = superseded_by_map.get(fid)
        if not include_superseded and sup_by:
            continue

        did = str(row["doc_id"])
        ns = str(row["namespace"])
        ev_ids = json.loads(row["evidence_ids_json"] or "[]")
        entities = json.loads(row["entities_json"] or "[]")
        ev_list = [seg_cache.get(did, {}).get(eid) for eid in ev_ids]
        ev_clean = [e for e in ev_list if e is not None]

        until_d = parse_until_date(entities, text=str(row["text"] or ""), evidence=ev_clean)
        expiry_info = evaluate_expiry(until_d, as_of_date)

        item: dict[str, Any] = {
            "fact_id": fid,
            "doc_id": did,
            "source_id": doc_source_map.get(did),
            "namespace": ns,
            "text": str(row["text"]),
            "type": str(row["type"]),
            "occurred_start": row["occurred_start"],
            "occurred_end": row["occurred_end"],
            "where": row["where_loc"],
            "entities": entities,
            "evidence": ev_clean,
            "action": str(row["action"]),
            "mentioned_at": str(row["mentioned_at"]),
            "until_expiry": expiry_info,
            "supersedes_fact_id": supersedes_map[fid][0] if fid in supersedes_map else None,
            "superseded_by": sup_by,
        }
        by_ns.setdefault(ns, []).append(item)
        referenced_docs.add(did)
        total_facts += 1
        if sup_by:
            superseded_cnt += 1
        if expiry_info:
            expired_cnt += 1

    counts = {
        "facts_exported": total_facts,
        "documents_referenced": len(referenced_docs),
        "superseded_facts": superseded_cnt,
        "expired_facts": expired_cnt,
        "evicted_records": len(evicted_batches),
    }
    return by_ns, evicted_batches, counts


def _render_evicted_section(evicted_rows: list[dict[str, Any]]) -> str:
    if not evicted_rows:
        return ""
    lines = [
        "---",
        "",
        "## 回收站归档记录（`evicted_records`）",
        "",
        "| 批次 ID | 类型 | 记录 ID | 空间 | 原因 | 淘汰时间 | SHA-256 (前12位) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in evicted_rows:
        lines.append(
            f"| `{r['batch_id']}` | `{r['record_kind']}` | `{r['record_id']}` | "
            f"`{r['namespace']}` | {redact_credentials(str(r['reason']))} | "
            f"`{r['evicted_at']}` | `{str(r['content_sha256'])[:12]}` |"
        )
    lines.append("")
    return "\n".join(lines)


def export_markdown(
    *,
    db_path: str | Path,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    namespace: str | None = None,
    occurred_from: str | date | None = None,
    occurred_to: str | date | None = None,
    as_of: str | date | None = None,
    include_superseded: bool = True,
    include_evicted: bool = False,
    frontmatter: bool = True,
) -> dict[str, Any]:
    """Read-only export of ZeroMem SQLite facts and bound original evidence to Obsidian/Git Markdown."""
    if output_path is not None and output_dir is not None:
        raise ValueError("export-target-conflict: specify either output_path or output_dir, not both")

    db_file = Path(db_path).expanduser().resolve()
    if not db_file.is_file():
        raise ValueError(f"db-not-found: {db_file}")

    from_d, to_d = normalize_occurred_range(occurred_from, occurred_to)
    as_of_d = resolve_as_of_date(as_of)
    exported_at = _utc_now_iso()

    by_ns, evicted_rows, counts = _load_export_data(
        db_file,
        namespace=namespace,
        from_date=from_d,
        to_date=to_d,
        as_of_date=as_of_d,
        include_superseded=include_superseded,
        include_evicted=include_evicted,
    )
    namespaces = sorted(by_ns.keys())

    if output_dir is not None:
        out_root = Path(output_dir).expanduser().resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        used_stems: set[str] = set()
        written_files: list[str] = []
        index_rows: list[str] = []

        for ns in namespaces:
            facts = by_ns[ns]
            stem = _safe_namespace_stem(ns, used_stems)
            ns_file = out_root / f"{stem}.md"
            ns_docs = len({f["doc_id"] for f in facts})
            ns_sup = sum(1 for f in facts if f.get("superseded_by"))
            ns_exp = sum(1 for f in facts if f.get("until_expiry"))

            parts: list[str] = []
            if frontmatter:
                parts.extend(
                    [
                        "---",
                        f"generator: {_yaml_scalar('ZeroMem')}",
                        "read_only: true",
                        f"namespace: {_yaml_scalar(ns)}",
                        f"exported_at: {_yaml_scalar(exported_at)}",
                        f"as_of: {_yaml_scalar(as_of_d.isoformat())}",
                        f"facts: {len(facts)}",
                        f"documents: {ns_docs}",
                        f"superseded_facts: {ns_sup}",
                        f"expired_facts: {ns_exp}",
                        "---",
                        "",
                    ]
                )
            parts.append(f"# ZeroMem 空间：{ns}")
            parts.append("")
            parts.append(
                f"> 只读导出快照 · 导出时间 `{exported_at}` · 事实 `{len(facts)}` 条 · 文档 `{ns_docs}` 篇"
                + (f" · 时间过滤 `{from_d.isoformat() if from_d else '*'} ~ {to_d.isoformat() if to_d else '*'}`" if (from_d or to_d) else "")
            )
            parts.append("")
            for f in facts:
                parts.append(_render_fact_markdown(f))
            ns_evicted = [r for r in evicted_rows if str(r.get("namespace")) == ns]
            if ns_evicted:
                parts.append(_render_evicted_section(ns_evicted))

            _atomic_write_text(ns_file, "\n".join(parts).rstrip() + "\n")
            written_files.append(str(ns_file))
            index_rows.append(
                f"| [{ns}](./{stem}.md) | {len(facts)} | {ns_docs} | {ns_sup} | {ns_exp} |"
            )

        idx_parts: list[str] = []
        if frontmatter:
            idx_parts.extend(
                [
                    "---",
                    f"generator: {_yaml_scalar('ZeroMem')}",
                    "read_only: true",
                    f"exported_at: {_yaml_scalar(exported_at)}",
                    f"as_of: {_yaml_scalar(as_of_d.isoformat())}",
                    f"namespaces: {len(namespaces)}",
                    f"facts: {counts['facts_exported']}",
                    f"documents: {counts['documents_referenced']}",
                    "---",
                    "",
                ]
            )
        idx_parts.extend(
            [
                "# ZeroMem 本地记忆只读索引",
                "",
                f"- **导出时间**：`{exported_at}`",
                f"- **基准日期 (`as_of`)**：`{as_of_d.isoformat()}`",
                f"- **命名空间数**：`{len(namespaces)}` · **事实总数**：`{counts['facts_exported']}` · **关联文档**：`{counts['documents_referenced']}`",
                "",
            ]
        )
        if index_rows:
            idx_parts.extend(
                [
                    "| 空间 | 事实数 | 文档数 | 已替代 | 已过期 |",
                    "| :--- | :---: | :---: | :---: | :---: |",
                    *index_rows,
                    "",
                ]
            )
        else:
            idx_parts.append("_当前筛选条件下无匹配记忆。_\n")
        if evicted_rows:
            idx_parts.append(_render_evicted_section(evicted_rows))

        idx_file = out_root / "INDEX.md"
        _atomic_write_text(idx_file, "\n".join(idx_parts).rstrip() + "\n")
        written_files.insert(0, str(idx_file))

        return {
            "ok": True,
            "mode": "directory",
            "format": "markdown",
            "read_only": True,
            "db_path": str(db_file),
            "output_dir": str(out_root),
            "exported_at": exported_at,
            "as_of": as_of_d.isoformat(),
            "occurred_from": from_d.isoformat() if from_d else None,
            "occurred_to": to_d.isoformat() if to_d else None,
            "namespaces": namespaces,
            **counts,
            "files_written": written_files,
        }

    # Single Markdown document (either written to output_path or returned in-memory for stdout)
    parts = []
    if frontmatter:
        parts.extend(
            [
                "---",
                f"generator: {_yaml_scalar('ZeroMem')}",
                "read_only: true",
                f"exported_at: {_yaml_scalar(exported_at)}",
                f"as_of: {_yaml_scalar(as_of_d.isoformat())}",
                f"namespaces: {len(namespaces)}",
                f"facts: {counts['facts_exported']}",
                f"documents: {counts['documents_referenced']}",
                f"superseded_facts: {counts['superseded_facts']}",
                f"expired_facts: {counts['expired_facts']}",
                "---",
                "",
            ]
        )
    parts.append("# ZeroMem 本地记忆只读导出")
    parts.append("")
    parts.append(
        f"> 导出时间 `{exported_at}` · 空间 `{len(namespaces)}` 个 · 事实 `{counts['facts_exported']}` 条 · 文档 `{counts['documents_referenced']}` 篇"
        + (f" · 事件时间过滤 `{from_d.isoformat() if from_d else '*'} ~ {to_d.isoformat() if to_d else '*'}`" if (from_d or to_d) else "")
    )
    parts.append("")
    if not namespaces:
        parts.append("_当前筛选条件下无匹配记忆。_")
        parts.append("")
    for ns in namespaces:
        parts.append(f"## 空间：`{ns}`（{len(by_ns[ns])} 条）")
        parts.append("")
        for f in by_ns[ns]:
            parts.append(_render_fact_markdown(f))
    if evicted_rows:
        parts.append(_render_evicted_section(evicted_rows))

    md_text = "\n".join(parts).rstrip() + "\n"
    written_files = []
    out_file_str: str | None = None
    if output_path is not None:
        out_file = Path(output_path).expanduser().resolve()
        _atomic_write_text(out_file, md_text)
        out_file_str = str(out_file)
        written_files.append(out_file_str)

    res: dict[str, Any] = {
        "ok": True,
        "mode": "file" if output_path is not None else "stdout",
        "format": "markdown",
        "read_only": True,
        "db_path": str(db_file),
        "output_path": out_file_str,
        "exported_at": exported_at,
        "as_of": as_of_d.isoformat(),
        "occurred_from": from_d.isoformat() if from_d else None,
        "occurred_to": to_d.isoformat() if to_d else None,
        "namespaces": namespaces,
        **counts,
        "files_written": written_files,
    }
    if output_path is None:
        res["markdown"] = md_text
    return res


def backup_database(
    *,
    db_path: str | Path,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    verify: bool = True,
    write_sha256: bool = False,
    pages: int = 256,
) -> dict[str, Any]:
    """Create a clean, self-contained SQLite Online Backup snapshot (journal_mode=DELETE, zero WAL sidecars)."""
    started = time.monotonic()
    db_file = Path(db_path).expanduser().resolve()
    if not db_file.is_file():
        raise ValueError(f"db-not-found: {db_file}")

    now_dt = datetime.now(timezone.utc)
    stamp = now_dt.strftime("%Y%m%d-%H%M%S")
    default_name = f"zeromem-backup-{stamp}.db"

    if output_path is None or not str(output_path).strip():
        dest = (db_file.parent / "backups" / default_name).resolve()
    else:
        raw_out = str(output_path).strip()
        cand = Path(raw_out).expanduser()
        if cand.is_dir() or raw_out.endswith(("/", "\\")):
            dest = (cand / default_name).resolve()
        else:
            dest = cand.resolve()

    if dest == db_file:
        raise ValueError("backup-target-equals-source: destination must be different from active database")
    if dest.exists() and not overwrite:
        raise ValueError(f"backup-target-exists: {dest} (pass --overwrite to replace)")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_name(f".{dest.name}.tmp.{os.getpid()}.{now_dt.strftime('%Y%m%dT%H%M%S%fZ')}")

    src_conn: sqlite3.Connection | None = None
    dst_conn: sqlite3.Connection | None = None
    try:
        # Open source read-only via query_only pragma (safe over active WAL databases)
        src_conn = sqlite3.connect(str(db_file), timeout=20.0)
        src_conn.execute("PRAGMA query_only = ON")
        src_conn.execute("PRAGMA busy_timeout = 20000")

        dst_conn = sqlite3.connect(str(tmp_path), timeout=20.0)
        dst_conn.execute("PRAGMA busy_timeout = 20000")

        # Official SQLite Online Backup API (transactionally consistent even with active WAL writers)
        src_conn.backup(dst_conn, pages=max(1, int(pages)), sleep=0.01)

        src_conn.close()
        src_conn = None

        # Flatten WAL into a single self-contained file safe for cloud drives (OneDrive/Baidu/iCloud)
        jmode_row = dst_conn.execute("PRAGMA journal_mode = DELETE").fetchone()
        final_jmode = str(jmode_row[0]).lower() if jmode_row else "delete"
        dst_conn.commit()

        integrity = "skipped"
        if verify:
            ic_rows = dst_conn.execute("PRAGMA integrity_check").fetchall()
            if len(ic_rows) != 1 or str(ic_rows[0][0]).lower() != "ok":
                raise ValueError(f"backup-integrity-failed: {[r[0] for r in ic_rows][:5]}")
            integrity = "ok"

        app_id = int(dst_conn.execute("PRAGMA application_id").fetchone()[0])
        schema_row = (
            dst_conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if _has_table(dst_conn, "meta")
            else None
        )
        schema_ver = str(schema_row[0]) if schema_row else None

        counts = {
            "documents": int(dst_conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
            if _has_table(dst_conn, "documents")
            else 0,
            "segments": int(dst_conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0])
            if _has_table(dst_conn, "segments")
            else 0,
            "facts": int(dst_conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
            if _has_table(dst_conn, "facts")
            else 0,
            "evicted_records": int(dst_conn.execute("SELECT COUNT(*) FROM evicted_records").fetchone()[0])
            if _has_table(dst_conn, "evicted_records")
            else 0,
            "supersessions": int(dst_conn.execute("SELECT COUNT(*) FROM fact_supersessions").fetchone()[0])
            if _has_table(dst_conn, "fact_supersessions")
            else 0,
        }

        dst_conn.close()
        dst_conn = None

        # Remove any stale sidecar files on tmp path if present
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(tmp_path) + suffix)
            if sidecar.exists():
                sidecar.unlink(missing_ok=True)

        os.replace(tmp_path, dest)
    except BaseException:
        if src_conn is not None:
            try:
                src_conn.close()
            except Exception:
                pass
        if dst_conn is not None:
            try:
                dst_conn.close()
            except Exception:
                pass
        for p in (tmp_path, Path(str(tmp_path) + "-wal"), Path(str(tmp_path) + "-shm")):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    wal_sidecars = any(Path(str(dest) + suf).exists() for suf in ("-wal", "-shm"))
    digest = _sha256_file(dest)
    size_bytes = dest.stat().st_size

    sha_file_path: str | None = None
    if write_sha256:
        sha_target = Path(str(dest) + ".sha256")
        _atomic_write_text(sha_target, f"{digest}  {dest.name}\n")
        sha_file_path = str(sha_target)

    elapsed_ms = round((time.monotonic() - started) * 1000.0, 2)
    return {
        "ok": True,
        "method": "sqlite_online_backup",
        "source_db": str(db_file),
        "backup_path": str(dest),
        "created_at": now_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "journal_mode": final_jmode,
        "wal_sidecars_present": wal_sidecars,
        "integrity_check": integrity,
        "application_id": app_id,
        "application_id_valid": app_id == APP_FORMAT_ID,
        "schema_version": schema_ver,
        "size_bytes": size_bytes,
        "sha256": digest,
        "sha256_file": sha_file_path,
        "counts": counts,
        "elapsed_ms": elapsed_ms,
    }
