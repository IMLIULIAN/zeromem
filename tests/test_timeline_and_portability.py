"""Offline tests for timeline evolution (supersedes_fact_id + occurred_from/to),
read-only Markdown export (`zmem export --md`), and SQLite Online Backup (`zmem backup`).
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

from v2mem import (
    Memory,
    backup_database,
    deduplicate_recall_evidence,
    export_markdown,
    format_recall_context,
)
from v2mem.cli import build_parser, main as cli_main
from v2mem.contracts import (
    FACT_ADMISSION_SYSTEM,
    FACT_SUPERSESSION_SYSTEM,
    Segment,
    has_supersession_cue,
    is_noop_revision,
)
from v2mem.governance import restore_evicted, soft_evict
from v2mem.mcp_server import MCPStdioServer
from v2mem.store import APP_FORMAT_ID, SCHEMA_SIG, SQLiteMemoryStore


class ScriptedClient:
    """Deterministic fake Ollama client capturing prompts and returning scripted JSON."""

    def __init__(self, responses: list[str], model: str = "mock-v2") -> None:
        self.responses = list(responses)
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def chat_json(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        *,
        num_predict: int = 1024,
        retries: int = 1,
        response_format: dict[str, Any] | str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_payload": user_payload,
                "num_predict": num_predict,
                "response_format": response_format,
            }
        )
        raw = self.responses.pop(0) if self.responses else "{}"
        return raw, {"model": self.model, "elapsed_sec": 0.01}


class TestTimelineAndPortability(unittest.TestCase):
    def test_prior_fact_lookup_supersedes_lineage_and_evict_restore(self) -> None:
        # Doc 1: 2025 decision (empty DB -> admission must keep exact legacy prompt without prior_facts)
        ext1 = json.dumps(
            {
                "facts": [
                    {
                        "text": "项目 A 采用方案 X（单体 MySQL）部署。",
                        "type": "world",
                        "occurred_start": "2025-06-01",
                        "occurred_end": None,
                        "where": "生产环境",
                        "entities": ["项目 A", "方案 X", "MySQL"],
                        "evidence": ["s0"],
                    }
                ]
            },
            ensure_ascii=False,
        )
        adm1 = json.dumps({"0": {"decision": "pass", "evidence": ["s0"], "replacement": ""}})

        # Doc 2: 2026 migration -> admission stays 100% pure native 3-field contract;
        # gated post-admission supersession call runs ONLY for slot 0 ("迁移至") against prior_facts=[#1]
        ext2 = json.dumps(
            {
                "facts": [
                    {
                        "text": "项目 A 从方案 X 迁移至方案 Y（SQLite WAL + FTS5）。",
                        "type": "world",
                        "occurred_start": "2026-03-10",
                        "occurred_end": None,
                        "where": "生产环境",
                        "entities": ["项目 A", "方案 Y", "SQLite"],
                        "evidence": ["s0"],
                    },
                    {
                        "text": "项目 A 日志保留周期为 30 天。",
                        "type": "world",
                        "occurred_start": "2026-03-10",
                        "occurred_end": None,
                        "where": None,
                        "entities": ["项目 A", "日志"],
                        "evidence": ["s1"],
                    },
                ]
            },
            ensure_ascii=False,
        )
        adm2 = json.dumps(
            {
                "0": {"decision": "pass", "evidence": ["s0"], "replacement": ""},
                "1": {"decision": "pass", "evidence": ["s1"], "replacement": ""},
            }
        )
        sup2 = json.dumps({"0": 1, "1": 9999})  # slot 1 wasn't even gated; fabricated ID 9999 must be ignored

        # Doc 3 in a DIFFERENT namespace -> must NOT see proj-a's facts or trigger supersession call
        ext3 = json.dumps(
            {
                "facts": [
                    {
                        "text": "项目 A 演示环境从方案 X 切换至方案 Z。",
                        "type": "world",
                        "occurred_start": "2026-04-01",
                        "occurred_end": None,
                        "where": None,
                        "entities": ["项目 A", "方案 Z"],
                        "evidence": ["s0"],
                    }
                ]
            },
            ensure_ascii=False,
        )
        adm3 = json.dumps({"0": {"decision": "pass", "evidence": ["s0"], "replacement": ""}})

        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "timeline.db"
            mock = ScriptedClient([ext1, adm1, ext2, adm2, sup2, ext3, adm3])
            mem = Memory(db_path=db, client=mock)

            r1 = mem.remember(
                "2025-06-01 决议：项目 A 采用方案 X（单体 MySQL）部署。",
                ns="proj-a",
                mentioned_at="2025-06-01T08:00:00Z",
            )
            self.assertEqual(r1.saved, 1)
            self.assertEqual(r1.superseded, 0)
            old_fid = r1.fact_ids[0]
            # First admission had no prior facts -> exact unmodified native prompt, payload & no forced schema
            self.assertEqual(len(mock.calls), 2)
            self.assertEqual(mock.calls[1]["system_prompt"], FACT_ADMISSION_SYSTEM)
            self.assertNotIn("prior_facts", mock.calls[1]["user_payload"])
            self.assertIsNone(mock.calls[1]["response_format"])

            r2 = mem.remember(
                "2026-03-10 迁移：项目 A 从方案 X 迁移至方案 Y（SQLite WAL + FTS5）。\n\n项目 A 日志保留周期为 30 天。",
                ns="proj-a",
                mentioned_at="2026-03-10T09:00:00Z",
            )
            self.assertEqual(r2.saved, 2)
            self.assertEqual(r2.superseded, 1)
            new_fid, log_fid = r2.fact_ids[0], r2.fact_ids[1]

            # Doc 2 admission (call 3) stays 100% pure native contract (no prior_facts, no schema pollution)
            self.assertEqual(mock.calls[3]["system_prompt"], FACT_ADMISSION_SYSTEM)
            self.assertNotIn("prior_facts", mock.calls[3]["user_payload"])
            self.assertIsNone(mock.calls[3]["response_format"])

            # Separate lightweight gated supersession call (call 4) checks ONLY slot "0" against old_fid
            self.assertEqual(mock.calls[4]["system_prompt"], FACT_SUPERSESSION_SYSTEM)
            self.assertEqual(list(mock.calls[4]["user_payload"]["new_facts"].keys()), ["0"])
            prior_ids = [pf["fact_id"] for pf in mock.calls[4]["user_payload"]["prior_facts"]]
            self.assertIn(old_fid, prior_ids)
            schema = mock.calls[4]["response_format"]
            assert isinstance(schema, dict)
            self.assertEqual(schema["properties"]["0"]["enum"], [None, old_fid])
            self.assertEqual(r2.facts[0].get("supersedes_fact_id"), old_fid)
            self.assertNotIn("supersedes_fact_id", r2.facts[1])

            # Cross-namespace isolation: proj-b runs strictly 2 calls (extract + pure admit), 0 supersession call
            mem.remember("2026-04-01：项目 A 演示环境从方案 X 切换至方案 Z。", ns="proj-b")
            self.assertEqual(len(mock.calls), 7)
            self.assertEqual(mock.calls[6]["system_prompt"], FACT_ADMISSION_SYSTEM)
            self.assertNotIn("prior_facts", mock.calls[6]["user_payload"])

            # Recall ranks current non-superseded facts ahead of superseded historical fact
            hits = mem.recall("项目 A 方案", ns="proj-a", adaptive=False, limit=10)
            hit_ids = [h.fact_id for h in hits]
            self.assertIn(new_fid, hit_ids)
            self.assertIn(old_fid, hit_ids)
            self.assertLess(hit_ids.index(new_fid), hit_ids.index(old_fid))

            by_id = {h.fact_id: h for h in hits}
            self.assertEqual(by_id[new_fid].supersedes_fact_id, old_fid)
            self.assertIsNone(by_id[new_fid].superseded_by)
            self.assertEqual(by_id[old_fid].superseded_by, [new_fid])
            self.assertIsNone(by_id[old_fid].supersedes_fact_id)
            # Negative control: unrelated fact omits both optional keys in to_dict()
            log_hits = mem.recall("日志保留", ns="proj-a")
            self.assertEqual(log_hits[0].fact_id, log_fid)
            self.assertNotIn("supersedes_fact_id", log_hits[0].to_dict())
            self.assertNotIn("superseded_by", log_hits[0].to_dict())

            # include_superseded=False hides old_fid cleanly
            active_only = mem.recall("项目 A 方案", ns="proj-a", include_superseded=False, adaptive=False)
            self.assertNotIn(old_fid, [h.fact_id for h in active_only])
            self.assertIn(new_fid, [h.fact_id for h in active_only])

            # Soft-evicting newer fact clears superseded_by on old_fid; restoring brings it back!
            ev = soft_evict(db_path=db, fact_ids=[new_fid], reason="rollback test", batch_id="b-sup", apply=True)
            self.assertEqual(ev["moved_facts"], 1)
            after_evict = mem.recall("项目 A 方案", ns="proj-a", adaptive=False)
            self.assertNotIn(new_fid, [h.fact_id for h in after_evict])
            self.assertEqual(after_evict[0].fact_id, old_fid)
            self.assertIsNone(after_evict[0].superseded_by)

            restored = restore_evicted(db_path=db, batch_id="b-sup", apply=True)
            self.assertEqual(restored["restored_facts"], 1)
            after_restore = {h.fact_id: h for h in mem.recall("项目 A 方案", ns="proj-a", adaptive=False)}
            self.assertEqual(after_restore[old_fid].superseded_by, [new_fid])
            self.assertEqual(after_restore[new_fid].supersedes_fact_id, old_fid)

            # SDK facade methods work end-to-end
            sdk_md = mem.export_markdown(ns="proj-a")
            self.assertTrue(sdk_md["ok"])
            self.assertIn("方案 Y", sdk_md["markdown"])
            sdk_bak = mem.backup(Path(td) / "sdk-snap.db")
            self.assertTrue(sdk_bak["ok"])
            self.assertEqual(sdk_bak["journal_mode"], "delete")

            # Schema v1 watermark invariants remain untouched
            conn = sqlite3.connect(str(db))
            try:
                self.assertEqual(conn.execute("PRAGMA application_id").fetchone()[0], APP_FORMAT_ID)
                self.assertEqual(conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0], "1")
                self.assertEqual(conn.execute("SELECT value FROM meta WHERE key='schema_sig'").fetchone()[0], SCHEMA_SIG)
            finally:
                conn.close()

    def test_occurred_from_to_filtering_and_mcp_cli_wiring(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "dates.db"
            store = SQLiteMemoryStore(db)
            store.save_document_and_facts(
                doc_id="doc-2025",
                namespace="eng",
                content_sha256="a" * 64,
                source_id="rfc-2025.md",
                mentioned_at="2025-06-02T00:00:00Z",
                created_at="2025-06-02T00:00:00Z",
                segments=[Segment(id="s0", path=["note", "0"], text="2025-06-01 项目 A 使用方案 X。")],
                facts=[
                    {
                        "text": "项目 A 使用方案 X。",
                        "type": "world",
                        "occurred_start": "2025-06-01",
                        "occurred_end": "2025-12-31",
                        "entities": ["项目 A", "方案 X"],
                        "evidence": ["s0"],
                        "action": "pass",
                    }
                ],
            )
            store.save_document_and_facts(
                doc_id="doc-2026",
                namespace="eng",
                content_sha256="b" * 64,
                source_id="rfc-2026.md",
                mentioned_at="2026-03-15T00:00:00Z",
                created_at="2026-03-15T00:00:00Z",
                segments=[
                    Segment(id="s0", path=["note", "0"], text="2026-03-15 项目 A 迁移到方案 Y。"),
                    Segment(id="s1", path=["note", "1"], text="项目 A 团队每周五做代码评审。"),
                ],
                facts=[
                    {
                        "text": "项目 A 迁移到方案 Y。",
                        "type": "world",
                        "occurred_start": "2026-03-15",
                        "occurred_end": None,
                        "entities": ["项目 A", "方案 Y"],
                        "evidence": ["s0"],
                        "action": "pass",
                        "supersedes_fact_id": 1,
                    },
                    {
                        "text": "项目 A 团队每周五做代码评审。",
                        "type": "world",
                        "occurred_start": None,
                        "occurred_end": None,
                        "entities": ["项目 A"],
                        "evidence": ["s1"],
                        "action": "pass",
                    },
                ],
            )

            mem = Memory(db_path=db)

            # 1) 2025 year shorthand matches only doc-2025 fact (excludes undated & 2026 facts)
            hits_2025 = mem.recall("项目 A", ns="eng", occurred_from="2025", occurred_to="2025")
            self.assertEqual([h.fact_id for h in hits_2025], [1])
            self.assertEqual(hits_2025[0].superseded_by, [2])

            # 2) Month shorthand 2026-03 matches only doc-2026 dated fact
            hits_mar = mem.recall("项目 A", ns="eng", occurred_from="2026-03", occurred_to="2026-03")
            self.assertEqual([h.fact_id for h in hits_mar], [2])
            self.assertEqual(hits_mar[0].supersedes_fact_id, 1)

            # 3) Empty query + date range returns timeline facts without undated rows
            hits_timeline = mem.recall("", ns="eng", occurred_from="2025-01-01", occurred_to="2026-12-31")
            self.assertEqual([h.fact_id for h in hits_timeline], [2, 1])

            # 4) Interval overlap: query 2025-11-01..2025-11-30 overlaps fact #1 (2025-06-01..2025-12-31)
            hits_overlap = mem.recall("", ns="eng", occurred_from="2025-11-01", occurred_to="2025-11-30")
            self.assertEqual([h.fact_id for h in hits_overlap], [1])

            # 5) Invalid dates & inverted range fail closed
            with self.assertRaises(ValueError):
                mem.recall("项目 A", occurred_from="2026-02-30")
            with self.assertRaises(ValueError):
                mem.recall("项目 A", occurred_from="2026-05-01", occurred_to="2026-01-01")

            # 5b) Deterministic guard backfills single grounded event date when model leaves occurred_start null,
            # while never leaking record date or until: deadline into occurred_start
            from v2mem.contracts import guard_fact
            smap = {
                "s0": Segment(id="s0", path=["note", "0"], text="2026-03-01 记录：服务将于 2026-03-15 迁移，临时方案 until:2026-03-20。"),
                "s1": Segment(id="s1", path=["note", "1"], text="临时灰度规则 until:2026-04-01。"),
            }
            g_evt, _ = guard_fact(
                {"text": "2026-03-01 记录：服务将于 2026-03-15 迁移。", "type": "world", "occurred_start": None, "evidence": ["s0"]},
                smap,
                mentioned_at="2026-03-01T08:00:00Z",
            )
            assert g_evt is not None
            self.assertEqual(g_evt["occurred_start"], "2026-03-15")
            g_until, _ = guard_fact(
                {"text": "临时灰度规则 until:2026-04-01。", "type": "world", "occurred_start": None, "evidence": ["s1"]},
                smap,
                mentioned_at="2026-03-01T08:00:00Z",
            )
            assert g_until is not None
            self.assertIsNone(g_until["occurred_start"])

            # 6) MCP memory_recall forwards occurred_from / occurred_to / include_superseded & keeps 4 tools
            srv = MCPStdioServer(mem)
            tlist = srv.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            assert tlist is not None
            self.assertEqual(
                [t["name"] for t in tlist["result"]["tools"]],
                ["memory_remember", "memory_recall", "memory_stats", "memory_delete"],
            )
            resp = srv.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "memory_recall",
                        "arguments": {
                            "query": "项目 A",
                            "namespace": "eng",
                            "occurred_from": "2025-01-01",
                            "occurred_to": "2025-12-31",
                        },
                    },
                }
            )
            assert resp is not None
            self.assertFalse(resp["result"]["isError"])
            payload = json.loads(resp["result"]["content"][0]["text"])
            self.assertEqual([it["fact_id"] for it in payload["items"]], [1])

    def test_readonly_markdown_export_file_stdout_and_obsidian_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "export.db"
            store = SQLiteMemoryStore(db)
            store.save_document_and_facts(
                doc_id="doc-a",
                namespace="project:alpha",
                content_sha256="1" * 64,
                source_id="notes/alpha.md",
                mentioned_at="2025-08-01T10:00:00Z",
                created_at="2025-08-01T10:00:00Z",
                segments=[
                    Segment(id="s0", path=["note", "0"], text="2025-08-01 初版端口 8080，临时规则 until:2025-12-31。", role="user")
                ],
                facts=[
                    {
                        "text": "初版服务端口为 8080（until:2025-12-31）。",
                        "type": "world",
                        "occurred_start": "2025-08-01",
                        "entities": ["project:alpha", "until:2025-12-31"],
                        "evidence": ["s0"],
                        "action": "pass",
                    }
                ],
            )
            store.save_document_and_facts(
                doc_id="doc-b",
                namespace="project:alpha",
                content_sha256="2" * 64,
                source_id="notes/alpha-v2.md",
                mentioned_at="2026-02-01T10:00:00Z",
                created_at="2026-02-01T10:00:00Z",
                segments=[
                    Segment(id="s0", path=["note", "0"], text="2026-02-01 端口改为 9090，禁用旧 8080。"),
                ],
                facts=[
                    {
                        "text": "正式服务端口迁移为 9090。",
                        "type": "world",
                        "occurred_start": "2026-02-01",
                        "entities": ["project:alpha"],
                        "evidence": ["s0"],
                        "action": "revise",
                        "supersedes_fact_id": 1,
                    }
                ],
            )
            store.save_document_and_facts(
                doc_id="doc-c",
                namespace="ops/sec",
                content_sha256="3" * 64,
                source_id="ops.md",
                mentioned_at="2026-03-01T00:00:00Z",
                created_at="2026-03-01T00:00:00Z",
                segments=[Segment(id="s0", path=["note", "0"], text="使用独立只读账号同步日志。")],
                facts=[
                    {
                        "text": "使用独立只读账号同步日志。",
                        "type": "experience",
                        "occurred_start": "2026-03-01",
                        "entities": ["日志"],
                        "evidence": ["s0"],
                        "action": "pass",
                    }
                ],
            )

            before_bytes = db.read_bytes()

            # 1) Single-file export
            single_md = root / "all.md"
            res_file = export_markdown(db_path=db, output_path=single_md, as_of="2026-05-01")
            self.assertTrue(res_file["ok"])
            self.assertTrue(res_file["read_only"])
            self.assertEqual(res_file["facts_exported"], 3)
            self.assertEqual(res_file["superseded_facts"], 1)
            self.assertEqual(res_file["expired_facts"], 1)
            md_content = single_md.read_text(encoding="utf-8")
            self.assertIn("generator: \"ZeroMem\"", md_content)
            self.assertIn("`替代 #1`", md_content)
            self.assertIn("`已被 #2 替代`", md_content)
            self.assertIn("已过期（until 2025-12-31）", md_content)
            self.assertIn("> **原文证据 (`s0` · `user`)**: 2025-08-01 初版端口 8080", md_content)
            self.assertIn("> **原文证据 (`s0`)**: 2026-02-01 端口改为 9090", md_content)

            # 2) Obsidian / Git per-namespace directory export
            vault_dir = root / "vault"
            res_dir = export_markdown(db_path=db, output_dir=vault_dir, as_of="2026-05-01")
            self.assertEqual(res_dir["mode"], "directory")
            self.assertTrue((vault_dir / "INDEX.md").is_file())
            self.assertTrue((vault_dir / "project_alpha.md").is_file())
            self.assertTrue((vault_dir / "ops_sec.md").is_file())
            index_txt = (vault_dir / "INDEX.md").read_text(encoding="utf-8")
            self.assertIn("[project:alpha](./project_alpha.md)", index_txt)
            self.assertIn("[ops/sec](./ops_sec.md)", index_txt)

            # 3) CLI `zmem export --md` stdout + date slice
            buf = io.StringIO()
            with patch.dict(os.environ, {"USERPROFILE": str(root), "HOME": str(root)}):
                with redirect_stdout(buf):
                    rc = cli_main(
                        [
                            "--db",
                            str(db),
                            "export",
                            "--md",
                            "--ns",
                            "project:alpha",
                            "--occurred-from",
                            "2026-01-01",
                        ]
                    )
            self.assertEqual(rc, 0)
            stdout_md = buf.getvalue()
            self.assertIn("正式服务端口迁移为 9090", stdout_md)
            self.assertNotIn("初版服务端口为 8080", stdout_md)

            # Source DB untouched (100% read-only)
            self.assertEqual(db.read_bytes(), before_bytes)

    def test_online_backup_flattens_wal_and_guards_targets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "live.db"
            store = SQLiteMemoryStore(db)
            store.save_document_and_facts(
                doc_id="doc-1",
                namespace="prod",
                content_sha256="c" * 64,
                source_id="prod.md",
                mentioned_at="2026-01-01T00:00:00Z",
                created_at="2026-01-01T00:00:00Z",
                segments=[Segment(id="s0", path=["note", "0"], text="主库连接池上限 200。")],
                facts=[
                    {
                        "text": "主库连接池上限 200。",
                        "type": "world",
                        "occurred_start": "2026-01-01",
                        "entities": ["主库"],
                        "evidence": ["s0"],
                        "action": "pass",
                    }
                ],
            )

            # Keep an active WAL connection open while performing online backup!
            live_conn = sqlite3.connect(str(db))
            try:
                live_conn.execute("PRAGMA journal_mode = WAL")
                # Perform a write via store while live_conn holds a read snapshot -> ensures -wal has live frames
                live_conn.execute("BEGIN")
                live_conn.execute("SELECT COUNT(*) FROM facts").fetchone()

                store.save_document_and_facts(
                    doc_id="doc-2",
                    namespace="prod",
                    content_sha256="d" * 64,
                    source_id="prod-v2.md",
                    mentioned_at="2026-04-01T00:00:00Z",
                    created_at="2026-04-01T00:00:00Z",
                    segments=[Segment(id="s0", path=["note", "0"], text="2026-04-01 主库连接池上限扩至 400。")],
                    facts=[
                        {
                            "text": "主库连接池上限扩至 400。",
                            "type": "world",
                            "occurred_start": "2026-04-01",
                            "entities": ["主库"],
                            "evidence": ["s0"],
                            "action": "pass",
                            "supersedes_fact_id": 1,
                        }
                    ],
                )
                # Confirm source DB actually has an active WAL file right now
                self.assertTrue(Path(str(db) + "-wal").exists())

                # Online backup succeeds cleanly despite active WAL + open reader!
                snap = root / "cloud_sync" / "zeromem-snap.db"
                receipt = backup_database(db_path=db, output_path=snap, write_sha256=True)
            finally:
                live_conn.close()

            self.assertTrue(receipt["ok"])
            self.assertEqual(receipt["method"], "sqlite_online_backup")
            self.assertEqual(receipt["journal_mode"], "delete")
            self.assertFalse(receipt["wal_sidecars_present"])
            self.assertEqual(receipt["integrity_check"], "ok")
            self.assertTrue(receipt["application_id_valid"])
            self.assertEqual(receipt["schema_version"], "1")
            self.assertEqual(receipt["counts"]["facts"], 2)
            self.assertEqual(receipt["counts"]["supersessions"], 1)
            self.assertFalse(Path(str(snap) + "-wal").exists())
            self.assertFalse(Path(str(snap) + "-shm").exists())
            self.assertTrue(Path(receipt["sha256_file"]).is_file())

            # Snapshot is immediately usable as a standalone ZeroMem DB with full lineage
            snap_mem = Memory(db_path=snap)
            hits = snap_mem.recall("主库连接池", ns="prod", adaptive=False)
            self.assertEqual([h.fact_id for h in hits], [2, 1])
            self.assertEqual(hits[0].supersedes_fact_id, 1)
            self.assertEqual(hits[1].superseded_by, [2])

            # Safety guards: refuse overwriting without --overwrite; refuse backing up onto source
            with self.assertRaises(ValueError):
                backup_database(db_path=db, output_path=snap, overwrite=False)
            with self.assertRaises(ValueError):
                backup_database(db_path=db, output_path=db, overwrite=True)

            # Directory target auto-names snapshot; CLI `--json` returns clean receipt
            bdir = root / "auto_backups"
            bdir.mkdir()
            buf = io.StringIO()
            with patch.dict(os.environ, {"USERPROFILE": str(root), "HOME": str(root)}):
                with redirect_stdout(buf):
                    rc = cli_main(["--db", str(db), "backup", str(bdir), "--json"])
            self.assertEqual(rc, 0)
            cli_receipt = json.loads(buf.getvalue())
            self.assertTrue(cli_receipt["ok"])
            self.assertTrue(Path(cli_receipt["backup_path"]).is_file())
            self.assertEqual(Path(cli_receipt["backup_path"]).parent.resolve(), bdir.resolve())

            # Parser accepts all new flags cleanly
            with patch.dict(os.environ, {"USERPROFILE": str(root), "HOME": str(root)}):
                parser = build_parser()
                for argv in (
                    ["search", "主库", "--occurred-from", "2026-01-01", "--occurred-to", "2026-12-31", "--hide-superseded"],
                    ["export", "--md", "--dir", str(root / "v"), "--include-evicted", "--no-frontmatter"],
                    ["backup", "-o", str(snap), "--overwrite", "--sha256", "--json"],
                ):
                    parsed = parser.parse_args(argv)
                    self.assertTrue(parsed.command)

    def test_noop_revise_intercepted_and_boilerplate_gating_skipped(self) -> None:
        self.assertTrue(is_noop_revision("User authorized project v1.", "User authorized project v1."))
        self.assertTrue(is_noop_revision("User authorized project v1. | Involving: user", "user authorized   project v1."))
        self.assertFalse(is_noop_revision("Pool limit is 500.", "Pool limit is 200."))
        self.assertTrue(has_supersession_cue("支付回调超时阈值正式调整为 15 秒。"))
        self.assertFalse(has_supersession_cue("运维团队每周日凌晨 03:00 执行冷备归档。"))

        ext1 = json.dumps(
            {
                "facts": [
                    {
                        "text": "Checklist confirms text contrast, icon clarity, and control spacing across desktop resolutions and dark/light themes without runtime changes.",
                        "type": "world",
                        "entities": ["checklist"],
                        "evidence": ["s0"],
                    }
                ]
            }
        )
        adm1 = json.dumps({"0": {"decision": "pass", "evidence": ["s0"], "replacement": ""}})

        # Doc 2 has: (0) a no-op revise repeating polluted text identically -> MUST be intercepted as held;
        # (1) a genuine revise changing 500 -> 200 -> MUST pass as revised;
        # (2) a routine additive fact with no supersession cue -> MUST NOT trigger any extra LLM call!
        ext2 = json.dumps(
            {
                "facts": [
                    {
                        "text": "User authorized project execution for version 1.0.0.",
                        "type": "world",
                        "entities": [],
                        "evidence": ["s0"],
                    },
                    {
                        "text": "主库连接池上限 500。",
                        "type": "world",
                        "entities": ["主库"],
                        "evidence": ["s0"],
                    },
                    {
                        "text": "每周日凌晨执行一次只读日志巡检。",
                        "type": "world",
                        "entities": ["日志"],
                        "evidence": ["s0"],
                    },
                ]
            },
            ensure_ascii=False,
        )
        adm2 = json.dumps(
            {
                "0": {
                    "decision": "revise",
                    "evidence": ["s0"],
                    "replacement": "User authorized project execution for version 1.0.0. | Involving: user",
                },
                "1": {
                    "decision": "revise",
                    "evidence": ["s0"],
                    "replacement": "主库连接池上限 200。",
                },
                "2": {"decision": "pass", "evidence": ["s0"], "replacement": ""},
            },
            ensure_ascii=False,
        )

        with tempfile.TemporaryDirectory() as td:
            mock = ScriptedClient([ext1, adm1, ext2, adm2])
            mem = Memory(db_path=Path(td) / "gate.db", client=mock)
            mem.remember("核对清单：确认亮暗色主题文字对比度与图标清晰度。", ns="ops")
            r2 = mem.remember("主库连接池上限 200，每周日凌晨执行只读日志巡检。", ns="ops")
            # Strictly 2 + 2 = 4 calls total (zero extra supersession call triggered!)
            self.assertEqual(len(mock.calls), 4)
            self.assertEqual(r2.saved, 2)
            self.assertEqual(r2.revised, 1)
            self.assertEqual(r2.held, 1)
            self.assertTrue(all("User authorized" not in f["text"] for f in r2.facts))

    def test_recall_evidence_dedup_and_verbatim_folding(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "dedup.db"
            store = SQLiteMemoryStore(db)
            shared_seg = "2026-03-05 数据库配置确认：主库连接池上限 200，只读副本连接池上限 500。"
            verbatim_seg = "2026-03-01 记录：网关公开查询接口限流为每分钟 120 次。"
            store.save_document_and_facts(
                doc_id="doc-shared",
                namespace="eng",
                content_sha256="e" * 64,
                source_id="db.md",
                mentioned_at="2026-03-05T08:00:00Z",
                created_at="2026-03-05T08:00:00Z",
                segments=[
                    Segment(id="s0", path=["note", "0"], text=shared_seg, role="user"),
                    Segment(id="s1", path=["note", "1"], text=verbatim_seg),
                ],
                facts=[
                    {
                        "text": "主库连接池上限设置为 200。",
                        "type": "world",
                        "entities": ["主库"],
                        "evidence": ["s0"],
                        "action": "pass",
                    },
                    {
                        "text": "只读副本连接池上限设置为 500。",
                        "type": "world",
                        "entities": ["只读副本"],
                        "evidence": ["s0"],
                        "action": "pass",
                    },
                    {
                        "text": verbatim_seg,
                        "type": "world",
                        "entities": ["网关"],
                        "evidence": ["s1"],
                        "action": "pass",
                    },
                ],
            )
            mem = Memory(db_path=db)
            hits = mem.recall("", ns="eng", limit=5, adaptive=False)
            self.assertEqual(len(hits), 3)

            # Deduplication never mutates original RecallHit objects
            deduped, stats = deduplicate_recall_evidence(hits, fold_verbatim=True)
            self.assertEqual(stats["shared_deduped"], 1)
            self.assertEqual(stats["verbatim_folded"], 1)
            self.assertEqual(stats["total_saved"], 2)
            # Raw hits still hold full evidence text intact
            self.assertEqual(sum(len(h.evidence[0]["text"]) > 0 for h in hits), 3)

            # Context formatter omits duplicated shared segment & verbatim segment while preserving citations
            ctx = format_recall_context(hits)
            self.assertEqual(ctx.count(shared_seg), 1)
            self.assertEqual(ctx.count(verbatim_seg), 1)
            self.assertIn("s0同#", ctx)
            self.assertIn("s1同正文", ctx)

            # MCP memory_recall emits deduplicated evidence payload with evidence_dedup_saved count
            srv = MCPStdioServer(mem)
            resp = srv.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 9,
                    "method": "tools/call",
                    "params": {"name": "memory_recall", "arguments": {"query": "", "namespace": "eng", "limit": 5, "adaptive": False}},
                }
            )
            assert resp is not None
            body = json.loads(resp["result"]["content"][0]["text"])
            self.assertEqual(body["evidence_dedup_saved"], 2)
            ev_entries = [ev for it in body["items"] for ev in it["evidence"]]
            self.assertTrue(any("dedup_ref" in e and "text" not in e for e in ev_entries))
            self.assertTrue(any(e.get("same_as_fact") is True and "text" not in e for e in ev_entries))


if __name__ == "__main__":
    unittest.main()
