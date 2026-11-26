"""Offline governance, time-expiry, adaptive dual-lane, trace, and audit tests for ZeroMem."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from v2mem import Memory
from v2mem.audit import audit_rules_vs_memory, load_zeromem_memories, render_audit_markdown, write_audit_report
from v2mem.cli import build_parser, main as cli_main
from v2mem.contracts import Segment, _GENERIC_BEARER_WORDS
from v2mem.expiry import (
    ADAPTIVE_MAX_LIMIT,
    ADAPTIVE_MIN_LIMIT,
    ADAPTIVE_SCORE_RATIO,
    parse_until_date,
    partition_expired,
    select_adaptive_hits,
)
from v2mem.governance import resolve_archive_root, restore_evicted, retract_record, soft_evict
from v2mem.mcp_server import MCPStdioServer
from v2mem.store import APP_FORMAT_ID, SCHEMA_SIG, SQLiteMemoryStore
from v2mem.trace import DEFAULT_TRACE_FILE, RecallTracer, resolve_trace_dir, sanitize_query_preview, summarize_traces


def _seed_store(db_path: Path, docs: list[tuple[str, str, list[tuple[str, str]], list[dict[str, Any]]]]) -> SQLiteMemoryStore:
    """Directly seed documents/segments/facts into SQLiteMemoryStore without Ollama."""
    store = SQLiteMemoryStore(db_path)
    for idx, (doc_id, ns, seg_specs, facts) in enumerate(docs):
        segs = [Segment(id=sid, path=["note", str(i)], text=txt) for i, (sid, txt) in enumerate(seg_specs)]
        store.save_document_and_facts(
            doc_id=doc_id,
            namespace=ns,
            content_sha256=f"{idx:064x}",
            source_id=f"src-{doc_id}",
            mentioned_at="2026-05-01T00:00:00Z",
            created_at="2026-05-01T00:00:00Z",
            segments=segs,
            facts=facts,
        )
    return store


class TestGovernanceSuite(unittest.TestCase):
    def test_until_expiry_parsing_and_negative_control(self) -> None:
        # Strict calendar validation (fail-open on invalid dates)
        self.assertIsNone(parse_until_date(["until:2026-02-30"]))
        self.assertIsNone(parse_until_date(["until:2026-13-01"]))
        # Latest extension wins across compact and dashed forms
        latest = parse_until_date(["until:20260401", "until:2026-06-15"])
        assert latest is not None
        self.assertEqual(latest.isoformat(), "2026-06-15")

        # Inline marker in text or bound evidence
        d_ev = parse_until_date(evidence=[{"id": "s0", "text": "临时促销方案 until:2026-03-01 截止。"}])
        assert d_ev is not None
        self.assertEqual(d_ev.isoformat(), "2026-03-01")

        # Negative control: untagged items or ZEROMEM_UNTIL=0 leave order and dicts unchanged
        raw_items = [
            {"fact_id": 1, "doc_id": "d1", "text": "临时方案 [until:2026-01-10]", "score": 5.0},
            {"fact_id": 2, "doc_id": "d2", "text": "长期正式规范", "score": 3.0},
        ]
        disabled_out, disabled_cnt = partition_expired(raw_items, today="2026-05-01", env={"ZEROMEM_UNTIL": "0"})
        self.assertEqual(disabled_cnt, 0)
        self.assertEqual([x["fact_id"] for x in disabled_out], [1, 2])
        self.assertNotIn("until_expiry", disabled_out[0])

        # Active before expired; text untouched
        demoted, exp_cnt = partition_expired(raw_items, today="2026-05-01", env={})
        self.assertEqual(exp_cnt, 1)
        self.assertEqual([x["fact_id"] for x in demoted], [2, 1])
        self.assertEqual(demoted[1]["text"], "临时方案 [until:2026-01-10]")
        self.assertTrue(demoted[1]["until_expiry"]["expired"])
        self.assertEqual(demoted[1]["until_expiry"]["until"], "2026-01-10")

    def test_adaptive_dual_lane_score_gating_and_rollback(self) -> None:
        # 1) Niche query: 2 strong active hits + 1 strong expired contrast hit + 4 weak noise hits
        items = [
            {"fact_id": 1, "doc_id": "d1", "text": "strong active 1", "score": 4.0},
            {"fact_id": 2, "doc_id": "d2", "text": "strong active 2", "score": 3.2},
            {"fact_id": 3, "doc_id": "d3", "text": "weak noise 1", "score": 0.6},
            {"fact_id": 4, "doc_id": "d4", "text": "weak noise 2", "score": 0.5},
            {"fact_id": 5, "doc_id": "d5", "text": "weak noise 3", "score": 0.4},
            {
                "fact_id": 6,
                "doc_id": "d6",
                "text": "strong expired contrast",
                "score": 4.5,
                "until_expiry": {"until": "2026-01-01", "expired": True, "expired_days": 100, "label": "已过期（until 2026-01-01）"},
            },
        ]
        selected = select_adaptive_hits(
            items,
            limit=ADAPTIVE_MAX_LIMIT,
            min_limit=ADAPTIVE_MIN_LIMIT,
            score_ratio=ADAPTIVE_SCORE_RATIO,
            env={},
        )
        # Weak noise dropped (0.6 < 4.0 * 0.35 = 1.4), strong expired contrast preserved at tail
        self.assertEqual([x["fact_id"] for x in selected], [1, 2, 6])

        # 2) Hot query: 9 strong active hits expands up to 8 (not capped at 6)
        hot = [{"fact_id": i, "doc_id": f"d{i}", "text": f"hit {i}", "score": 5.0 - i * 0.1} for i in range(1, 10)]
        hot_sel = select_adaptive_hits(hot, env={})
        self.assertEqual(len(hot_sel), 8)

        # 3) Rollback flag reverts default limit to fixed 6
        rb_sel = select_adaptive_hits(hot, env={"ZEROMEM_ADAPTIVE_LIMIT_DISABLED": "1"})
        self.assertEqual(len(rb_sel), 6)

    def test_store_search_expiry_demotion_and_recall_trace(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "mem.db"
            trace_dir = root / "traces"

            _seed_store(
                db,
                [
                    (
                        "doc-expired",
                        "ops",
                        [("s0", "临时灰度端口 8888 仅测试使用 until:2026-01-15。")],
                        [
                            {
                                "text": "灰度网关测试端口 8888 临时生效 until:2026-01-15。",
                                "type": "world",
                                "entities": ["until:2026-01-15", "灰度网关"],
                                "evidence": ["s0"],
                                "action": "pass",
                            }
                        ],
                    ),
                    (
                        "doc-active",
                        "ops",
                        [("s0", "正式生产灰度网关端口固定为 9090。")],
                        [
                            {
                                "text": "灰度网关正式端口为 9090。",
                                "type": "world",
                                "entities": ["灰度网关"],
                                "evidence": ["s0"],
                                "action": "pass",
                            }
                        ],
                    ),
                ],
            )

            mem = Memory(db_path=db)

            # Default env (unset trace dir): zero trace files created
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("ZEROMEM_RECALL_LOG_DIR", None)
                os.environ.pop("MEMORY_RECALL_LOG_DIR", None)
                hits = mem.recall("灰度网关端口", ns="ops", as_of="2026-05-01")
                self.assertEqual(len(hits), 2)
                self.assertIn("9090", hits[0].text)
                self.assertIsNone(hits[0].until_expiry)
                self.assertNotIn("until_expiry", hits[0].to_dict())
                self.assertIn("8888", hits[1].text)
                assert hits[1].until_expiry is not None
                self.assertTrue(hits[1].until_expiry["expired"])
                self.assertFalse(trace_dir.exists())

            # Opt-in sidecar trace logging: metadata only, zero body leakage
            mem.recall("灰度网关端口", ns="ops", as_of="2026-05-01", recall_log_dir=trace_dir)
            mem.recall("不存在的关键词XYZ", ns="ops", as_of="2026-05-01", recall_log_dir=trace_dir)
            mem.recall("sk-1234567890abcdefghijklmnop 查密钥", ns="ops", recall_log_dir=trace_dir)

            trace_file = trace_dir / DEFAULT_TRACE_FILE
            self.assertTrue(trace_file.is_file())
            raw_log = trace_file.read_text(encoding="utf-8")
            # Zero body text or secret leakage
            self.assertNotIn("正式端口为 9090", raw_log)
            self.assertNotIn("sk-1234567890", raw_log)
            self.assertIn("[REDACTED_SENSITIVE_QUERY]", raw_log)

            summary = summarize_traces(trace_dir, days=30)
            self.assertEqual(summary["total_calls"], 3)
            self.assertEqual(summary["by_status"]["ok"], 1)
            self.assertEqual(summary["zero_hit_count"], 2)
            self.assertEqual(summary["expired_hits_total"], 1)

            # Disable switch beats explicit dir
            self.assertIsNone(resolve_trace_dir(trace_dir, env={"ZEROMEM_RECALL_LOG": "off"}))
            self.assertEqual(sanitize_query_preview("secret query", redact=True), "[REDACTED_BY_POLICY]")

            # Rolling rotation & fail-open
            rot_dir = root / "rot"
            now_t = [1_700_000_000.0]
            tracer = RecallTracer(rot_dir, max_bytes=200, keep_files=2, keep_days=30, now_fn=lambda: now_t[0])
            for i in range(10):
                now_t[0] += 5.0
                self.assertTrue(tracer.log({"channel": "test", "status": "ok", "seq": i, "pad": "x" * 60}))
            self.assertGreaterEqual(tracer.rotations, 2)
            rotated = [p for p in rot_dir.iterdir() if p.name != DEFAULT_TRACE_FILE]
            self.assertLessEqual(len(rotated), 2)

            # Fail-open when target path is an existing file (cannot mkdir)
            blocker = root / "not_a_dir"
            blocker.write_text("x", encoding="utf-8")
            bad_tracer = RecallTracer(blocker)
            self.assertFalse(bad_tracer.log({"channel": "test"}))
            self.assertEqual(bad_tracer.errors, 1)

    def test_pre_retraction_archive_ledger_and_fail_closed_guards(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "db_dir" / "mem.db"
            archive_root = root / "external_archive"
            ledger_dir = root / "ledger"

            _seed_store(
                db,
                [
                    (
                        "doc-1",
                        "proj",
                        [("s0", "旧版数据库端口 5432 已废弃。")],
                        [{"text": "旧版数据库端口 5432。", "type": "world", "entities": ["Postgres"], "evidence": ["s0"], "action": "pass"}],
                    ),
                    (
                        "doc-2",
                        "proj",
                        [("s0", "新版数据库端口 5433 正在使用。")],
                        [{"text": "新版数据库端口 5433。", "type": "world", "entities": ["Postgres"], "evidence": ["s0"], "action": "pass"}],
                    ),
                ],
            )

            # Fail-closed when archive_root missing or inside DB directory
            with self.assertRaises(ValueError):
                retract_record(db_path=db, doc_id="doc-1", reason="cleanup", env={})
            with self.assertRaises(ValueError):
                resolve_archive_root(db.parent / "nested", db_path=db, env={})

            # Dry-run by default: zero mutation, zero archive files
            dry = retract_record(
                db_path=db,
                doc_id="doc-1",
                reason="obsolete config",
                archive_root=archive_root,
                ledger_dir=ledger_dir,
                apply=False,
            )
            self.assertEqual(dry["mode"], "dry_run")
            self.assertFalse(dry["mutated"])
            self.assertEqual(SQLiteMemoryStore(db).stats()["documents"], 2)
            self.assertFalse(archive_root.exists())

            # Apply: archive verified on disk + body-free ledger + doc deleted
            res = retract_record(
                db_path=db,
                doc_id="doc-1",
                reason="obsolete config",
                archive_root=archive_root,
                ledger_dir=ledger_dir,
                apply=True,
            )
            self.assertTrue(res["mutated"])
            self.assertEqual(SQLiteMemoryStore(db).stats()["documents"], 1)
            archive_file = Path(res["ledger_receipt"]["archive_file"])
            self.assertTrue(archive_file.is_file())
            archived_obj = json.loads(archive_file.read_text(encoding="utf-8"))
            self.assertEqual(archived_obj["snapshot"]["document"]["doc_id"], "doc-1")
            ledger_text = (ledger_dir / "retractions.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("旧版数据库端口 5432", ledger_text)
            self.assertIn(res["ledger_receipt"]["content_sha256"], ledger_text)

    def test_soft_evict_conservation_and_tamper_checked_restore(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "mem.db"
            manifest = root / "evict-manifest.json"
            ledger_dir = root / "ledger"

            store = _seed_store(
                db,
                [
                    (
                        "doc-old",
                        "ops",
                        [("s0", "旧活动折扣 8 折 until:2026-01-01。")],
                        [{"text": "旧活动折扣 8 折 until:2026-01-01。", "type": "world", "entities": ["until:2026-01-01"], "evidence": ["s0"], "action": "pass"}],
                    ),
                    (
                        "doc-cur",
                        "ops",
                        [("s0", "当前长期会员权益保持 95 折。")],
                        [{"text": "当前长期会员权益保持 95 折。", "type": "world", "entities": ["会员权益"], "evidence": ["s0"], "action": "pass"}],
                    ),
                ],
            )

            # Refuse emptying 100% of active facts without --force
            with self.assertRaises(ValueError):
                soft_evict(db_path=db, doc_ids=["doc-old", "doc-cur"], reason="wipe", apply=True)
            self.assertEqual(store.stats()["facts"], 2)

            # Dry-run expired eviction
            dry = soft_evict(db_path=db, expired_only=True, as_of="2026-05-01", apply=False)
            self.assertFalse(dry["mutated"])
            self.assertEqual(len(dry["candidate_fact_ids"]), 1)

            # Apply expired eviction with manifest & ledger
            ev = soft_evict(
                db_path=db,
                expired_only=True,
                as_of="2026-05-01",
                reason="quarterly expired trim",
                batch_id="batch-q2",
                dump_manifest=manifest,
                ledger_dir=ledger_dir,
                apply=True,
            )
            self.assertTrue(ev["mutated"])
            self.assertTrue(ev["conservation_verified"])
            self.assertEqual(ev["moved_facts"], 1)
            self.assertEqual(store.stats()["facts"], 1)
            self.assertEqual(len(store.search("旧活动折扣")), 0)

            # Schema version & watermark invariants untouched
            conn = sqlite3.connect(str(db))
            try:
                self.assertEqual(conn.execute("PRAGMA application_id").fetchone()[0], APP_FORMAT_ID)
                self.assertEqual(conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0], "1")
                self.assertEqual(conn.execute("SELECT value FROM meta WHERE key='schema_sig'").fetchone()[0], SCHEMA_SIG)
            finally:
                conn.close()

            # Tamper test: corrupting payload_json aborts restore fail-closed with 0 rows changed
            conn = sqlite3.connect(str(db))
            with conn:
                orig_payload = conn.execute("SELECT payload_json FROM evicted_records WHERE batch_id='batch-q2'").fetchone()[0]
                conn.execute("UPDATE evicted_records SET payload_json = ? WHERE batch_id='batch-q2'", (orig_payload + " ",))
            conn.close()
            with self.assertRaises(ValueError):
                restore_evicted(db_path=db, batch_id="batch-q2", verify_manifest=manifest, apply=True)
            self.assertEqual(store.stats()["facts"], 1)

            # Restore genuine payload, then run verified atomic restore
            conn = sqlite3.connect(str(db))
            with conn:
                conn.execute("UPDATE evicted_records SET payload_json = ? WHERE batch_id='batch-q2'", (orig_payload,))
            conn.close()
            restored = restore_evicted(
                db_path=db,
                batch_id="batch-q2",
                verify_manifest=manifest,
                ledger_dir=ledger_dir,
                apply=True,
            )
            self.assertTrue(restored["mutated"])
            self.assertTrue(restored["fingerprint_verified"])
            self.assertTrue(restored["conservation_verified"])
            self.assertEqual(restored["restored_facts"], 1)
            self.assertEqual(store.stats()["facts"], 2)
            # Rebuilt FTS5 immediately recalls restored fact
            self.assertEqual(len(store.search("旧活动折扣", as_of="2026-05-01")), 1)

    def test_readonly_rules_memory_audit_and_cli_mcp_compat(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "mem.db"
            proj = root / "repo"
            proj.mkdir()
            (root / ".codex").mkdir()
            (root / ".gemini").mkdir()

            # Seed conflicting memory & expired memory
            _seed_store(
                db,
                [
                    (
                        "doc-model",
                        "ops",
                        [("s0", "本地推理服务默认使用 qwen3.5:9b，监听 127.0.0.1:11435，脚本 infer.py。")],
                        [
                            {
                                "text": "本地推理服务默认使用 qwen3.5:9b，监听 127.0.0.1:11435，脚本 infer.py。",
                                "type": "world",
                                "entities": ["qwen3.5:9b", "infer.py"],
                                "evidence": ["s0"],
                                "action": "pass",
                            },
                            {
                                "text": "临时活动规则 until:2025-12-31。",
                                "type": "world",
                                "entities": ["until:2025-12-31"],
                                "evidence": ["s0"],
                                "action": "pass",
                            },
                        ],
                    )
                ],
            )

            rule_a = root / ".gemini" / "AGENTS.md"
            rule_b = root / ".codex" / "AGENTS.md"
            rule_proj = proj / "AGENTS.md"

            shared_block = (
                "<!-- inference-default:begin -->\n"
                "## 默认推理模型\n"
                "本地推理服务长期默认使用 qwen3.5:4b-memory-v2，监听 127.0.0.1:11434，通过 infer.py 调用。\n"
                "<!-- inference-default:end -->\n"
            )
            drifted_block = (
                "<!-- inference-default:begin -->\n"
                "## 默认推理模型\n"
                "本客户端仅说明调用入口：运行 infer.py 并指定自定义模型。\n"
                "<!-- inference-default:end -->\n"
            )
            rule_a.write_text(shared_block, encoding="utf-8")
            rule_b.write_text(drifted_block, encoding="utf-8")
            rule_proj.write_text(shared_block, encoding="utf-8")

            mems, exp = load_zeromem_memories(db, as_of="2026-05-01")
            self.assertEqual(exp["until_tagged_expired"], 1)
            report = audit_rules_vs_memory(
                rule_files=[rule_a, rule_b, rule_proj],
                memory_records=mems,
                expiry_summary=exp,
            )
            self.assertEqual(report["summary"]["same_session_duplicate_count"], 2)
            self.assertEqual(report["summary"]["cross_host_drift_count"], 1)
            self.assertGreaterEqual(report["summary"]["conflict_count"], 1)

            md = render_audit_markdown(report)
            self.assertIn("Potential Conflicts", md)
            out_paths = write_audit_report(report, root / "audit-out")
            self.assertTrue(Path(out_paths["json"]).is_file())
            self.assertTrue(Path(out_paths["markdown"]).is_file())

            # MCP server preserves exact 4 tools and supports opt-in trace logging
            trace_dir = root / "mcp_traces"
            with patch.dict(os.environ, {"ZEROMEM_RECALL_LOG_DIR": str(trace_dir)}):
                srv = MCPStdioServer(Memory(db_path=db))
                tlist = srv.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
                assert tlist is not None
                self.assertEqual(
                    [t["name"] for t in tlist["result"]["tools"]],
                    ["memory_remember", "memory_recall", "memory_stats", "memory_delete"],
                )
                rcall = srv.handle_request(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {"name": "memory_recall", "arguments": {"query": "推理服务", "as_of": "2026-05-01"}},
                    }
                )
                assert rcall is not None
                self.assertFalse(rcall["result"]["isError"])
            self.assertEqual(summarize_traces(trace_dir)["by_channel"].get("zeromem_mcp"), 1)

            # CLI parser accepts all new governance subcommands cleanly
            with patch.dict(os.environ, {"USERPROFILE": str(root), "HOME": str(root)}):
                parser = build_parser()
                for argv in (
                    ["search", "推理", "--no-adaptive", "--as-of", "2026-05-01"],
                    ["trace-summary", "--dir", str(trace_dir), "--json"],
                    ["retract", "--doc-id", "doc-model", "--reason", "test", "--archive-root", str(root / "arch")],
                    ["evict", "--expired", "--as-of", "2026-05-01"],
                    ["restore", "--batch", "b1"],
                    ["audit", "--project", str(proj), "--json"],
                ):
                    parsed = parser.parse_args(argv)
                    self.assertTrue(parsed.command)
                self.assertEqual(cli_main(["--db", str(db), "audit", "--rules", str(rule_a), "--json"]), 0)

            # Verify watermark invariants remain untouched
            self.assertEqual("".join(w[0] for w in _GENERIC_BEARER_WORDS[:9]), "imliulian")


if __name__ == "__main__":
    unittest.main()
