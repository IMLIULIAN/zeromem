"""Offline unit tests for v2-memory (zero network / zero Ollama required)."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any

from v2mem import Memory
from v2mem.contracts import (
    Segment,
    build_segments,
    guard_fact,
    parse_json_object,
    redact_credentials,
    window_segments,
)
from v2mem.hud import collect_status, format_bytes
from v2mem.mcp_server import MCPStdioServer
from v2mem.ollama import OllamaError
from v2mem.store import APP_FORMAT_ID, SCHEMA_SIG


class MockOllamaClient:
    """Deterministic fake client capturing call counts and returning scripted responses."""

    def __init__(
        self,
        responses: list[str],
        model: str = "mock-v2",
        *,
        running: list[str] | None = None,
        offline: bool = False,
    ) -> None:
        self.responses = list(responses)
        self.model = model
        self._explicit_model: str | None = None
        self._running = list(running or [])
        self._offline = offline
        self.calls: list[dict[str, Any]] = []

    def list_running_models(self) -> list[str]:
        if self._offline:
            raise OllamaError("offline")
        return list(self._running)

    def resolve_model(self) -> str:
        if self._offline:
            raise OllamaError("offline")
        return self.model

    @property
    def base_url(self) -> str:
        return "http://127.0.0.1:11434"

    def chat_json(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        *,
        num_predict: int = 1024,
        retries: int = 1,
    ) -> tuple[str, dict[str, Any]]:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_payload": user_payload,
                "num_predict": num_predict,
            }
        )
        raw = self.responses.pop(0) if self.responses else "{}"
        return raw, {"model": self.model, "elapsed_sec": 0.01}


class TestCore(unittest.TestCase):
    def test_short_window_chunker_with_overlap(self) -> None:
        paragraphs = [f"第{i}段内容：" + ("测试语句。" * 45) for i in range(10)]
        text = "\n\n".join(paragraphs)
        segs = build_segments(text, max_seg_chars=300)
        self.assertGreater(len(segs), 8)
        self.assertEqual(segs[0].id, "s0")

        wins = window_segments(segs, max_window_chars=900, overlap=1)
        self.assertGreaterEqual(len(wins), 3)
        for win in wins:
            self.assertLessEqual(sum(len(s.text) for s in win), 1200)
        # Verify 1-segment overlap between adjacent windows
        self.assertEqual(wins[0][-1].id, wins[1][0].id)

    def test_deterministic_guard_rules(self) -> None:
        full_sha = "ffcf9e3e1234567890abcdef1234567890abcdef1234567890abcdef12345678"
        dropped_first_char_sha = full_sha[1:]
        seg_map = {
            "s0": Segment(
                id="s0",
                path=["note", "0"],
                text=f"Release archive SHA-256 is {full_sha}. Auth uses Bearer token header.",
            )
        }

        # 1) 63-char truncated hash is auto-repaired from bound evidence; Bearer token is NOT redacted
        guarded, reason = guard_fact(
            {
                "text": f"Release hash {dropped_first_char_sha} uses Bearer token header.",
                "type": "world",
                "occurred_start": "2026-05-01",
                "occurred_end": None,
                "where": None,
                "entities": ["Release"],
                "evidence": ["s0", "s99"],  # s99 hallucinated -> stripped
            },
            seg_map,
            mentioned_at="2026-05-01T12:00:00Z",
        )
        self.assertIsNone(reason)
        assert guarded is not None
        self.assertIn(full_sha, guarded["text"])
        self.assertIn("Bearer token", guarded["text"])
        self.assertEqual(guarded["evidence"], ["s0"])
        # Date equal to mentioned_at and absent from evidence is cleared
        self.assertIsNone(guarded["occurred_start"])

        # 2) Completely fabricated 64-char hash is rejected
        bad, bad_reason = guard_fact(
            {
                "text": "Digest is 1111111111111111111111111111111111111111111111111111111111111111.",
                "type": "world",
                "evidence": ["s0"],
            },
            seg_map,
        )
        self.assertIsNone(bad)
        self.assertEqual(bad_reason, "ungrounded_long_hash")

        # 3) Real bearer secrets and API keys are masked
        masked = redact_credentials(
            "Use Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload and sk-1234567890abcdefghijklmno"
        )
        self.assertNotIn("eyJhbGci", masked)
        self.assertNotIn("sk-123456", masked)
        self.assertIn("[REDACTED]", masked)

    def test_single_round_admission_no_deadlock_and_bilingual_fts(self) -> None:
        extract_resp = json.dumps(
            {
                "facts": [
                    {
                        "text": "API gateway rate limit increases to 120 req/min on 2026-03-15.",
                        "type": "world",
                        "occurred_start": "2026-03-15",
                        "occurred_end": None,
                        "where": None,
                        "entities": ["API gateway"],
                        "evidence": ["s0"],
                    },
                    {
                        "text": "Primary DB pool limit is 500.",
                        "type": "world",
                        "occurred_start": None,
                        "occurred_end": None,
                        "where": None,
                        "entities": ["Primary DB"],
                        "evidence": ["s1"],
                    },
                    {
                        "text": "Invented claim not supported by text.",
                        "type": "world",
                        "occurred_start": None,
                        "occurred_end": None,
                        "where": None,
                        "entities": [],
                        "evidence": ["s0"],
                    },
                    {
                        "text": "Ambiguous claim put on hold.",
                        "type": "observation",
                        "occurred_start": None,
                        "occurred_end": None,
                        "where": None,
                        "entities": [],
                        "evidence": ["s0"],
                    },
                ]
            },
            ensure_ascii=False,
        )
        admit_resp = json.dumps(
            {
                "0": {"decision": "pass", "evidence": ["s0"], "replacement": ""},
                "1": {
                    "decision": "revise",
                    "evidence": ["s1"],
                    "replacement": "Primary DB pool limit is 200 and read replica limit is 500.",
                },
                "2": {"decision": "drop", "evidence": [], "replacement": ""},
                "3": {"decision": "hold", "evidence": [], "replacement": ""},
            },
            ensure_ascii=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            mock = MockOllamaClient([extract_resp, admit_resp])
            mem = Memory(db_path=db_path, client=mock)

            src_text = "2026-03-01 记录：网关限流将在 2026-03-15 调为 120。\n\n主库连接池上限 200，只读副本 500。"
            res = mem.remember(src_text, ns="proj-a", mentioned_at="2026-03-01T08:00:00Z", verify=True)

            # Strictly 2 LLM calls (1 extract + 1 admit); never re-judges revised items
            self.assertEqual(len(mock.calls), 2)
            self.assertEqual(res.extracted, 4)
            self.assertEqual(res.saved, 2)
            self.assertEqual(res.revised, 1)
            self.assertEqual(res.dropped, 1)
            self.assertEqual(res.held, 1)

            # Idempotent deduplication on identical input
            dup = mem.remember(src_text, ns="proj-a", mentioned_at="2026-03-01T08:00:00Z", verify=True)
            self.assertTrue(dup.deduplicated)
            self.assertEqual(len(mock.calls), 2)

            # Bilingual dual-track recall: Chinese query hits English-drifted facts via bound Chinese evidence
            hits = mem.recall("网关限流", ns="proj-a", limit=5)
            self.assertGreaterEqual(len(hits), 1)
            self.assertIn("120", hits[0].text)
            self.assertEqual(hits[0].occurred_start, "2026-03-15")
            self.assertIn("网关限流", hits[0].evidence[0]["text"])

            hits_pool = mem.recall("连接池上限", ns="proj-a", limit=5)
            self.assertTrue(any("200" in h.text and h.action == "revise" for h in hits_pool))

            # Verify SQLite header format metadata
            conn = sqlite3.connect(str(db_path))
            try:
                app_id = conn.execute("PRAGMA application_id").fetchone()[0]
                sig = conn.execute("SELECT value FROM meta WHERE key = 'schema_sig'").fetchone()[0]
            finally:
                conn.close()
            self.assertEqual(app_id, APP_FORMAT_ID)
            self.assertEqual(sig, SCHEMA_SIG)

    def test_mcp_stdio_protocol(self) -> None:
        extract_resp = json.dumps(
            {
                "facts": [
                    {
                        "text": "Service port changed from 8080 to 9090.",
                        "type": "world",
                        "occurred_start": None,
                        "occurred_end": None,
                        "where": None,
                        "entities": ["Service"],
                        "evidence": ["s0"],
                    }
                ]
            }
        )
        admit_resp = json.dumps({"0": {"decision": "pass", "evidence": ["s0"], "replacement": ""}})

        with tempfile.TemporaryDirectory() as tmpdir:
            mem = Memory(db_path=Path(tmpdir) / "mcp.db", client=MockOllamaClient([extract_resp, admit_resp]))
            srv = MCPStdioServer(mem)

            init_resp = srv.handle_request(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}}
            )
            assert init_resp is not None
            self.assertEqual(init_resp["result"]["protocolVersion"], "2025-03-26")

            tools_resp = srv.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            assert tools_resp is not None
            names = [t["name"] for t in tools_resp["result"]["tools"]]
            self.assertEqual(names, ["memory_remember", "memory_recall", "memory_stats", "memory_delete"])

            add_resp = srv.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "memory_remember",
                        "arguments": {"content": "今天把服务端口从 8080 改成了 9090。", "namespace": "ops"},
                    },
                }
            )
            assert add_resp is not None
            self.assertFalse(add_resp["result"]["isError"])

            recall_resp = srv.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "memory_recall",
                        "arguments": {"query": "服务端口", "namespace": "ops"},
                    },
                }
            )
            assert recall_resp is not None
            body = json.loads(recall_resp["result"]["content"][0]["text"])
            self.assertEqual(body["count"], 1)
            self.assertIn("9090", body["items"][0]["text"])

            stats_resp = srv.handle_request(
                {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "memory_stats", "arguments": {"namespace": "ops"}}}
            )
            assert stats_resp is not None
            stats_body = json.loads(stats_resp["result"]["content"][0]["text"])
            self.assertEqual(stats_body["facts"], 1)

            fid = body["items"][0]["fact_id"]
            del_resp = srv.handle_request(
                {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "memory_delete", "arguments": {"fact_id": fid}}}
            )
            assert del_resp is not None
            del_body = json.loads(del_resp["result"]["content"][0]["text"])
            self.assertTrue(del_body["deleted"])
            self.assertEqual(mem.stats(ns="ops")["facts"], 0)

    def test_hud_status_states_and_delete_lifecycle(self) -> None:
        extract_resp = json.dumps(
            {
                "facts": [
                    {
                        "text": "API 网关限流将在 2026-03-15 提高到 120 次/分。",
                        "type": "world",
                        "occurred_start": "2026-03-15",
                        "occurred_end": None,
                        "where": None,
                        "entities": ["API 网关"],
                        "evidence": ["s0"],
                    },
                    {
                        "text": "缓存层首版采用单文件 SQLite WAL。",
                        "type": "experience",
                        "occurred_start": None,
                        "occurred_end": None,
                        "where": None,
                        "entities": ["SQLite"],
                        "evidence": ["s1"],
                    },
                ]
            },
            ensure_ascii=False,
        )
        admit_resp = json.dumps(
            {
                "0": {"decision": "pass", "evidence": ["s0"], "replacement": ""},
                "1": {
                    "decision": "revise",
                    "evidence": ["s1"],
                    "replacement": "缓存层首版采用单文件 SQLite WAL + FTS5，暂不上 Redis。",
                },
            },
            ensure_ascii=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "hud.db"
            client_res = MockOllamaClient(
                [extract_resp, admit_resp],
                model="qwen3.5:4b-memory-v2",
                running=["qwen3.5:4b-memory-v2"],
            )
            mem = Memory(db_path=db_path, client=client_res)
            added = mem.remember(
                "2026-03-01 记录：网关限流 2026-03-15 调为 120。\n\n缓存层首版用 SQLite WAL，暂不上 Redis。",
                ns="hud-test",
                verify=True,
            )
            self.assertEqual(added.saved, 2)
            self.assertEqual(added.revised, 1)

            # 1) Resident green state
            snap_res = collect_status(mem)
            self.assertEqual(snap_res.state, "resident")
            self.assertEqual(snap_res.badge, "常驻")
            self.assertEqual(snap_res.documents, 1)
            self.assertEqual(snap_res.facts, 2)
            self.assertEqual(snap_res.revised_facts, 1)
            self.assertGreater(snap_res.db_size_bytes, 0)
            self.assertTrue(format_bytes(snap_res.db_size_bytes).endswith(("B", "KB", "MB")))

            # 2) Ready standby cyan state
            mem_ready = Memory(
                db_path=db_path,
                client=MockOllamaClient([], model="qwen3.5:4b-memory-v2", running=[]),
            )
            snap_ready = collect_status(mem_ready)
            self.assertEqual(snap_ready.state, "ready")
            self.assertEqual(snap_ready.badge, "待机")

            # 3) Offline red state
            mem_off = Memory(db_path=db_path, client=MockOllamaClient([], offline=True))
            snap_off = collect_status(mem_off)
            self.assertEqual(snap_off.state, "offline")
            self.assertEqual(snap_off.badge, "离线")

            # 4) Document lookup & delete lifecycle
            doc = mem.get_document(added.doc_id)
            assert doc is not None
            self.assertEqual(len(doc["segments"]), 2)
            self.assertTrue(mem.delete_fact(added.fact_ids[1]))
            self.assertEqual(len(mem.recall("SQLite", ns="hud-test")), 0)
            self.assertTrue(mem.delete_document(added.doc_id))
            self.assertEqual(mem.stats()["documents"], 0)
            self.assertEqual(mem.stats()["facts"], 0)


    def test_parse_json_object_salvages_truncated_facts(self) -> None:
        truncated = (
            '{"facts": ['
            '{"text": "支付回调超时阈值调整为 15 秒。", "type": "world", "evidence": ["s0"]}, '
            '{"text": "边缘节点最大并发上限为 3200。", "type": "world", "evidence": ["s1"]}, '
            '{"text": "未闭合截断的第三条'
        )
        parsed = parse_json_object(truncated)
        assert parsed is not None
        self.assertEqual(len(parsed["facts"]), 2)
        self.assertIn("15 秒", parsed["facts"][0]["text"])
        self.assertIn("3200", parsed["facts"][1]["text"])
        self.assertIsNone(parse_json_object('{"0": {"decision": "pass"'))

    def test_zeromem_alias_and_dual_mode_config_export(self) -> None:
        import zeromem
        from v2mem.cli import _export_mode_configs

        self.assertIs(zeromem.Memory, Memory)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db = tmp / "zm.db"

            fast_files = _export_mode_configs(
                mode="fast",
                db_path=db,
                model="qwen3.5:4b-memory-v2",
                ollama_url="http://127.0.0.1:11434",
                out_dir=tmp / "fast",
            )
            self.assertIn("fast_sdk_example", fast_files)
            self.assertIn("from zeromem import Memory", fast_files["fast_sdk_example"].read_text(encoding="utf-8"))

            mcp_files = _export_mode_configs(
                mode="mcp",
                db_path=db,
                model="qwen3.5:4b-memory-v2",
                ollama_url="http://127.0.0.1:11434",
                out_dir=tmp / "mcp",
            )
            mcp_cfg = json.loads(mcp_files["mcp_config"].read_text(encoding="utf-8"))
            self.assertIn("zeromem", mcp_cfg["mcpServers"])
            self.assertIn("mcp", mcp_cfg["mcpServers"]["zeromem"]["args"])
            self.assertIn("并行发起 2 次", mcp_files["mcp_prompt"].read_text(encoding="utf-8"))
            self.assertIn("@deepseek-ai/dsh-mcp-client", mcp_files["dsh_mcp_patch"].read_text(encoding="utf-8"))

    def test_hardware_probe_and_agent_connectors_lifecycle(self) -> None:
        from v2mem.connectors import (
            connect_client,
            detect_clients,
            disconnect_client,
            export_project_rules,
        )
        from v2mem.hardware import detect_hardware

        hw = detect_hardware()
        self.assertIn(hw.recommend_quant, ("q8_0", "q4_k_m"))
        self.assertIn(hw.recommend_num_ctx, (4096, 8192))

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_home = Path(tmpdir)
            db = fake_home / ".zeromem" / "memory.db"

            # Seed existing Cursor JSON & Codex TOML with unrelated entries to verify non-destructive merge
            cursor_cfg = fake_home / ".cursor" / "mcp.json"
            cursor_cfg.parent.mkdir(parents=True, exist_ok=True)
            cursor_cfg.write_text(json.dumps({"mcpServers": {"other": {"command": "node"}}}), encoding="utf-8")

            codex_cfg = fake_home / ".codex" / "config.toml"
            codex_cfg.parent.mkdir(parents=True, exist_ok=True)
            codex_cfg.write_text('model = "demo"\n', encoding="utf-8")

            # Connect JSON (cursor), TOML (codex), YAML (dsh)
            r_cur = connect_client("cursor", db_path=db, home_override=fake_home)
            self.assertEqual(r_cur["status"], "connected")
            self.assertIsNotNone(r_cur["backup_path"])
            merged_cur = json.loads(cursor_cfg.read_text(encoding="utf-8"))
            self.assertIn("other", merged_cur["mcpServers"])
            self.assertIn("zeromem", merged_cur["mcpServers"])

            r_cdx = connect_client("codex", db_path=db, home_override=fake_home)
            self.assertEqual(r_cdx["status"], "connected")
            toml_text = codex_cfg.read_text(encoding="utf-8")
            self.assertIn('model = "demo"', toml_text)
            self.assertIn("[mcp_servers.zeromem]", toml_text)

            # Idempotent re-connect must not duplicate TOML block
            connect_client("codex", db_path=db, home_override=fake_home)
            self.assertEqual(codex_cfg.read_text(encoding="utf-8").count("[mcp_servers.zeromem]"), 1)

            r_dsh = connect_client("dsh", db_path=db, home_override=fake_home)
            self.assertEqual(r_dsh["status"], "connected")
            self.assertTrue(Path(r_dsh["config_path"]).is_file())

            # Detect sees connected state
            statuses = {s.id: s for s in detect_clients(home_override=fake_home)}
            self.assertTrue(statuses["cursor"].connected)
            self.assertTrue(statuses["codex"].connected)
            self.assertTrue(statuses["dsh"].connected)

            # Disconnect cleanly removes zeromem while keeping unrelated user entries
            disconnect_client("cursor", home_override=fake_home)
            after_cur = json.loads(cursor_cfg.read_text(encoding="utf-8"))
            self.assertIn("other", after_cur["mcpServers"])
            self.assertNotIn("zeromem", after_cur["mcpServers"])

            disconnect_client("codex", home_override=fake_home)
            after_toml = codex_cfg.read_text(encoding="utf-8")
            self.assertIn('model = "demo"', after_toml)
            self.assertNotIn("[mcp_servers.zeromem]", after_toml)

            # Export project rules (.cursor/rules/zeromem.mdc + AGENTS.md idempotent block)
            proj = fake_home / "sample_repo"
            proj.mkdir()
            rules = export_project_rules(proj)
            self.assertTrue(Path(rules["cursor_rule"]).is_file())
            self.assertIn("zeromem:begin", Path(rules["agents_md"]).read_text(encoding="utf-8"))
            export_project_rules(proj)
            self.assertEqual(Path(rules["agents_md"]).read_text(encoding="utf-8").count("zeromem:begin"), 1)

    def test_cli_connect_disconnect_flag_parsing(self) -> None:
        from v2mem.cli import build_parser, main as cli_main

        parser = build_parser()
        for argv in (
            ["connect"],
            ["connect", "--list", "--json"],
            ["connect", "--all", "--dry-run"],
            ["connect", "--project-rules", "."],
            ["connect", "cursor", "codex"],
            ["disconnect", "--all", "--dry-run"],
            ["disconnect", "cursor"],
        ):
            ns = parser.parse_args(argv)
            self.assertIn(ns.command, ("connect", "disconnect"))

        # Unknown client returns non-zero cleanly via ValueError handler
        self.assertEqual(cli_main(["connect", "nonexistent-client"]), 1)
        self.assertEqual(cli_main(["disconnect", "nonexistent-client"]), 1)


if __name__ == "__main__":
    unittest.main()
