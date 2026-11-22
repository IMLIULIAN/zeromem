"""Comprehensive cloud end-to-end verification suite for v2-memory on live V2."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from v2mem import Memory
from v2mem.contracts import FACT_ADMISSION_SYSTEM, Segment, parse_json_object
from v2mem.ollama import OllamaClient


def run_cmd(cmd: list[str], *, input_text: str | None = None, timeout: int = 240) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
    )


def main() -> int:
    v2mem_bin = os.environ.get("ZEROMEM_BIN") or os.environ.get("V2MEM_BIN") or "zmem"
    timeout_sec = float(os.environ.get("V2MEM_LIVE_TIMEOUT", "180"))
    num_ctx = int(os.environ.get("ZEROMEM_NUM_CTX") or os.environ.get("V2MEM_NUM_CTX") or "4096")

    report: list[dict] = []

    def record(cid: str, name: str, ok: bool, detail: str, elapsed: float) -> None:
        item = {"id": cid, "name": name, "ok": ok, "elapsed_sec": round(elapsed, 2), "detail": detail}
        report.append(item)
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {cid} {name} ({elapsed:.2f}s) -> {detail}", flush=True)

    client = OllamaClient(num_ctx=num_ctx, timeout=timeout_sec)
    resolved_model = client.resolve_model()
    print(f"=== Cloud E2E Verification using model: {resolved_model} (num_ctx={num_ctx}) ===", flush=True)

    with tempfile.TemporaryDirectory(prefix="v2mem_cloud_e2e_") as td:
        root = Path(td)
        db = root / "cloud.db"
        mem = Memory(db_path=db, model=resolved_model, num_ctx=num_ctx, timeout=timeout_sec)

        # 1. W-1: Plain text remember + record-date vs effective-date separation
        t0 = time.monotonic()
        r1 = mem.remember(
            "2026-03-01 记录：API 网关速率上限将于 2026-03-15 起调整为每分钟 120 次，原上限为每分钟 60 次。",
            ns="t1",
            mentioned_at="2026-03-01T08:00:00Z",
            verify=True,
        )
        no_leak = all("2026-03-01" not in str(f.get("occurred_start") or "") for f in r1.facts)
        ok1 = r1.saved >= 1 and r1.guard_rejected == 0 and no_leak
        record(
            "W-1",
            "纯文本写入与记录日隔离",
            ok1,
            f"extracted={r1.extracted}, saved={r1.saved}, revised={r1.revised}, no_date_leak={no_leak}",
            time.monotonic() - t0,
        )

        # 2. W-7: Idempotent deduplication
        t0 = time.monotonic()
        r_dup = mem.remember(
            "2026-03-01 记录：API 网关速率上限将于 2026-03-15 起调整为每分钟 120 次，原上限为每分钟 60 次。",
            ns="t1",
            mentioned_at="2026-03-01T08:00:00Z",
            verify=True,
        )
        record("W-7", "幂等去重", r_dup.deduplicated, f"deduplicated={r_dup.deduplicated}", time.monotonic() - t0)

        # 3. W-2 & R-5: CLI file ingest + stdin ingest + JSON output
        t0 = time.monotonic()
        note_file = root / "note.md"
        note_file.write_text("团队决定数据库主节点连接池上限设为 200，只读副本上限设为 500。", encoding="utf-8")
        p_file = run_cmd([v2mem_bin, "--db", str(db), "--num-ctx", str(num_ctx), "add", "-f", str(note_file), "--ns", "cli-ns", "--json"])
        j_file = json.loads(p_file.stdout) if p_file.returncode == 0 and p_file.stdout.strip() else {}
        p_stdin = run_cmd(
            [v2mem_bin, "--db", str(db), "--num-ctx", str(num_ctx), "add", "--ns", "cli-ns", "--no-verify", "--json"],
            input_text="运维约定每周日凌晨 03:00 执行 SQLite 冷备归档。",
        )
        j_stdin = json.loads(p_stdin.stdout) if p_stdin.returncode == 0 and p_stdin.stdout.strip() else {}
        ok2 = p_file.returncode == 0 and j_file.get("saved", 0) >= 1 and p_stdin.returncode == 0 and j_stdin.get("saved", 0) >= 1
        record(
            "W-2/W-6/R-5",
            "CLI 文件/管道写入、免验收与 JSON 输出",
            ok2,
            f"file_saved={j_file.get('saved')}, stdin_saved={j_stdin.get('saved')}",
            time.monotonic() - t0,
        )

        # 4. W-3: Multi-turn conversation with role preservation
        t0 = time.monotonic()
        r3 = mem.remember(
            [
                {"role": "user", "content": "缓存层先上 Redis 还是 SQLite？"},
                {"role": "assistant", "content": "首版并发低于 500 QPS，建议先用单文件 SQLite WAL 加 FTS5，暂不上 Redis。"},
            ],
            ns="chat",
            verify=True,
        )
        hits_chat = mem.recall("为什么不用 Redis", ns="chat", limit=3)
        has_role = any(any(ev.get("role") for ev in h.evidence) for h in hits_chat)
        record(
            "W-3",
            "多轮对话写入与角色溯源",
            r3.saved >= 1 and len(hits_chat) >= 1 and has_role,
            f"saved={r3.saved}, hits={len(hits_chat)}, has_role={has_role}",
            time.monotonic() - t0,
        )

        # 5. W-4: Long text (>2200 chars) multi-window chunking
        t0 = time.monotonic()
        filler_paras = [
            "日常文档维护与归档工作遵循统一的标准格式规范，各团队在提交技术材料时只需保持段落清晰、语义连贯即可，无需重复登记常规性背景介绍或通用流程说明，此类通用叙述仅用于补齐上下文篇幅与章节过渡。"
            for _ in range(24)
        ]
        filler_paras.insert(0, "2026-08-10 核心配置变更：支付回调超时阈值调整为 15 秒。")
        filler_paras.append("2026-08-12 压测结论：边缘节点最大并发连接数上限确认为 3200。")
        long_text = "\n\n".join(filler_paras)
        r4 = mem.remember(long_text, ns="long-doc", verify=False, max_window_chars=1200)
        h_head = mem.recall("支付回调超时", ns="long-doc", limit=3)
        h_tail = mem.recall("3200", ns="long-doc", limit=3)
        ok4 = len(long_text) > 2000 and r4.windows >= 2 and r4.saved >= 2 and len(h_head) >= 1 and len(h_tail) >= 1
        record(
            "W-4",
            "长文本多窗口微切块抽取与首尾召回",
            ok4,
            f"chars={len(long_text)}, windows={r4.windows}, saved={r4.saved}, head_hit={len(h_head)}, tail_hit={len(h_tail)}",
            time.monotonic() - t0,
        )

        # 6. W-5 Live V2 Admission gate check: deliberate wrong number -> revise, hallucination -> drop
        t0 = time.monotonic()
        adm_payload = {
            "evidence": [
                Segment(id="s0", path=["note", "0"], text="数据库主节点连接池上限设置为 200，只读副本上限为 500。").to_payload()
            ],
            "facts": {
                "0": {
                    "text": "数据库主节点连接池上限设置为 500。",
                    "type": "world",
                    "occurred_start": None,
                    "occurred_end": None,
                    "where": None,
                    "entities": ["数据库主节点"],
                    "mentioned_at": "2026-05-03T08:00:00Z",
                    "context": "cloud-e2e",
                    "tags": [],
                },
                "1": {
                    "text": "火星基地将于下周二启用量子路由器。",
                    "type": "world",
                    "occurred_start": None,
                    "occurred_end": None,
                    "where": None,
                    "entities": ["火星基地"],
                    "mentioned_at": "2026-05-03T08:00:00Z",
                    "context": "cloud-e2e",
                    "tags": [],
                },
            },
        }
        raw_adm, _ = client.chat_json(FACT_ADMISSION_SYSTEM, adm_payload, num_predict=512)
        verdicts = parse_json_object(raw_adm) or {}
        d0 = (verdicts.get("0") or {}).get("decision")
        rep0 = str((verdicts.get("0") or {}).get("replacement") or "")
        d1 = (verdicts.get("1") or {}).get("decision")
        ok_adm = (d0 == "revise" and "200" in rep0) and (d1 in {"drop", "hold"})
        record(
            "W-5",
            "V2 真机写前纠错 (revise) 与幻觉拦截 (drop)",
            ok_adm,
            f"slot0={d0} (rep={rep0[:60]!r}), slot1={d1}",
            time.monotonic() - t0,
        )

        # 7. W-8 & R-1 & R-3 & R-4: Namespace isolation, bilingual recall, empty search, delete sync
        t0 = time.monotonic()
        iso_hits = mem.recall("网关", ns="chat", limit=5)
        t1_hits = mem.recall("网关 120", ns="t1", limit=5)
        recent = mem.recall("", ns=None, limit=5)
        doc_detail = mem.get_document(r1.doc_id)
        del_ok = False
        if t1_hits:
            fid = t1_hits[0].fact_id
            del_ok = mem.delete_fact(fid) and all(h.fact_id != fid for h in mem.recall("网关 120", ns="t1", limit=5))
        ok_r = len(iso_hits) == 0 and len(t1_hits) >= 1 and len(recent) >= 2 and doc_detail is not None and del_ok
        record(
            "W-8/R-1..R-4",
            "命名空间隔离、零向量检索、原文溯源与删除同步",
            ok_r,
            f"iso_leak={len(iso_hits)}, t1_hits={len(t1_hits)}, recent={len(recent)}, del_ok={del_ok}",
            time.monotonic() - t0,
        )

        # 8. M-1 / M-2 / M-3: Real MCP stdio subprocess lifecycle (4 tools) + error handling
        t0 = time.monotonic()
        mcp_db = root / "mcp.db"
        msgs = [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}),
            json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "memory_remember",
                        "arguments": {"content": "云端 MCP 测试：构建服务统一迁移至 9443 端口。", "namespace": "mcp"},
                    },
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {"name": "memory_recall", "arguments": {"query": "9443 端口", "namespace": "mcp"}},
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": "tools/call",
                    "params": {"name": "memory_stats", "arguments": {"namespace": "mcp"}},
                },
                ensure_ascii=False,
            ),
            "{broken-json",
            json.dumps({"jsonrpc": "2.0", "id": 7, "method": "unknown/method"}),
        ]
        p_mcp = run_cmd(
            [v2mem_bin, "--db", str(mcp_db), "--num-ctx", str(num_ctx), "mcp"],
            input_text="\n".join(msgs) + "\n",
            timeout=240,
        )
        replies = [json.loads(line) for line in p_mcp.stdout.splitlines() if line.strip()]
        tool_names = [t["name"] for t in (replies[2].get("result", {}).get("tools") or [])] if len(replies) >= 3 else []
        r_add = json.loads(replies[3]["result"]["content"][0]["text"]) if len(replies) >= 4 else {}
        r_rec = json.loads(replies[4]["result"]["content"][0]["text"]) if len(replies) >= 5 else {}
        r_stats = json.loads(replies[5]["result"]["content"][0]["text"]) if len(replies) >= 6 else {}
        err_parse = replies[6].get("error", {}).get("code") if len(replies) >= 7 else None
        err_meth = replies[7].get("error", {}).get("code") if len(replies) >= 8 else None
        del_mcp_ok = False
        if r_rec.get("items"):
            target_fid = r_rec["items"][0]["fact_id"]
            del_msg = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 8,
                    "method": "tools/call",
                    "params": {"name": "memory_delete", "arguments": {"fact_id": target_fid}},
                }
            )
            p_del = run_cmd([v2mem_bin, "--db", str(mcp_db), "mcp"], input_text=del_msg + "\n", timeout=60)
            del_replies = [json.loads(line) for line in p_del.stdout.splitlines() if line.strip()]
            r_del = json.loads(del_replies[0]["result"]["content"][0]["text"]) if del_replies else {}
            del_mcp_ok = p_del.returncode == 0 and bool(r_del.get("deleted"))
        ok_mcp = (
            p_mcp.returncode == 0
            and len(replies) == 8
            and tool_names == ["memory_remember", "memory_recall", "memory_stats", "memory_delete"]
            and r_add.get("saved", 0) >= 1
            and r_rec.get("count", 0) >= 1
            and r_stats.get("facts", 0) >= 1
            and del_mcp_ok
            and err_parse == -32700
            and err_meth == -32601
        )
        record(
            "M-1..M-3",
            "MCP stdio 握手、4 工具端到端读写删与异常容错",
            ok_mcp,
            f"replies={len(replies)}, tools={len(tool_names)}, saved={r_add.get('saved')}, recalled={r_rec.get('count')}, stats_facts={r_stats.get('facts')}, deleted={del_mcp_ok}, parse_err={err_parse}, meth_err={err_meth}",
            time.monotonic() - t0,
        )

        # 9. T-4: Graceful error when Ollama is unreachable
        t0 = time.monotonic()
        p_bad = run_cmd([v2mem_bin, "--ollama-url", "http://127.0.0.1:19999", "setup"])
        ok_bad = p_bad.returncode != 0 and "Traceback" not in p_bad.stderr and "无法连接 Ollama" in p_bad.stderr
        record(
            "T-4",
            "不可达 Ollama 优雅报错（无裸 Traceback）",
            ok_bad,
            f"rc={p_bad.returncode}, stderr={p_bad.stderr.strip().splitlines()[0] if p_bad.stderr.strip() else ''}",
            time.monotonic() - t0,
        )

    passed = sum(1 for r in report if r["ok"])
    total = len(report)
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps({"model": resolved_model, "passed": passed, "total": total, "results": report}, ensure_ascii=False, indent=2), flush=True)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
