"""Synthetic end-to-end demo for v2-memory (uses an isolated temporary SQLite DB)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from v2mem import Memory


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")

    with tempfile.TemporaryDirectory(prefix="v2mem_demo_") as tmpdir:
        db_path = Path(tmpdir) / "demo.db"
        mem = Memory(db_path=db_path)

        samples = [
            (
                "2026-03-01 记录：API 网关速率上限将于 2026-03-15 起调整为每分钟 120 次，"
                "原上限为每分钟 60 次。本次变更仅适用于公开查询接口，不改变内部管理端口。",
                "2026-03-01T09:00:00Z",
            ),
            (
                [
                    {"role": "user", "content": "缓存层要不要先上 Redis 集群？"},
                    {
                        "role": "assistant",
                        "content": "首版并发不高，建议先用单文件 SQLite WAL 模式并开启 FTS5 索引，等 QPS 超过 500 再评估 Redis。",
                    },
                ],
                "2026-04-10T14:30:00Z",
            ),
            (
                "2026-05-02 压测结果：服务部署在 8090 端口，主节点连接池上限设置为 200，"
                "只读副本上限为 500，构建镜像摘要为 sha256:abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789。",
                "2026-05-02T18:00:00Z",
            ),
        ]

        print("=== 1. 写入 3 组纯合成样例（Extract -> Admit -> Guard -> SQLite）===")
        for idx, (content, ts) in enumerate(samples, 1):
            res = mem.remember(content, ns="demo", mentioned_at=ts, verify=True)
            print(
                f"[样例 {idx}] model={res.model} | extracted={res.extracted} | saved={res.saved} "
                f"(revised={res.revised}, dropped={res.dropped}, held={res.held})"
            )
            for f in res.facts:
                print(
                    f"   - ({f['type']}/{f['action']}) occurred={f['occurred_start']} | "
                    f"{f['text']} <ev:{','.join(f['evidence'])}>"
                )

        print("\n=== 2. 双轨检索演示（同时匹配事实与绑定原文）===")
        for q in ("网关限流", "为什么不用 Redis", "连接池上限"):
            hits = mem.recall(q, ns="demo", limit=3)
            print(f"\n查询: {q!r} -> 命中 {len(hits)} 条")
            for h in hits:
                print(f"   * 事实: {h.text}")
                print(f"     时间: occurred={h.occurred_start or '-'} | recorded={h.mentioned_at}")
                for ev in h.evidence:
                    print(f"     原文({ev['id']}): {ev['text']}")

        print("\n=== 3. 库统计 ===")
        print(mem.stats(ns="demo"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
