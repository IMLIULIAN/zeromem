"""ZeroMem (零忆) CLI: setup / connect / disconnect / doctor / config / add / search / stats / mcp / ui / export / backup / audit / evict / restore / retract / trace-summary."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from .connectors import (
    connect_client,
    detect_clients,
    disconnect_client,
    export_project_rules,
    get_client_specs,
)
from .hardware import detect_hardware
from .hud import launch_hud
from .mcp_server import MCPStdioServer
from .audit import audit_rules_vs_memory, discover_rule_files, load_zeromem_memories, render_audit_markdown, write_audit_report
from .governance import restore_evicted, retract_record, soft_evict
from .memory import Memory, default_db_path, default_home_dir
from .portability import backup_database, export_markdown
from .trace import summarize_traces
from .ollama import (
    DEFAULT_NUM_CTX,
    DEFAULT_OLLAMA_URL,
    OFFICIAL_WEIGHTS,
    OllamaClient,
    OllamaError,
    download_gguf,
    pull_ollama_model,
    register_gguf_in_ollama,
)

MCP_SYSTEM_PROMPT = """# ZeroMem (零忆) MCP 推荐检索约定

你已接入本地记忆库 MCP 工具 `memory_recall`、`memory_remember`、`memory_stats`、`memory_delete`（由本地 Qwen3.5-4B-Memory-V2 驱动，写入不消耗云端 API，入库自动脱敏）：

1. **先查后答**：涉及历史项目配置、参数、决议、完整哈希或进展状态时，先调用 `memory_recall` 核实，避免凭印象猜测。
2. **单主题 1 次短查**：单一主题问题提炼 2–4 个核心关键词调用 1 次 `memory_recall`（`limit=4` 或 `6`）即可。
3. **跨文档/多指标并行拆查**：当问题同时涉及多个独立模块或指标（如同时问“网关限流上限”与“数据库连接池上限”）时，在同一轮中**并行发起 2 次聚焦短关键词 `memory_recall`**，避免把整段长问句一次塞进单次 FTS5 查询。
4. **命中即答、诚实拒答**：拿到绑定原文证据后直接准确作答（保留精确数值、生效日期、64 位哈希及“仅提议未执行”等边界）；若检索无记录，最多换词补查 1 次即明确回答“未提及/无记录”，不空转、不编造。
"""

FAST_SDK_EXAMPLE = '''"""ZeroMem (零忆) 方案一：轻量省钱版（单次预注 Python SDK 示例）。

在 5.2 万字对照基准中相比全量上下文注入约节省 94.3% 输入 Token，并自动做本地凭据脱敏。
"""

from zeromem import Memory, format_recall_context


def build_prompt_with_memory(question: str, *, ns: str = "default", top_k: int = 6) -> str:
    mem = Memory()
    hits = mem.recall(question, ns=ns, limit=top_k)
    if not hits:
        return f"问题：{question}"
    ctx = format_recall_context(hits)
    return f"{ctx}\\n\\n---\\n请仅依据上述记忆准确回答：{question}"


if __name__ == "__main__":
    mem = Memory()
    mem.remember("2026-03-01 记录：API 网关公开接口限流将于 2026-03-15 调整为每分钟 120 次。", ns="demo")
    print(build_prompt_with_memory("API 网关限流上限和生效日期？", ns="demo"))
'''


def _ensure_utf8_io() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _mcp_snippet(db_path: Path, model: str | None, ollama_url: str) -> dict:
    args = ["-m", "zeromem", "--db", str(db_path)]
    if model:
        args.extend(["--model", model])
    if ollama_url != DEFAULT_OLLAMA_URL:
        args.extend(["--ollama-url", ollama_url])
    args.append("mcp")
    return {
        "mcpServers": {
            "zeromem": {
                "command": sys.executable,
                "args": args,
            }
        }
    }


def _dsh_mcp_yaml(db_path: Path, model: str | None, ollama_url: str) -> str:
    py_exe = Path(sys.executable).as_posix()
    db_str = Path(db_path).as_posix()
    extra_args = ""
    if model:
        extra_args += f"          - --model\n          - {model}\n"
    if ollama_url != DEFAULT_OLLAMA_URL:
        extra_args += f"          - --ollama-url\n          - {ollama_url}\n"
    return (
        "# ZeroMem (零忆) 方案二：DSH MCP 自主拆查补丁（配合 `dsh --patch` 使用）\n"
        "- insert:\n"
        "    - id: zeromem-mcp\n"
        "      name: '@deepseek-ai/dsh-mcp-client'\n"
        "      config:\n"
        "        serverName: zeromem\n"
        "        transport: stdio\n"
        f"        command: {py_exe}\n"
        "        args:\n"
        "          - -m\n"
        "          - zeromem\n"
        "          - --db\n"
        f"          - {db_str}\n"
        f"{extra_args}"
        "          - mcp\n"
        "        env: {}\n"
    )


def _export_mode_configs(
    *,
    mode: str,
    db_path: Path,
    model: str | None,
    ollama_url: str,
    out_dir: Path,
) -> dict[str, Path]:
    """Write ready-to-use configuration templates for the chosen installation mode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    if mode == "fast":
        sdk_file = out_dir / "fast_sdk_example.py"
        sdk_file.write_text(FAST_SDK_EXAMPLE, encoding="utf-8")
        written["fast_sdk_example"] = sdk_file
    else:
        mcp_json = out_dir / "mcp_config.json"
        mcp_json.write_text(
            json.dumps(_mcp_snippet(db_path, model, ollama_url), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        dsh_yml = out_dir / "dsh_mcp_patch.yml"
        dsh_yml.write_text(_dsh_mcp_yaml(db_path, model, ollama_url), encoding="utf-8")
        prompt_md = out_dir / "MCP_SYSTEM_PROMPT.md"
        prompt_md.write_text(MCP_SYSTEM_PROMPT, encoding="utf-8")
        written["mcp_config"] = mcp_json
        written["dsh_mcp_patch"] = dsh_yml
        written["mcp_prompt"] = prompt_md
    return written


def _cmd_setup(args: argparse.Namespace) -> int:
    hw = detect_hardware()
    quant = hw.recommend_quant if args.quant == "auto" else args.quant.lower()
    num_ctx = args.num_ctx if args.num_ctx is not None else hw.recommend_num_ctx

    mode_label = "方案一：轻量省钱版（CLI / SDK 单次预注）" if args.mode == "fast" else "方案二：Agent MCP 版（自主拆查）"
    print(f"=== ZeroMem (零忆) by ZeroZoneX · 初始化 [{mode_label}] ===")
    print(f"[硬件] {hw.summary_line()}")
    print(f"       依据：{hw.reason}")

    client = OllamaClient(
        base_url=args.ollama_url,
        model=args.model,
        num_ctx=num_ctx,
        timeout=hw.recommend_timeout,
    )

    try:
        existing = client.list_models()
    except OllamaError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        print("请先启动 Ollama（桌面版直接运行，或终端执行 `ollama serve`）。", file=sys.stderr)
        return 1

    chosen_model: str | None = args.model

    if args.gguf:
        tag = chosen_model or OFFICIAL_WEIGHTS[quant]["local_tag"]
        print(f"[1/3] 正在从本地 GGUF 注册 Ollama 模型 `{tag}` (ctx={num_ctx}) ...")
        chosen_model = register_gguf_in_ollama(Path(args.gguf), tag=tag, num_ctx=num_ctx)
    elif args.source == "ollama":
        print(f"[1/3] 正在从 Ollama Hub 拉取模型 ({quant.upper()}) ...")
        chosen_model = pull_ollama_model(quant=quant, use_hf=False)
    elif args.source == "hf":
        print(f"[1/3] 正在通过 Ollama 拉取 HuggingFace 官方权重 ({quant.upper()}) ...")
        chosen_model = pull_ollama_model(quant=quant, use_hf=True)
    elif args.source == "modelscope":
        print(f"[1/3] 正在通过魔搭 ModelScope 国内镜像下载权重 ({quant.upper()}) ...")
        gguf_path = download_gguf(quant=quant, source="modelscope")
        tag = chosen_model or OFFICIAL_WEIGHTS[quant]["local_tag"]
        print(f"      正在注册到 Ollama: `{tag}` (ctx={num_ctx}) ...")
        chosen_model = register_gguf_in_ollama(gguf_path, tag=tag, num_ctx=num_ctx)
    else:
        try:
            chosen_model = client.resolve_model()
            print(f"[1/3] 已检测到本机 V2 模型: `{chosen_model}`（复用，无需重复下载）")
        except OllamaError:
            if args.no_pull:
                print(f"[1/3] 本机已有模型 {len(existing)} 个，但未检测到 V2（已启用 --no-pull）。")
                print("      请运行 `zmem setup --source modelscope`（国内）或 `--source hf`（海外）获取权重。")
                return 2
            print(f"[1/3] 本机未检测到 V2 模型，正在按硬件推荐自动拉取 {quant.upper()}（优先魔搭镜像，失败自动切 HuggingFace）...")
            try:
                gguf_path = download_gguf(quant=quant, source="modelscope")
                tag = OFFICIAL_WEIGHTS[quant]["local_tag"]
                chosen_model = register_gguf_in_ollama(gguf_path, tag=tag, num_ctx=num_ctx)
            except Exception as exc_ms:
                print(f"      魔搭镜像暂不可用 ({exc_ms})，自动回退 HuggingFace 直拉 ...", file=sys.stderr)
                chosen_model = pull_ollama_model(quant=quant, use_hf=True)

    if not args.skip_check:
        print(f"[2/3] 正在用 `{chosen_model}` 执行全链路合成自检（抽取 + 验收 + 检索）...")
        with tempfile.TemporaryDirectory(prefix="zeromem_smoke_") as tmpdir:
            smoke_mem = Memory(
                db_path=Path(tmpdir) / "smoke.db",
                model=chosen_model,
                ollama_url=args.ollama_url,
                num_ctx=num_ctx,
                timeout=hw.recommend_timeout,
            )
            sample = (
                "2026-03-01 记录：团队决定将网关限流上限在 2026-03-15 调整为每分钟 120 次，"
                "替代原先每分钟 60 次的配置。"
            )
            res = smoke_mem.remember(
                sample,
                ns="smoke-check",
                mentioned_at="2026-03-01T08:00:00Z",
                verify=True,
            )
            hits = smoke_mem.recall("网关限流 120", ns="smoke-check", limit=3)
            if res.saved < 1 or not hits:
                print(f"[警告] 自检未命中预期结果: remember={res.to_dict()}", file=sys.stderr)
                return 3
            print(
                f"      自检通过！saved={res.saved}, revised={res.revised}, "
                f"recalled={len(hits)}, occurred_start={hits[0].occurred_start!r}"
            )
    else:
        print("[2/3] 已按参数跳过冒烟自检。")

    db_path = Path(args.db).expanduser() if args.db else default_db_path()
    cfg_dir = default_home_dir() / "configs"
    written = _export_mode_configs(
        mode=args.mode,
        db_path=db_path,
        model=chosen_model,
        ollama_url=args.ollama_url,
        out_dir=cfg_dir,
    )

    print(f"[3/3] 就绪！默认数据库路径: {db_path}")
    if args.mode == "fast":
        print(f"\n[方案一 · 轻量省钱版] 已生成 Python SDK 单次预注示例: {written['fast_sdk_example']}")
        print("  快速体验：")
        print('    zmem add "2026-03-01 记录：网关限流 2026-03-15 调为 120 次/分。" --ns demo')
        print('    zmem search "网关限流" --ns demo')
        print("  如需随时切换为方案二（Agent MCP 版），执行：zmem connect --all")
    else:
        print(f"\n[方案二 · Agent MCP 版] 已落盘配置模板至: {cfg_dir}")
        print(f"  - 通用 MCP 配置 (Claude/Cursor/Antigravity): {written['mcp_config']}")
        print(f"  - DSH MCP 补丁:                             {written['dsh_mcp_patch']}")
        print(f"  - 推荐并行拆查检索提示词:                    {written['mcp_prompt']}")

        installed_agents = [c for c in detect_clients() if c.installed]
        if args.connect_all and installed_agents:
            print("\n[自动接入] 正在为本机已检测到的主流 Agent 一键写入 ZeroMem MCP（均先备份 .bak）：")
            for c in installed_agents:
                r = connect_client(c.id, db_path=db_path, model=chosen_model, ollama_url=args.ollama_url)
                print(f"  ✓ {c.display_name:<24} -> {r['config_path']}")
        elif installed_agents:
            names = ", ".join(c.display_name for c in installed_agents)
            print(f"\n[检测到主流 Agent] {names}")
            print("  运行 `zmem connect --all` 即可自动备份并一键接入上述全部客户端！")
    return 0


def _cmd_connect(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser() if args.db else default_db_path()

    if args.project_rules:
        res = export_project_rules(Path(args.project_rules), dry_run=args.dry_run)
        if args.json:
            print(json.dumps(res, ensure_ascii=False, indent=2))
        else:
            tag = "[预览]" if args.dry_run else "[已写入]"
            print(f"{tag} Cursor 规则: {res['cursor_rule']}")
            print(f"{tag} AGENTS.md 块: {res['agents_md']}")
        if not args.clients and not args.all:
            return 0

    statuses = detect_clients()
    if args.list or (not args.clients and not args.all):
        if args.json:
            print(json.dumps([s.to_dict() for s in statuses], ensure_ascii=False, indent=2))
            return 0
        print("=== ZeroMem 主流 AI Agent 探测与接入状态 ===")
        for s in statuses:
            st_badge = "已接入 ✓" if s.connected else ("已安装（未接入）" if s.installed else "未检测到")
            print(f"  - {s.id:<15} | {s.display_name:<22} | {st_badge:<14} | {s.config_path}")
        print("\n一键接入全部已安装客户端：zmem connect --all")
        print("指定单个客户端接入：     zmem connect cursor codex claude-desktop")
        print("同时导出项目规则文件：   zmem connect --project-rules .")
        return 0

    targets: list[str] = []
    if args.all:
        targets = [s.id for s in statuses if s.installed]
        if not targets:
            print("[zmem] 未检测到已安装的主流 Agent，可指定名称强制生成配置，例如：`zmem connect cursor`")
            return 0
    else:
        targets = list(args.clients)

    results = []
    for cid in targets:
        r = connect_client(
            cid,
            db_path=db_path,
            model=args.model,
            ollama_url=args.ollama_url,
            dry_run=args.dry_run,
        )
        results.append(r)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    for r in results:
        tag = "[预览]" if r["dry_run"] else "[已接入]"
        bak = f" (原配置已备份: {r['backup_path']})" if r.get("backup_path") else ""
        print(f"{tag} {r['display_name']} -> {r['config_path']}{bak}")
        if r.get("note"):
            print(f"         提示: {r['note']}")
    return 0


def _cmd_disconnect(args: argparse.Namespace) -> int:
    statuses = detect_clients()
    targets = [s.id for s in statuses if s.connected] if args.all else list(args.clients)
    if not targets:
        print("[zmem] 未发现需要断开的客户端（可传 `--all` 或指定客户端 ID）。")
        return 0

    results = [disconnect_client(cid, dry_run=args.dry_run) for cid in targets]
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    for r in results:
        tag = "[预览]" if r.get("dry_run") else f"[{r['status']}]"
        print(f"{tag} {r.get('display_name', r['client'])} -> {r['config_path']}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    hw = detect_hardware()
    db_path = Path(args.db).expanduser() if args.db else default_db_path()
    mem = Memory(db_path=db_path, model=args.model, ollama_url=args.ollama_url)
    stats = mem.stats()
    ollama_state = "offline"
    resolved_model = ""
    running: list[str] = []
    try:
        running = mem.client.list_running_models()
        resolved_model = mem.client.resolve_model()
        ollama_state = "resident" if resolved_model in running else "ready"
    except OllamaError as exc:
        resolved_model = str(exc).splitlines()[0]

    clients = detect_clients()
    report = {
        "hardware": hw.to_dict(),
        "ollama_state": ollama_state,
        "model": resolved_model,
        "running_models": running,
        "db_path": str(db_path),
        "stats": stats,
        "clients": [c.to_dict() for c in clients],
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print("=== ZeroMem (零忆) 本机环境体检 ===")
    print(f"  硬件与推荐档位 : {hw.summary_line()}")
    print(f"  Ollama 状态    : {ollama_state} ({resolved_model})")
    print(f"  本地数据库     : {db_path} ({stats['documents']} 文档 / {stats['facts']} 事实 / {stats['revised_facts']} 纠错)")
    installed = [c for c in clients if c.installed]
    connected = [c for c in clients if c.connected]
    print(f"  主流 Agent     : 检测到 {len(installed)} 个，已接入 {len(connected)} 个")
    for c in installed:
        mark = "✓ 已接入" if c.connected else "○ 未接入"
        print(f"    - {c.display_name:<22} {mark}")
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser() if args.db else default_db_path()
    out_dir = Path(args.write_dir).expanduser() if args.write_dir else (default_home_dir() / "configs")
    written = _export_mode_configs(
        mode=args.mode,
        db_path=db_path,
        model=args.model,
        ollama_url=args.ollama_url,
        out_dir=out_dir,
    )
    if args.json:
        payload: dict[str, object] = {
            "mode": args.mode,
            "db_path": str(db_path),
            "written": {k: str(v) for k, v in written.items()},
        }
        if args.mode == "mcp":
            payload["mcp_config"] = _mcp_snippet(db_path, args.model, args.ollama_url)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.mode == "fast":
        print(f"[zmem] 方案一（轻量省钱版 · CLI / SDK 单次预注）示例已写入: {written['fast_sdk_example']}\n")
        print(FAST_SDK_EXAMPLE)
    else:
        print(f"[zmem] 方案二（Agent MCP 版 · 自主拆查）配置已写入: {out_dir}")
        for label, path in written.items():
            print(f"  - {label}: {path}")
        print("\n--- MCP JSON（Claude Desktop / Cursor / Antigravity / Windsurf）---")
        print(json.dumps(_mcp_snippet(db_path, args.model, args.ollama_url), ensure_ascii=False, indent=2))
        print("\n--- 推荐粘贴到客户端规则的检索提示词 ---")
        print(MCP_SYSTEM_PROMPT.strip())
    return 0


def _cmd_add(args: argparse.Namespace) -> int:
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
        source_id = args.source_id or str(Path(args.file).resolve())
    elif args.text:
        text = " ".join(args.text)
        source_id = args.source_id
    else:
        text = sys.stdin.read()
        source_id = args.source_id or "stdin"

    mem = Memory(
        db_path=args.db,
        model=args.model,
        ollama_url=args.ollama_url,
        num_ctx=args.num_ctx,
    )
    res = mem.remember(
        text,
        ns=args.ns,
        source_id=source_id,
        mentioned_at=args.mentioned_at,
        verify=not args.no_verify,
    )
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(f"[zmem] doc={res.doc_id} ns={res.namespace}")
    if res.deduplicated:
        print("  已存在相同内容，自动幂等跳过。")
        return 0
    print(
        f"  模型={res.model} | 窗口={res.windows} | 抽取={res.extracted} | "
        f"入库={res.saved} (修订={res.revised}, 替代旧事实={res.superseded}, 丢弃={res.dropped}, 隔离={res.held}, 守卫拦截={res.guard_rejected})"
    )
    for f in res.facts:
        occ = f" [发生:{f['occurred_start']}]" if f.get("occurred_start") else ""
        sup = f" [替代:#{f['supersedes_fact_id']}]" if f.get("supersedes_fact_id") is not None else ""
        print(f"  - ({f['type']}/{f['action']}){occ}{sup} {f['text']}  <ev:{','.join(f['evidence'])}>")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    mem = Memory(
        db_path=args.db,
        model=args.model,
        ollama_url=args.ollama_url,
        num_ctx=args.num_ctx,
    )
    query = " ".join(args.query)
    hits = mem.recall(
        query,
        ns=args.ns,
        limit=args.limit,
        adaptive=not args.no_adaptive,
        as_of=args.as_of,
        occurred_from=args.occurred_from,
        occurred_to=args.occurred_to,
        include_superseded=not args.hide_superseded,
        recall_log_dir=args.recall_log_dir,
        trace_channel="zeromem_cli",
    )
    if args.json:
        print(json.dumps([h.to_dict() for h in hits], ensure_ascii=False, indent=2))
        return 0

    print(f"[zmem] 检索 {query!r} -> 命中 {len(hits)} 条")
    for idx, h in enumerate(hits, 1):
        time_info = f"occurred={h.occurred_start or '-'} | recorded={h.mentioned_at}"
        badges: list[str] = []
        if h.supersedes_fact_id is not None:
            badges.append(f"替代 #{h.supersedes_fact_id}")
        if h.superseded_by:
            badges.append("已被 " + ",".join(f"#{x}" for x in h.superseded_by) + " 替代")
        if h.until_expiry:
            badges.append(str(h.until_expiry["label"]))
        badge_str = (" [" + " | ".join(badges) + "]") if badges else ""
        print(f"\n{idx}. (#{h.fact_id} · {h.namespace}) ({h.type}/{h.action}, score={h.score}){badge_str} {time_info}")
        print(f"   事实: {h.text}")
        for ev in h.evidence:
            role_tag = f"[{ev['role']}] " if ev.get("role") else ""
            print(f"   原文({ev['id']}): {role_tag}{ev['text']}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser() if args.db else default_db_path()
    res = export_markdown(
        db_path=db_path,
        output_path=args.output,
        output_dir=args.dir,
        namespace=args.ns,
        occurred_from=args.occurred_from,
        occurred_to=args.occurred_to,
        as_of=args.as_of,
        include_superseded=not args.hide_superseded,
        include_evicted=args.include_evicted,
        frontmatter=not args.no_frontmatter,
    )
    if args.json:
        out_obj = {k: v for k, v in res.items() if k != "markdown"}
        print(json.dumps(out_obj, ensure_ascii=False, indent=2))
        return 0
    if res["mode"] == "stdout":
        print(res["markdown"], end="")
        return 0
    if res["mode"] == "directory":
        print(
            f"[zmem] 只读导出 Markdown 目录完成: {res['output_dir']} "
            f"({len(res['namespaces'])} 个空间 / {res['facts_exported']} 条事实 / {len(res['files_written'])} 个文件)"
        )
    else:
        print(
            f"[zmem] 只读导出 Markdown 完成: {res['output_path']} "
            f"({len(res['namespaces'])} 个空间 / {res['facts_exported']} 条事实)"
        )
    return 0


def _cmd_backup(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser() if args.db else default_db_path()
    target = args.output or args.target
    res = backup_database(
        db_path=db_path,
        output_path=target,
        overwrite=args.overwrite,
        verify=not args.no_verify,
        write_sha256=args.sha256,
    )
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    print(f"[zmem] SQLite 在线备份完成（无 WAL 伴生文件，可安全网盘同步）")
    print(f"  源文件   : {res['source_db']}")
    print(f"  快照路径 : {res['backup_path']}")
    print(
        f"  校验结果 : integrity={res['integrity_check']} | journal_mode={res['journal_mode']} | "
        f"大小={res['size_bytes']} B | 耗时={res['elapsed_ms']} ms"
    )
    print(
        f"  记录统计 : 文档={res['counts']['documents']} / 原文段={res['counts']['segments']} / "
        f"事实={res['counts']['facts']} / 替代链={res['counts']['supersessions']}"
    )
    print(f"  SHA-256  : {res['sha256']}")
    if res.get("sha256_file"):
        print(f"  校验文件 : {res['sha256_file']}")
    return 0


def _cmd_trace_summary(args: argparse.Namespace) -> int:
    res = summarize_traces(args.dir, days=args.days)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def _cmd_retract(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser() if args.db else default_db_path()
    res = retract_record(
        db_path=db_path,
        fact_id=args.fact_id,
        doc_id=args.doc_id,
        reason=args.reason,
        archive_root=args.archive_root,
        ledger_dir=args.ledger_dir,
        apply=args.apply,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def _cmd_evict(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser() if args.db else default_db_path()
    res = soft_evict(
        db_path=db_path,
        fact_ids=args.fact_id,
        doc_ids=args.doc_id,
        expired_only=args.expired,
        namespace=args.ns,
        as_of=args.as_of,
        reason=args.reason,
        batch_id=args.batch_id,
        dump_manifest=args.dump,
        ledger_dir=args.ledger_dir,
        force=args.force,
        apply=args.apply,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def _cmd_restore(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser() if args.db else default_db_path()
    res = restore_evicted(
        db_path=db_path,
        batch_id=args.batch,
        verify_manifest=args.verify,
        ledger_dir=args.ledger_dir,
        apply=args.apply,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser() if args.db else default_db_path()
    rule_files: list[Path] = []
    if args.rules:
        rule_files.extend(Path(r).expanduser().resolve() for r in args.rules)
    if args.project or args.include_globals or not rule_files:
        proj = args.project if args.project else (None if rule_files else ".")
        discovered = discover_rule_files(project_dir=proj, include_home_globals=args.include_globals)
        seen = {str(p).casefold() for p in rule_files}
        for p in discovered:
            if str(p).casefold() not in seen:
                rule_files.append(p)
                seen.add(str(p).casefold())
    mems, exp = load_zeromem_memories(db_path, namespace=args.ns, as_of=args.as_of)
    report = audit_rules_vs_memory(rule_files=rule_files, memory_records=mems, expiry_summary=exp)
    if args.out:
        report["written_files"] = write_audit_report(report, args.out)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_audit_markdown(report))
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    mem = Memory(db_path=args.db, ollama_url=args.ollama_url)
    print(json.dumps(mem.stats(ns=args.ns), ensure_ascii=False, indent=2))
    return 0


def _cmd_mcp(args: argparse.Namespace) -> int:
    mem = Memory(
        db_path=args.db,
        model=args.model,
        ollama_url=args.ollama_url,
        num_ctx=args.num_ctx,
    )
    return MCPStdioServer(mem).serve_forever()


def _cmd_ui(args: argparse.Namespace) -> int:
    mem = Memory(
        db_path=args.db,
        model=args.model,
        ollama_url=args.ollama_url,
        num_ctx=args.num_ctx,
    )
    return launch_hud(
        mem,
        refresh_sec=args.refresh,
        start_expanded=args.expanded,
        start_browser=args.browser,
        topmost=not args.no_topmost,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zmem",
        description="ZeroMem (零忆) by ZeroZoneX — zero-dependency local memory engine for Qwen3.5-4B-Memory-V2.",
    )
    parser.add_argument("--db", default=None, help="SQLite 数据库路径（默认 ~/.zeromem/memory.db）")
    parser.add_argument("--model", default=None, help="Ollama 模型名（省略则自动探测本机 V2）")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama 地址（默认 http://127.0.0.1:11434）")
    parser.add_argument("--num-ctx", type=int, default=None, help=f"推理上下文长度（默认按硬件自动推荐，基准 {DEFAULT_NUM_CTX}）")

    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="一键硬件选档、检查/拉取 V2 模型、冒烟自检并生成双方案配置")
    p_setup.add_argument(
        "--mode",
        choices=["fast", "mcp"],
        default="mcp",
        help="安装方案：fast（方案一·轻量省钱 CLI/SDK）或 mcp（方案二·Agent MCP，默认）",
    )
    p_setup.add_argument(
        "--source",
        choices=["auto", "modelscope", "hf", "ollama"],
        default="auto",
        help="权重来源：auto（已有复用，缺失自动按魔搭→HF 拉取）、modelscope、hf、ollama",
    )
    p_setup.add_argument("--quant", choices=["auto", "q8_0", "q4_k_m"], default="auto", help="量化档位（默认 auto 按显存自动选）")
    p_setup.add_argument("--gguf", default=None, help="直接用本地已下载的 .gguf 注册到 Ollama")
    p_setup.add_argument("--no-pull", action="store_true", help="仅探测本机已有模型，缺失时不自动下载")
    p_setup.add_argument("--skip-check", action="store_true", help="跳过冒烟自检")
    p_setup.add_argument("--connect-all", action="store_true", help="方案二初始化后自动为全部已检测主流 Agent 写入 MCP 配置")
    p_setup.set_defaults(func=_cmd_setup)

    valid_clients = [s.id for s in get_client_specs()]
    p_conn = sub.add_parser("connect", help="自动探测并一键接入主流 AI Agent（Claude / Cursor / Codex / Antigravity / DSH 等）")
    p_conn.add_argument("clients", nargs="*", metavar="CLIENT", help=f"目标客户端：{', '.join(valid_clients)}")
    p_conn.add_argument("--list", action="store_true", help="列出全部 7 大主流客户端的探测与接入状态")
    p_conn.add_argument("--all", action="store_true", help="一键接入本机检测到的全部已安装客户端（自动备份 .bak）")
    p_conn.add_argument("--project-rules", default=None, help="在指定项目目录写入 .cursor/rules/zeromem.mdc 与 AGENTS.md 检索规则")
    p_conn.add_argument("--dry-run", action="store_true", help="仅预览变更路径，不修改任何文件")
    p_conn.add_argument("--json", action="store_true", help="输出 JSON")
    p_conn.set_defaults(func=_cmd_connect)

    p_dis = sub.add_parser("disconnect", help="一键从主流 AI Agent 配置中安全移除 ZeroMem（先备份 .bak）")
    p_dis.add_argument("clients", nargs="*", metavar="CLIENT", help=f"目标客户端：{', '.join(valid_clients)}")
    p_dis.add_argument("--all", action="store_true", help="断开全部已接入客户端")
    p_dis.add_argument("--dry-run", action="store_true", help="仅预览不写入")
    p_dis.add_argument("--json", action="store_true", help="输出 JSON")
    p_dis.set_defaults(func=_cmd_disconnect)

    p_doc = sub.add_parser("doctor", help="一键体检硬件显存、Ollama 状态、数据库与主流 Agent 接入情况")
    p_doc.add_argument("--json", action="store_true", help="输出 JSON")
    p_doc.set_defaults(func=_cmd_doctor)

    p_config = sub.add_parser("config", help="随时导出/查看两套接入方案的配置模板与推荐提示词")
    p_config.add_argument("--mode", choices=["fast", "mcp"], default="mcp", help="方案选择：fast 或 mcp（默认）")
    p_config.add_argument("--write-dir", default=None, help="模板输出目录（默认 ~/.zeromem/configs）")
    p_config.add_argument("--json", action="store_true", help="以 JSON 格式输出路径与 MCP 配置")
    p_config.set_defaults(func=_cmd_config)

    p_add = sub.add_parser("add", help="写入文本或文件到记忆库")
    p_add.add_argument("text", nargs="*", help="要记忆的文本（留空则读取 --file 或标准输入）")
    p_add.add_argument("-f", "--file", default=None, help="从 UTF-8 文本/Markdown 文件读取")
    p_add.add_argument("--ns", default="default", help="命名空间（如项目名）")
    p_add.add_argument("--source-id", default=None, help="来源标识")
    p_add.add_argument("--mentioned-at", default=None, help="ISO-8601 记录时间")
    p_add.add_argument("--no-verify", action="store_true", help="跳过二阶段写前验收（低显存/纯 CPU 可选）")
    p_add.add_argument("--json", action="store_true", help="输出原始 JSON")
    p_add.set_defaults(func=_cmd_add)

    p_search = sub.add_parser("search", help="检索记忆并回溯原文证据")
    p_search.add_argument("query", nargs="*", help="查询词（留空返回最新或时间区间内记忆）")
    p_search.add_argument("--ns", default=None, help="按命名空间过滤")
    p_search.add_argument("-k", "--limit", type=int, default=None, help="返回条数上限（省略则启用 2–8 动态分数门槛）")
    p_search.add_argument("--no-adaptive", action="store_true", help="关闭动态相对分数门槛，恢复固定条数")
    p_search.add_argument("--as-of", default=None, help="指定 YYYY-MM-DD 基准日核验 until: 时效")
    p_search.add_argument("--occurred-from", default=None, help="事件发生时间下限（YYYY / YYYY-MM / YYYY-MM-DD，含当日）")
    p_search.add_argument("--occurred-to", default=None, help="事件发生时间上限（YYYY / YYYY-MM / YYYY-MM-DD，含当日）")
    p_search.add_argument("--hide-superseded", action="store_true", help="隐藏已被更新事实替代的历史记忆")
    p_search.add_argument("--recall-log-dir", default=None, help="可选：本次召回旁路元数据日志目录")
    p_search.add_argument("--json", action="store_true", help="输出原始 JSON")
    p_search.set_defaults(func=_cmd_search)

    p_exp = sub.add_parser("export", help="100% 只读导出带原文引用的 Markdown（适配 Obsidian / Git 检视）")
    p_exp.add_argument("--md", action="store_true", help="导出 Markdown 格式（默认即为 Markdown）")
    p_exp.add_argument("-o", "--output", default=None, help="输出为单个 .md 文件（省略且未指定 --dir 时输出到标准输出）")
    p_exp.add_argument("--dir", default=None, help="按命名空间拆分为 Obsidian/Git 目录（生成 INDEX.md 与各空间 .md）")
    p_exp.add_argument("--ns", default=None, help="仅导出指定命名空间")
    p_exp.add_argument("--occurred-from", default=None, help="按事件时间下限过滤（YYYY / YYYY-MM / YYYY-MM-DD）")
    p_exp.add_argument("--occurred-to", default=None, help="按事件时间上限过滤（YYYY / YYYY-MM / YYYY-MM-DD）")
    p_exp.add_argument("--as-of", default=None, help="指定 YYYY-MM-DD 基准日评估 until: 过期标注")
    p_exp.add_argument("--hide-superseded", action="store_true", help="不导出已被更新事实替代的历史记录")
    p_exp.add_argument("--include-evicted", action="store_true", help="附带回收站（evicted_records）摘要表")
    p_exp.add_argument("--no-frontmatter", action="store_true", help="省略 YAML frontmatter")
    p_exp.add_argument("--json", action="store_true", help="输出导出回执 JSON")
    p_exp.set_defaults(func=_cmd_export)

    p_bak = sub.add_parser("backup", help="走 SQLite Online Backup API 导出无 WAL 干净快照（安全网盘同步）")
    p_bak.add_argument("target", nargs="?", default=None, help="快照文件路径或目标目录（省略则写入 <db同级>/backups/）")
    p_bak.add_argument("-o", "--output", default=None, help="同 target：指定快照文件路径或目录")
    p_bak.add_argument("--overwrite", action="store_true", help="若目标文件已存在则原子覆盖")
    p_bak.add_argument("--no-verify", action="store_true", help="跳过快照 PRAGMA integrity_check 完整性校验")
    p_bak.add_argument("--sha256", action="store_true", help="额外生成同级 .sha256 校验文件")
    p_bak.add_argument("--json", action="store_true", help="输出备份回执 JSON")
    p_bak.set_defaults(func=_cmd_backup)

    p_stats = sub.add_parser("stats", help="查看记忆库统计")
    p_stats.add_argument("--ns", default=None, help="按命名空间过滤")
    p_stats.set_defaults(func=_cmd_stats)

    p_mcp = sub.add_parser("mcp", help="启动零依赖 MCP stdio 服务（4 工具）")
    p_mcp.set_defaults(func=_cmd_mcp)

    p_ui = sub.add_parser("ui", aliases=["hud"], help="启动桌面悬浮状态胶囊与 ZeroMem 记忆工作台（无网页）")
    p_ui.add_argument("--refresh", type=float, default=5.0, help="后台状态刷新间隔秒数（默认 5）")
    p_ui.add_argument("--expanded", action="store_true", help="启动时默认展开紧凑仪表卡")
    p_ui.add_argument("--browser", action="store_true", help="启动时直接打开 ZeroMem 记忆工作台窗口")
    p_ui.add_argument("--no-topmost", action="store_true", help="悬浮胶囊不默认置顶")
    p_ui.set_defaults(func=_cmd_ui)

    p_ts = sub.add_parser("trace-summary", help="只读汇总召回旁路日志（命中率、延迟、零命中与过期命中）")
    p_ts.add_argument("--dir", required=True, help="包含 *-recall-trace*.jsonl 的日志目录")
    p_ts.add_argument("--days", type=int, default=30, help="回看天数（默认 30）")
    p_ts.add_argument("--json", action="store_true", help="输出 JSON（默认即为 JSON）")
    p_ts.set_defaults(func=_cmd_trace_summary)

    p_ret = sub.add_parser("retract", help="删前归档 + 写入 retractions.jsonl 台账并校验后再删除（默认 dry-run）")
    p_ret.add_argument("--fact-id", type=int, default=None, help="要撤回的单条 fact_id")
    p_ret.add_argument("--doc-id", default=None, help="要撤回的完整 doc_id")
    p_ret.add_argument("--reason", required=True, help="撤回原因（必填）")
    p_ret.add_argument("--archive-root", default=None, help="外部归档目录（或设 ZEROMEM_REMOVED_ROOT）")
    p_ret.add_argument("--ledger-dir", default=None, help="台账目录（默认 <db同级>/governance）")
    p_ret.add_argument("--apply", action="store_true", help="确认落盘归档并执行删除（默认 dry-run）")
    p_ret.add_argument("--json", action="store_true", help="输出 JSON")
    p_ret.set_defaults(func=_cmd_retract)

    p_ev = sub.add_parser("evict", help="无损软淘汰至同库 evicted_records 边表（默认 dry-run）")
    p_ev.add_argument("--fact-id", type=int, action="append", default=None, help="按 fact_id 选择（可重复）")
    p_ev.add_argument("--doc-id", action="append", default=None, help="按 doc_id 选择（可重复）")
    p_ev.add_argument("--expired", action="store_true", help="选择全部带已过期 until: 标签的活跃事实")
    p_ev.add_argument("--ns", default=None, help="限定命名空间")
    p_ev.add_argument("--as-of", default=None, help="指定 YYYY-MM-DD 基准日")
    p_ev.add_argument("--reason", default="soft-evict", help="淘汰原因")
    p_ev.add_argument("--batch-id", default=None, help="自定义批次 ID")
    p_ev.add_argument("--dump", default=None, help="输出逐行 SHA-256 指纹清单 JSON")
    p_ev.add_argument("--ledger-dir", default=None, help="可选：同步追加 retractions.jsonl")
    p_ev.add_argument("--force", action="store_true", help="允许清空范围内的全部活跃记录")
    p_ev.add_argument("--apply", action="store_true", help="确认执行软淘汰（默认 dry-run）")
    p_ev.add_argument("--json", action="store_true", help="输出 JSON")
    p_ev.set_defaults(func=_cmd_evict)

    p_res = sub.add_parser("restore", help="校验 SHA-256 指纹后原子恢复软淘汰批次（默认 dry-run）")
    p_res.add_argument("--batch", required=True, help="待恢复的 batch_id")
    p_res.add_argument("--verify", default=None, help="可选：配合 evict --dump 生成的指纹清单核验")
    p_res.add_argument("--ledger-dir", default=None, help="可选：同步追加 retractions.jsonl")
    p_res.add_argument("--apply", action="store_true", help="确认执行原子恢复（默认 dry-run）")
    p_res.add_argument("--json", action="store_true", help="输出 JSON")
    p_res.set_defaults(func=_cmd_restore)

    p_aud = sub.add_parser("audit", help="100% 只读无 LLM 体检 AGENTS.md / 规则文件与 ZeroMem 记忆重叠及冲突")
    p_aud.add_argument("--rules", action="append", default=None, help="显式指定规则文件路径（可重复）")
    p_aud.add_argument("--project", default=None, help="自动发现指定项目目录下的 AGENTS.md / CLAUDE.md / .cursor/rules")
    p_aud.add_argument("--include-globals", action="store_true", help="一并扫描当前用户主目录下的全局 Agent 规则")
    p_aud.add_argument("--ns", default=None, help="限定扫描的记忆命名空间")
    p_aud.add_argument("--as-of", default=None, help="指定 YYYY-MM-DD 基准日统计 until: 时效")
    p_aud.add_argument("--out", default=None, help="可选：将 JSON 与 Markdown 体检报告写入目录")
    p_aud.add_argument("--json", action="store_true", help="输出 JSON")
    p_aud.set_defaults(func=_cmd_audit)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_utf8_io()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OllamaError, ValueError) as exc:
        print(f"[zmem] 错误: {exc}", file=sys.stderr)
        return 1
