"""ZeroMem (零忆) CLI: setup / connect / disconnect / doctor / config / add / search / stats / mcp / ui."""

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
from .memory import Memory, default_db_path, default_home_dir
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

MCP_SYSTEM_PROMPT = """# ZeroMem (零忆) MCP 高效检索纪律（实测 12/12 满分配置）

你已接入本地记忆库 MCP 工具 `memory_recall`、`memory_remember`、`memory_stats`、`memory_delete`（由本地 Qwen3.5-4B-Memory-V2 驱动，零云端写入费、本地双层脱敏）：

1. **先查后答**：涉及历史项目配置、参数、决议、完整哈希或进展状态时，先调用 `memory_recall` 核实，严禁凭记忆猜测。
2. **单主题 1 次短查**：单一主题问题提炼 2–4 个核心关键词调用 1 次 `memory_recall`（`limit=4` 或 `6`）即可。
3. **跨文档/多指标并行拆查**：当问题同时涉及多个独立模块或指标（如同时问“网关限流上限”与“数据库连接池上限”）时，在同一轮中**并行发起 2 次聚焦短关键词 `memory_recall`**，避免把整段长问句一次塞进单次 FTS5 查询。
4. **命中即答、诚实拒答**：拿到绑定原文证据后直接准确作答（保留精确数值、生效日期、64 位哈希及“仅提议未执行”等边界）；若检索无记录，最多换词补查 1 次即明确回答“未提及/无记录”，不空转、不编造。
"""

FAST_SDK_EXAMPLE = '''"""ZeroMem (零忆) 方案一：极速省钱版（单次预注 Python SDK 示例）。

5.2 万字实测对比全量硬塞：节省 94.3% 输入 Token、冷启动省 93.0% 费用、响应最快、0 云端密钥泄漏。
"""

from zeromem import Memory


def build_prompt_with_memory(question: str, *, ns: str = "default", top_k: int = 6) -> str:
    mem = Memory()
    hits = mem.recall(question, ns=ns, limit=top_k)
    if not hits:
        return f"问题：{question}"
    lines = ["以下是 ZeroMem 本地记忆检索到的相关事实与绑定原文证据："]
    for idx, h in enumerate(hits, 1):
        when = f"记录于 {h.mentioned_at}" + (f" | 生效于 {h.occurred_start}" if h.occurred_start else "")
        lines.append(f"{idx}. [{when}] 事实：{h.text}")
        for ev in h.evidence:
            lines.append(f"   原文({ev['id']}): {ev['text']}")
    lines.append(f"\\n---\\n请仅依据上述记忆准确回答：{question}")
    return "\\n".join(lines)


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

    mode_label = "方案一：极速省钱版（CLI / SDK 单次预注）" if args.mode == "fast" else "方案二：满分 Agent 版（MCP 自主拆查）"
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
        print(f"\n[方案一 · 极速省钱版] 已生成 Python SDK 单次预注示例: {written['fast_sdk_example']}")
        print("  快速体验：")
        print('    zmem add "2026-03-01 记录：网关限流 2026-03-15 调为 120 次/分。" --ns demo')
        print('    zmem search "网关限流" --ns demo')
        print("  如需随时切换为方案二（MCP 满分 Agent 版），执行：zmem connect --all")
    else:
        print(f"\n[方案二 · 满分 Agent 版] 已自动落盘全部配置模板至: {cfg_dir}")
        print(f"  - 通用 MCP 配置 (Claude/Cursor/Antigravity): {written['mcp_config']}")
        print(f"  - DSH MCP 补丁:                             {written['dsh_mcp_patch']}")
        print(f"  - 并行拆查满分检索提示词:                    {written['mcp_prompt']}")

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
        print(f"[zmem] 方案一（极速省钱版 · CLI / SDK 单次预注）示例已写入: {written['fast_sdk_example']}\n")
        print(FAST_SDK_EXAMPLE)
    else:
        print(f"[zmem] 方案二（满分 Agent 版 · MCP 自主拆查）配置已写入: {out_dir}")
        for label, path in written.items():
            print(f"  - {label}: {path}")
        print("\n--- MCP JSON（Claude Desktop / Cursor / Antigravity / Windsurf）---")
        print(json.dumps(_mcp_snippet(db_path, args.model, args.ollama_url), ensure_ascii=False, indent=2))
        print("\n--- 推荐粘贴到客户端规则的检索纪律提示词（12/12 满分实测配置）---")
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
        f"入库={res.saved} (修订={res.revised}, 丢弃={res.dropped}, 隔离={res.held}, 守卫拦截={res.guard_rejected})"
    )
    for f in res.facts:
        occ = f" [发生:{f['occurred_start']}]" if f.get("occurred_start") else ""
        print(f"  - ({f['type']}/{f['action']}){occ} {f['text']}  <ev:{','.join(f['evidence'])}>")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    mem = Memory(
        db_path=args.db,
        model=args.model,
        ollama_url=args.ollama_url,
        num_ctx=args.num_ctx,
    )
    query = " ".join(args.query)
    hits = mem.recall(query, ns=args.ns, limit=args.limit)
    if args.json:
        print(json.dumps([h.to_dict() for h in hits], ensure_ascii=False, indent=2))
        return 0

    print(f"[zmem] 检索 {query!r} -> 命中 {len(hits)} 条")
    for idx, h in enumerate(hits, 1):
        time_info = f"occurred={h.occurred_start or '-'} | recorded={h.mentioned_at}"
        print(f"\n{idx}. [{h.namespace}] ({h.type}/{h.action}, score={h.score}) {time_info}")
        print(f"   事实: {h.text}")
        for ev in h.evidence:
            role_tag = f"[{ev['role']}] " if ev.get("role") else ""
            print(f"   原文({ev['id']}): {role_tag}{ev['text']}")
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
        help="安装方案：fast（方案一·极速省钱 CLI/SDK）或 mcp（方案二·满分 Agent MCP，默认）",
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

    p_config = sub.add_parser("config", help="随时导出/查看两套接入方案的配置模板与满分提示词")
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
    p_search.add_argument("query", nargs="*", help="查询词（留空返回最新记忆）")
    p_search.add_argument("--ns", default=None, help="按命名空间过滤")
    p_search.add_argument("-k", "--limit", type=int, default=6, help="返回条数（默认 6）")
    p_search.add_argument("--json", action="store_true", help="输出原始 JSON")
    p_search.set_defaults(func=_cmd_search)

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
