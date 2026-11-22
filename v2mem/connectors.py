"""Zero-dependency auto-discovery & safe config connector for mainstream AI Agents.

Supported clients (Windows / macOS / Linux):
  - claude-desktop : Claude Desktop (JSON `mcpServers.zeromem`)
  - claude-code    : Claude Code CLI (`~/.claude.json`, JSON `mcpServers.zeromem`)
  - cursor         : Cursor IDE (`~/.cursor/mcp.json`, JSON `mcpServers.zeromem`)
  - windsurf       : Windsurf IDE (`~/.codeium/windsurf/mcp_config.json`, JSON)
  - codex          : OpenAI Codex CLI (`~/.codex/config.toml`, fenced TOML block)
  - antigravity    : Google Antigravity (`~/.gemini/antigravity/mcp_config.json`, JSON)
  - dsh            : DeepSeek Harness (`~/.zeromem/configs/dsh_mcp_patch.yml`, YAML patch)

Every write creates a timestamped `.bak` backup first, supports `--dry-run`, and is
cleanly reversible via `disconnect_client()`.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .memory import default_db_path, default_home_dir
from .ollama import DEFAULT_OLLAMA_URL

TOML_BEGIN = "# zeromem:begin"
TOML_END = "# zeromem:end"
RULE_BEGIN = "<!-- zeromem:begin -->"
RULE_END = "<!-- zeromem:end -->"

CURSOR_MDC_CONTENT = """---
description: ZeroMem (零忆) local memory retrieval & parallel multi-query discipline
alwaysApply: true
---

# ZeroMem (零忆) MCP 高效检索纪律

当回答涉及项目历史配置、参数、决议、完整哈希或状态进展时，优先调用 `zeromem` MCP 工具（`memory_recall` / `memory_remember` / `memory_stats`）：

1. **单主题 1 次短查询**：提炼 2–4 个核心关键词调用 1 次 `memory_recall`（`limit=4` 或 `6`）。
2. **跨文档/多指标并行拆查**：问题同时涉及多个独立模块或指标（如同时问“限流上限”与“数据库连接池上限”）时，同轮**并行发起 2 次聚焦短关键词 `memory_recall`**，避免把超长原句塞进单次查询。
3. **证据驱动、严禁编造**：命中绑定原文证据后直接准确作答，准确保留数值、生效日期、64 位哈希及“仅提议未执行”等边界；若检索无记录，最多换词补查 1 次后明确说明“未提及/无记录”。
"""

AGENTS_RULE_BLOCK = f"""{RULE_BEGIN}
## ZeroMem（零忆）本地记忆检索纪律
- 涉及历史配置、决议、参数、哈希或项目背景时，先调用 `zeromem` MCP 的 `memory_recall` 查证。
- 单主题提炼 2–4 个关键词查 1 次；跨模块/多指标复合问题同轮并行发起 2 次聚焦短关键词 `memory_recall`。
- 拿到原文证据即作答并保留精确数值、生效日期与状态边界；无记录最多换词补查 1 次即如实说明，严禁编造。
{RULE_END}
"""


@dataclass(frozen=True)
class ClientSpec:
    id: str
    display_name: str
    kind: str  # "json" | "toml" | "dsh_yaml"
    config_path: Path
    detect_paths: tuple[Path, ...]
    detect_cmds: tuple[str, ...] = ()
    note: str = ""


@dataclass
class ClientStatus:
    id: str
    display_name: str
    kind: str
    config_path: str
    installed: bool
    connected: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _home(override_home: Path | None = None) -> Path:
    return override_home if override_home is not None else Path.home()


def _claude_desktop_config(home: Path) -> tuple[Path, tuple[Path, ...]]:
    sys_name = platform.system()
    if sys_name == "Windows":
        appdata = Path(os.environ.get("APPDATA") or (home / "AppData" / "Roaming"))
        cfg = appdata / "Claude" / "claude_desktop_config.json"
        return cfg, (cfg.parent,)
    if sys_name == "Darwin":
        cfg = home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        return cfg, (cfg.parent, Path("/Applications/Claude.app"))
    cfg = home / ".config" / "Claude" / "claude_desktop_config.json"
    return cfg, (cfg.parent,)


def get_client_specs(home_override: Path | None = None) -> list[ClientSpec]:
    h = _home(home_override)
    cd_cfg, cd_detect = _claude_desktop_config(h)
    dsh_cfg = (h / ".zeromem" / "configs" / "dsh_mcp_patch.yml") if home_override else (default_home_dir() / "configs" / "dsh_mcp_patch.yml")

    return [
        ClientSpec(
            id="claude-desktop",
            display_name="Claude Desktop",
            kind="json",
            config_path=cd_cfg,
            detect_paths=cd_detect,
            note="写入 mcpServers.zeromem（重启 Claude Desktop 生效）",
        ),
        ClientSpec(
            id="claude-code",
            display_name="Claude Code",
            kind="json",
            config_path=h / ".claude.json",
            detect_paths=(h / ".claude.json", h / ".claude"),
            detect_cmds=("claude",),
            note="写入 ~/.claude.json mcpServers.zeromem",
        ),
        ClientSpec(
            id="cursor",
            display_name="Cursor",
            kind="json",
            config_path=h / ".cursor" / "mcp.json",
            detect_paths=(h / ".cursor",),
            detect_cmds=("cursor",),
            note="写入 ~/.cursor/mcp.json（重启或刷新 MCP 列表生效）",
        ),
        ClientSpec(
            id="windsurf",
            display_name="Windsurf",
            kind="json",
            config_path=h / ".codeium" / "windsurf" / "mcp_config.json",
            detect_paths=(h / ".codeium" / "windsurf",),
            detect_cmds=("windsurf",),
            note="写入 ~/.codeium/windsurf/mcp_config.json",
        ),
        ClientSpec(
            id="codex",
            display_name="Codex CLI",
            kind="toml",
            config_path=h / ".codex" / "config.toml",
            detect_paths=(h / ".codex",),
            detect_cmds=("codex",),
            note="写入 ~/.codex/config.toml [mcp_servers.zeromem] 隔离块",
        ),
        ClientSpec(
            id="antigravity",
            display_name="Antigravity",
            kind="json",
            config_path=h / ".gemini" / "antigravity" / "mcp_config.json",
            detect_paths=(h / ".gemini" / "antigravity",),
            note="写入 ~/.gemini/antigravity/mcp_config.json",
        ),
        ClientSpec(
            id="dsh",
            display_name="DeepSeek Harness (DSH)",
            kind="dsh_yaml",
            config_path=dsh_cfg,
            detect_paths=(h / ".dsh",),
            detect_cmds=("dsh",),
            note="生成官方 --patch 补丁文件（不破坏桌面端 profile）",
        ),
    ]


def _build_command_and_args(
    *,
    db_path: Path | None = None,
    model: str | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    python_exe: str | None = None,
) -> tuple[str, list[str]]:
    exe = python_exe or sys.executable
    resolved_db = Path(db_path).expanduser() if db_path else default_db_path()
    args = ["-m", "zeromem", "--db", str(resolved_db)]
    if model:
        args.extend(["--model", model])
    if ollama_url and ollama_url != DEFAULT_OLLAMA_URL:
        args.extend(["--ollama-url", ollama_url])
    args.append("mcp")
    return exe, args


def _is_connected(spec: ClientSpec) -> bool:
    p = spec.config_path
    if not p.is_file():
        return False
    try:
        raw = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    if spec.kind == "json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return False
        servers = data.get("mcpServers") if isinstance(data, dict) else None
        return isinstance(servers, dict) and ("zeromem" in servers or "v2-memory" in servers)
    if spec.kind == "toml":
        return (TOML_BEGIN in raw) or ("[mcp_servers.zeromem]" in raw)
    return "zeromem-mcp" in raw or "serverName: zeromem" in raw


def _is_installed(spec: ClientSpec) -> bool:
    if any(p.exists() for p in spec.detect_paths):
        return True
    if any(shutil.which(cmd) for cmd in spec.detect_cmds):
        return True
    return False


def detect_clients(home_override: Path | None = None) -> list[ClientStatus]:
    """Scan local machine for installed mainstream AI Agents and their ZeroMem connection state."""
    out: list[ClientStatus] = []
    for spec in get_client_specs(home_override=home_override):
        out.append(
            ClientStatus(
                id=spec.id,
                display_name=spec.display_name,
                kind=spec.kind,
                config_path=str(spec.config_path),
                installed=_is_installed(spec),
                connected=_is_connected(spec),
                note=spec.note,
            )
        )
    return out


def _backup_if_exists(path: Path) -> Path | None:
    if not path.is_file():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = path.with_name(f"{path.name}.bak-zeromem-{stamp}")
    shutil.copy2(path, bak)
    return bak


def _toml_quote(s: str) -> str:
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _render_codex_toml_block(exe: str, args: list[str]) -> str:
    args_str = ", ".join(_toml_quote(a) for a in args)
    return (
        f"{TOML_BEGIN}\n"
        "[mcp_servers.zeromem]\n"
        f"command = {_toml_quote(exe)}\n"
        f"args = [{args_str}]\n"
        f"{TOML_END}\n"
    )


def _render_dsh_yaml(exe: str, args: list[str]) -> str:
    exe_posix = Path(exe).as_posix()
    rendered_args = [Path(a).as_posix() if ("/" in a or "\\" in a) else a for a in args]
    args_lines = "\n".join(f"          - {item}" for item in rendered_args)
    return (
        "# ZeroMem (零忆) DSH MCP 补丁（使用 `dsh --patch <本文件>` 挂载）\n"
        "- insert:\n"
        "    - id: zeromem-mcp\n"
        "      name: '@deepseek-ai/dsh-mcp-client'\n"
        "      config:\n"
        "        serverName: zeromem\n"
        "        transport: stdio\n"
        f"        command: {exe_posix}\n"
        "        args:\n"
        f"{args_lines}\n"
        "        env: {}\n"
    )


def connect_client(
    client_id: str,
    *,
    db_path: Path | None = None,
    model: str | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    python_exe: str | None = None,
    dry_run: bool = False,
    home_override: Path | None = None,
) -> dict[str, Any]:
    """Safely back up and inject ZeroMem MCP configuration into a target Agent client."""
    specs = {s.id: s for s in get_client_specs(home_override=home_override)}
    spec = specs.get(client_id)
    if spec is None:
        raise ValueError(f"不支持的客户端 `{client_id}`，可选: {', '.join(specs)}")

    exe, args = _build_command_and_args(
        db_path=db_path,
        model=model,
        ollama_url=ollama_url,
        python_exe=python_exe,
    )
    cfg_path = spec.config_path

    if spec.kind == "json":
        existing_data: dict[str, Any] = {}
        if cfg_path.is_file():
            raw = cfg_path.read_text(encoding="utf-8")
            if raw.strip():
                loaded = json.loads(raw)
                if not isinstance(loaded, dict):
                    raise ValueError(f"{cfg_path} 根结构不是 JSON 对象，已中止写入以防损坏。")
                existing_data = loaded
        servers = existing_data.get("mcpServers")
        if not isinstance(servers, dict):
            servers = {}
        servers["zeromem"] = {"command": exe, "args": args}
        existing_data["mcpServers"] = servers
        new_text = json.dumps(existing_data, ensure_ascii=False, indent=2) + "\n"
    elif spec.kind == "toml":
        block = _render_codex_toml_block(exe, args)
        raw = cfg_path.read_text(encoding="utf-8") if cfg_path.is_file() else ""
        pattern = re.compile(rf"{re.escape(TOML_BEGIN)}.*?{re.escape(TOML_END)}\r?\n?", flags=re.DOTALL)
        if pattern.search(raw):
            new_text = pattern.sub(block, raw)
        else:
            sep = "\n\n" if raw.strip() else ""
            new_text = raw.rstrip() + sep + block
    else:
        new_text = _render_dsh_yaml(exe, args)

    backup_path: Path | None = None
    if not dry_run:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path = _backup_if_exists(cfg_path)
        cfg_path.write_text(new_text, encoding="utf-8")

    return {
        "client": spec.id,
        "display_name": spec.display_name,
        "config_path": str(cfg_path),
        "backup_path": str(backup_path) if backup_path else None,
        "dry_run": dry_run,
        "status": "preview" if dry_run else "connected",
        "note": spec.note,
    }


def disconnect_client(
    client_id: str,
    *,
    dry_run: bool = False,
    home_override: Path | None = None,
) -> dict[str, Any]:
    """Cleanly remove ZeroMem entry from a target Agent client (backing up first)."""
    specs = {s.id: s for s in get_client_specs(home_override=home_override)}
    spec = specs.get(client_id)
    if spec is None:
        raise ValueError(f"不支持的客户端 `{client_id}`，可选: {', '.join(specs)}")

    cfg_path = spec.config_path
    if not cfg_path.is_file():
        return {"client": spec.id, "config_path": str(cfg_path), "status": "not_found", "changed": False}

    changed = False
    backup_path: Path | None = None

    if spec.kind == "json":
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        servers = data.get("mcpServers") if isinstance(data, dict) else None
        if isinstance(servers, dict):
            for key in ("zeromem", "v2-memory"):
                if key in servers:
                    del servers[key]
                    changed = True
        if changed and not dry_run:
            backup_path = _backup_if_exists(cfg_path)
            cfg_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif spec.kind == "toml":
        raw = cfg_path.read_text(encoding="utf-8")
        pattern = re.compile(rf"\r?\n?{re.escape(TOML_BEGIN)}.*?{re.escape(TOML_END)}\r?\n?", flags=re.DOTALL)
        if pattern.search(raw):
            changed = True
            new_text = pattern.sub("\n", raw).strip() + "\n"
            if not dry_run:
                backup_path = _backup_if_exists(cfg_path)
                cfg_path.write_text(new_text, encoding="utf-8")
    else:
        changed = True
        if not dry_run:
            backup_path = _backup_if_exists(cfg_path)
            cfg_path.unlink(missing_ok=True)

    return {
        "client": spec.id,
        "display_name": spec.display_name,
        "config_path": str(cfg_path),
        "backup_path": str(backup_path) if backup_path else None,
        "changed": changed,
        "dry_run": dry_run,
        "status": "preview" if dry_run else ("disconnected" if changed else "unchanged"),
    }


def export_project_rules(project_dir: Path, *, dry_run: bool = False) -> dict[str, str]:
    """Write ready-to-use Cursor `.cursor/rules/zeromem.mdc` and append AGENTS.md rule block."""
    proj = project_dir.expanduser().resolve()
    mdc_path = proj / ".cursor" / "rules" / "zeromem.mdc"
    agents_path = proj / "AGENTS.md"

    existing_agents = agents_path.read_text(encoding="utf-8") if agents_path.is_file() else ""
    pattern = re.compile(rf"{re.escape(RULE_BEGIN)}.*?{re.escape(RULE_END)}\r?\n?", flags=re.DOTALL)
    if pattern.search(existing_agents):
        new_agents = pattern.sub(AGENTS_RULE_BLOCK, existing_agents)
    else:
        sep = "\n\n" if existing_agents.strip() else ""
        new_agents = existing_agents.rstrip() + sep + AGENTS_RULE_BLOCK

    if not dry_run:
        mdc_path.parent.mkdir(parents=True, exist_ok=True)
        mdc_path.write_text(CURSOR_MDC_CONTENT, encoding="utf-8")
        if agents_path.is_file():
            _backup_if_exists(agents_path)
        agents_path.write_text(new_agents, encoding="utf-8")

    return {
        "cursor_rule": str(mdc_path),
        "agents_md": str(agents_path),
    }
