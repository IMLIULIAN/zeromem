"""Zero-dependency desktop UI (tkinter) for ZeroMem (零忆):
1) Floating rounded status pill (always-on-top compact indicator)
2) Expandable mini status card (counts + quick actions)
3) Modern dark-slate Memory Workbench window:
   - Instant <1ms FTS5 search & namespace filter
   - Fact cards with pre-write admission/revision badges
   - Bound original evidence viewer with [REDACTED] security highlighting
   - Background Add-Memory modal & confirmed Delete action
   - Visual 1-click Connector panel for 7 mainstream AI Agents
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .connectors import ClientStatus, connect_client, detect_clients, disconnect_client
from .hardware import detect_hardware
from .memory import Memory, RecallHit
from .ollama import OllamaError, _is_v2_model_name

TRANSPARENT_KEY = "#010203"

PALETTE = {
    "bg_deep": "#0b1220",
    "bg": "#111827",
    "card": "#152033",
    "card_hover": "#1b2a42",
    "card_selected": "#1e3252",
    "input_bg": "#0e1726",
    "border": "#24344f",
    "border_hover": "#3b5278",
    "border_active": "#38bdf8",
    "text": "#f8fafc",
    "sub": "#cbd5e1",
    "muted": "#94a3b8",
    "dim": "#64748b",
    "resident_dot": "#10b981",
    "resident_ring": "#064e3b",
    "ready_dot": "#38bdf8",
    "ready_ring": "#0c4a6e",
    "offline_dot": "#f43f5e",
    "offline_ring": "#881337",
    "ok": "#10b981",
    "ok_bg": "#0d382e",
    "sky": "#38bdf8",
    "sky_bg": "#10324a",
    "warn": "#f59e0b",
    "warn_bg": "#422c0c",
    "danger": "#f43f5e",
    "danger_bg": "#451423",
    "pin_on": "#818cf8",
}

MCP_PROMPT_COPY = (
    "你已接入 ZeroMem（零忆）本地记忆 MCP 工具。涉及项目历史配置、参数、决议、哈希或进展时，先调用 memory_recall：\n"
    "1) 单主题问题提炼 2-4 个核心关键词查 1 次；\n"
    "2) 跨多个独立模块/指标的复合问题，同轮并行发起 2 次聚焦短关键词 memory_recall；\n"
    "3) 命中原文证据即准确作答（保留数值、生效日、完整哈希与提议边界），无记录最多换词补查 1 次后明说未提及。"
)


@dataclass
class HudSnapshot:
    """Thread-safe status snapshot rendered by the floating pill and workbench."""

    state: str  # "resident" | "ready" | "offline"
    badge: str
    model: str
    documents: int
    facts: int
    revised_facts: int
    namespaces: int
    db_size_bytes: int
    error: str = ""
    namespace_list: tuple[str, ...] = ()
    connected_agents: int = 0
    installed_agents: int = 0


def format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def collect_status(memory: Memory) -> HudSnapshot:
    """Collect SQLite counts, Agent connections, and Ollama V2 residency status without raising."""
    stats = memory.stats()
    db_size = 0
    try:
        db_file = Path(memory.db_path)
        if db_file.is_file():
            db_size = db_file.stat().st_size
    except OSError:
        db_size = 0

    state = "offline"
    badge = "离线"
    model_name = "Ollama 未连接"
    err_msg = ""

    try:
        running = memory.client.list_running_models()
        explicit = memory.client._explicit_model
        running_match = None
        for name in running:
            if (explicit and name == explicit) or (not explicit and _is_v2_model_name(name)):
                running_match = name
                break
        if running_match:
            state = "resident"
            badge = "常驻"
            model_name = running_match
        else:
            resolved = memory.client.resolve_model()
            state = "ready"
            badge = "待机"
            model_name = resolved
    except OllamaError as exc:
        err_msg = str(exc)
        if "未检测到" in err_msg:
            model_name = "未安装 V2"

    ns_raw = stats.get("namespaces") or []
    if isinstance(ns_raw, dict):
        ns_names = tuple(sorted(str(k) for k in ns_raw.keys()))
    elif isinstance(ns_raw, (list, tuple, set)):
        ns_names = tuple(sorted(str(x) for x in ns_raw))
    else:
        ns_names = ()

    connected = 0
    installed = 0
    try:
        clients = detect_clients()
        installed = sum(1 for c in clients if c.installed)
        connected = sum(1 for c in clients if c.connected)
    except Exception:
        pass

    return HudSnapshot(
        state=state,
        badge=badge,
        model=model_name,
        documents=int(stats.get("documents") or 0),
        facts=int(stats.get("facts") or 0),
        revised_facts=int(stats.get("revised_facts") or 0),
        namespaces=len(ns_names),
        db_size_bytes=db_size,
        error=err_msg,
        namespace_list=ns_names,
        connected_agents=connected,
        installed_agents=installed,
    )


def _enable_windows_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _enable_windows_dark_titlebar(tk_win: Any) -> None:
    """Enable native Windows 10/11 dark titlebar via DWM without external dependencies."""
    if sys.platform != "win32":
        return
    try:
        tk_win.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(tk_win.winfo_id()) or tk_win.winfo_id()
        val = ctypes.c_int(1)
        for attr in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE (Win11 / Win10 1903+)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(val), ctypes.sizeof(val))
    except Exception:
        pass


def _make_app_icon(tk_mod: Any) -> Any:
    """Generate a crisp 24x24 emerald-ring ZeroMem window icon in pure stdlib PhotoImage."""
    icon = tk_mod.PhotoImage(width=24, height=24)
    bg = "#0b1220"
    ring = "#10b981"
    core = "#38bdf8"
    for y in range(24):
        row_colors = []
        for x in range(24):
            dx, dy = x - 11.5, y - 11.5
            d2 = dx * dx + dy * dy
            if 52 <= d2 <= 104:
                row_colors.append(ring)
            elif d2 <= 18:
                row_colors.append(core)
            else:
                row_colors.append(bg)
        icon.put("{" + " ".join(row_colors) + "}", to=(0, y))
    return icon


def _ui_font() -> str:
    if sys.platform == "win32":
        return "Microsoft YaHei UI"
    if sys.platform == "darwin":
        return "PingFang SC"
    return "DejaVu Sans"


def _mono_font() -> str:
    if sys.platform == "win32":
        return "Consolas"
    if sys.platform == "darwin":
        return "Menlo"
    return "DejaVu Sans Mono"


def _draw_rounded_rect(
    canvas: Any,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    r: float,
    *,
    fill: str,
    outline: str = "",
    width: float = 1.0,
) -> None:
    r = min(r, (x2 - x1) / 2.0, (y2 - y1) / 2.0)
    pts = [
        x1 + r, y1,
        x2 - r, y1,
        x2, y1,
        x2, y1 + r,
        x2, y2 - r,
        x2, y2,
        x2 - r, y2,
        x1 + r, y2,
        x1, y2,
        x1, y2 - r,
        x1, y1 + r,
        x1, y1,
    ]
    canvas.create_polygon(pts, smooth=True, splinesteps=24, fill=fill, outline=outline, width=width)


class MemoryWorkbenchWindow:
    """Modern dark-slate desktop workbench for visual search, evidence inspection, add/delete & Agent connect."""

    def __init__(self, parent_hud: "FloatingStatusHud") -> None:
        self.hud = parent_hud
        self.tk = parent_hud.tk
        from tkinter import messagebox, ttk

        self.ttk = ttk
        self.messagebox = messagebox
        self.memory = parent_hud.memory
        self.font = _ui_font()
        self.mono = _mono_font()
        self.hw_label = detect_hardware().summary_line()

        self.win = self.tk.Toplevel(parent_hud.root)
        self.win.title("ZeroMem · 零忆工作台")
        self.win.configure(bg=PALETTE["bg_deep"])
        self.win.geometry("1040x680")
        self.win.minsize(820, 540)
        self.win.protocol("WM_DELETE_WINDOW", self.hide)
        try:
            self._icon = _make_app_icon(self.tk)
            self.win.iconphoto(False, self._icon)
        except Exception:
            self._icon = None
        _enable_windows_dark_titlebar(self.win)

        self._hits: list[RecallHit] = []
        self._selected_id: int | None = None
        self._search_after_id: str | None = None
        self._card_frames: dict[int, Any] = {}

        self._configure_ttk_styles()
        self._build_layout()
        self.refresh_all()

    def _configure_ttk_styles(self) -> None:
        style = self.ttk.Style(self.win)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "ZM.TCombobox",
            fieldbackground=PALETTE["input_bg"],
            background=PALETTE["card"],
            foreground=PALETTE["text"],
            bordercolor=PALETTE["border"],
            arrowcolor=PALETTE["muted"],
            padding=4,
        )
        style.map(
            "ZM.TCombobox",
            fieldbackground=[("readonly", PALETTE["input_bg"])],
            foreground=[("readonly", PALETTE["text"])],
        )
        style.configure(
            "ZM.Vertical.TScrollbar",
            background=PALETTE["card_hover"],
            troughcolor=PALETTE["bg_deep"],
            bordercolor=PALETTE["bg_deep"],
            arrowcolor=PALETTE["muted"],
            relief="flat",
        )

    def _make_btn(
        self,
        parent: Any,
        text: str,
        command: Any,
        *,
        bg: str = PALETTE["card"],
        fg: str = PALETTE["text"],
        hover_bg: str = PALETTE["card_hover"],
        border: str = PALETTE["border"],
        padx: int = 12,
        pady: int = 5,
        bold: bool = False,
        font_size: int = 9,
    ) -> Any:
        tk = self.tk
        weight = "bold" if bold else "normal"
        btn = tk.Label(
            parent,
            text=text,
            bg=bg,
            fg=fg,
            font=(self.font, font_size, weight),
            padx=padx,
            pady=pady,
            cursor="hand2",
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor=border,
        )
        btn.bind("<Button-1>", lambda _e: command())
        btn.bind("<Enter>", lambda _e: btn.configure(bg=hover_bg))
        btn.bind("<Leave>", lambda _e: btn.configure(bg=bg))
        return btn

    def _build_layout(self) -> None:
        tk = self.tk

        # 1. Top Toolbar
        top = tk.Frame(self.win, bg=PALETTE["bg"], height=60)
        top.pack(fill="x", side="top")
        top.pack_propagate(False)
        tk.Frame(self.win, bg=PALETTE["border"], height=1).pack(fill="x", side="top")

        # Brand & live status badge
        brand_box = tk.Frame(top, bg=PALETTE["bg"])
        brand_box.pack(side="left", padx=(18, 14), pady=12)
        self.dot_lbl = tk.Label(brand_box, text="●", bg=PALETTE["bg"], fg=PALETTE["ok"], font=(self.font, 13))
        self.dot_lbl.pack(side="left", padx=(0, 6))
        tk.Label(
            brand_box,
            text="ZeroMem",
            bg=PALETTE["bg"],
            fg=PALETTE["text"],
            font=(self.font, 13, "bold"),
        ).pack(side="left")
        self.state_pill = tk.Label(
            brand_box,
            text="检测中",
            bg=PALETTE["ok_bg"],
            fg=PALETTE["ok"],
            font=(self.font, 8, "bold"),
            padx=7,
            pady=2,
        )
        self.state_pill.pack(side="left", padx=(8, 0))

        # Search bar
        search_wrap = tk.Frame(
            top,
            bg=PALETTE["input_bg"],
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
            highlightcolor=PALETTE["sky"],
        )
        search_wrap.pack(side="left", fill="x", expand=True, pady=12, padx=(4, 10))
        tk.Label(search_wrap, text="⌕", bg=PALETTE["input_bg"], fg=PALETTE["muted"], font=(self.font, 11)).pack(
            side="left", padx=(10, 4)
        )
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(
            search_wrap,
            textvariable=self.search_var,
            bg=PALETTE["input_bg"],
            fg=PALETTE["text"],
            insertbackground=PALETTE["sky"],
            relief="flat",
            bd=0,
            font=(self.font, 10),
        )
        self.search_entry.pack(side="left", fill="x", expand=True, ipady=6)
        self.search_entry.bind("<KeyRelease>", self._on_search_change)
        self.search_entry.bind("<Return>", lambda _e: self.run_search())

        self.search_meta = tk.Label(
            search_wrap,
            text="",
            bg=PALETTE["input_bg"],
            fg=PALETTE["dim"],
            font=(self.font, 8),
            padx=8,
        )
        self.search_meta.pack(side="right")

        # Namespace selector
        self.ns_var = tk.StringVar(value="全部空间")
        self.ns_combo = self.ttk.Combobox(
            top,
            textvariable=self.ns_var,
            state="readonly",
            width=14,
            style="ZM.TCombobox",
            font=(self.font, 9),
        )
        self.ns_combo.pack(side="left", padx=(0, 10), pady=12)
        self.ns_combo.bind("<<ComboboxSelected>>", lambda _e: self.run_search())

        # Action buttons
        actions = tk.Frame(top, bg=PALETTE["bg"])
        actions.pack(side="right", padx=(0, 16), pady=12)

        self.btn_add = self._make_btn(
            actions,
            "+ 记一条",
            self.open_add_dialog,
            bg=PALETTE["ok_bg"],
            fg=PALETTE["ok"],
            hover_bg="#145344",
            border=PALETTE["ok"],
            bold=True,
        )
        self.btn_add.pack(side="left", padx=(0, 8))

        self.btn_agents = self._make_btn(
            actions,
            "⚡ Agent",
            self.open_agents_dialog,
            bg=PALETTE["sky_bg"],
            fg=PALETTE["sky"],
            hover_bg="#174463",
            border=PALETTE["sky"],
            bold=True,
        )
        self.btn_agents.pack(side="left", padx=(0, 8))

        self.btn_refresh = self._make_btn(actions, "↻", self.refresh_all, padx=9)
        self.btn_refresh.pack(side="left")

        # 2. Compact KPI & Hardware Bar
        kpi = tk.Frame(self.win, bg=PALETTE["bg_deep"], height=36)
        kpi.pack(fill="x", side="top", padx=18, pady=(8, 4))
        self.kpi_left = tk.Label(
            kpi,
            text="加载统计中…",
            bg=PALETTE["bg_deep"],
            fg=PALETTE["muted"],
            font=(self.font, 9),
            anchor="w",
        )
        self.kpi_left.pack(side="left")
        self.kpi_right = tk.Label(
            kpi,
            text=self.hw_label,
            bg=PALETTE["bg_deep"],
            fg=PALETTE["dim"],
            font=(self.font, 8),
            anchor="e",
        )
        self.kpi_right.pack(side="right")

        # 3. Main Split View (Left List 44% | Right Evidence Detail 56%)
        body = tk.Frame(self.win, bg=PALETTE["bg_deep"])
        body.pack(fill="both", expand=True, padx=16, pady=(4, 14))
        body.columnconfigure(0, weight=44, uniform="col")
        body.columnconfigure(1, weight=56, uniform="col")
        body.rowconfigure(0, weight=1)

        # Left Pane: Scrollable Fact Cards
        left_shell = tk.Frame(
            body,
            bg=PALETTE["bg"],
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
        )
        left_shell.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_shell.rowconfigure(1, weight=1)
        left_shell.columnconfigure(0, weight=1)

        left_hdr = tk.Frame(left_shell, bg=PALETTE["bg"])
        left_hdr.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 6))
        self.list_title = tk.Label(
            left_hdr,
            text="记忆列表",
            bg=PALETTE["bg"],
            fg=PALETTE["text"],
            font=(self.font, 10, "bold"),
        )
        self.list_title.pack(side="left")
        tk.Label(
            left_hdr,
            text="点选查看原文证据",
            bg=PALETTE["bg"],
            fg=PALETTE["dim"],
            font=(self.font, 8),
        ).pack(side="right")

        self.list_canvas = tk.Canvas(left_shell, bg=PALETTE["bg"], highlightthickness=0, bd=0)
        self.list_canvas.grid(row=1, column=0, sticky="nsew", padx=(8, 0), pady=(0, 8))
        self.list_scroll = self.ttk.Scrollbar(
            left_shell,
            orient="vertical",
            command=self.list_canvas.yview,
            style="ZM.Vertical.TScrollbar",
        )
        self.list_scroll.grid(row=1, column=1, sticky="ns", pady=(0, 8), padx=(2, 4))
        self.list_canvas.configure(yscrollcommand=self.list_scroll.set)

        self.list_inner = tk.Frame(self.list_canvas, bg=PALETTE["bg"])
        self._list_win_id = self.list_canvas.create_window((0, 0), window=self.list_inner, anchor="nw")
        self.list_inner.bind("<Configure>", lambda _e: self.list_canvas.configure(scrollregion=self.list_canvas.bbox("all")))
        self.list_canvas.bind("<Configure>", lambda e: self.list_canvas.itemconfigure(self._list_win_id, width=e.width))
        self.list_canvas.bind("<MouseWheel>", self._on_mousewheel)

        # Right Pane: Detail & Bound Original Evidence
        right_shell = tk.Frame(
            body,
            bg=PALETTE["bg"],
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
        )
        right_shell.grid(row=0, column=1, sticky="nsew")
        right_shell.rowconfigure(1, weight=1)
        right_shell.columnconfigure(0, weight=1)

        right_hdr = tk.Frame(right_shell, bg=PALETTE["bg"])
        right_hdr.grid(row=0, column=0, sticky="ew", padx=16, pady=(10, 6))
        self.detail_hdr_lbl = tk.Label(
            right_hdr,
            text="原文溯源与证据详情",
            bg=PALETTE["bg"],
            fg=PALETTE["text"],
            font=(self.font, 10, "bold"),
        )
        self.detail_hdr_lbl.pack(side="left")

        self.detail_actions = tk.Frame(right_hdr, bg=PALETTE["bg"])
        self.detail_actions.pack(side="right")
        self.btn_copy = self._make_btn(self.detail_actions, "复制", self._copy_selected, padx=10, pady=3, font_size=8)
        self.btn_copy.pack(side="left", padx=(0, 6))
        self.btn_del = self._make_btn(
            self.detail_actions,
            "删除",
            self._delete_selected,
            bg=PALETTE["danger_bg"],
            fg=PALETTE["danger"],
            hover_bg="#631b30",
            border=PALETTE["danger"],
            padx=10,
            pady=3,
            font_size=8,
        )
        self.btn_del.pack(side="left")

        tk.Frame(right_shell, bg=PALETTE["border"], height=1).grid(row=0, column=0, sticky="sew", padx=12)

        self.detail_text = tk.Text(
            right_shell,
            bg=PALETTE["bg"],
            fg=PALETTE["text"],
            relief="flat",
            bd=0,
            wrap="word",
            padx=16,
            pady=12,
            spacing1=3,
            spacing2=2,
            spacing3=6,
            font=(self.font, 10),
            cursor="arrow",
        )
        self.detail_text.grid(row=1, column=0, sticky="nsew")
        self.detail_text.configure(state="disabled")

        # Rich text tags for clean typography
        self.detail_text.tag_configure("fact", foreground=PALETTE["text"], font=(self.font, 11, "bold"))
        self.detail_text.tag_configure("sec_hdr", foreground=PALETTE["sky"], font=(self.font, 9, "bold"), spacing1=10)
        self.detail_text.tag_configure("meta_key", foreground=PALETTE["dim"], font=(self.font, 9))
        self.detail_text.tag_configure("meta_val", foreground=PALETTE["sub"], font=(self.font, 9))
        self.detail_text.tag_configure("badge_ok", foreground=PALETTE["ok"], background=PALETTE["ok_bg"], font=(self.font, 8, "bold"))
        self.detail_text.tag_configure("badge_warn", foreground=PALETTE["warn"], background=PALETTE["warn_bg"], font=(self.font, 8, "bold"))
        self.detail_text.tag_configure("badge_sky", foreground=PALETTE["sky"], background=PALETTE["sky_bg"], font=(self.font, 8, "bold"))
        self.detail_text.tag_configure("ev_id", foreground=PALETTE["ok"], font=(self.mono, 9, "bold"))
        self.detail_text.tag_configure("ev_body", foreground=PALETTE["sub"], background=PALETTE["card"], font=(self.font, 9), lmargin1=10, lmargin2=10, rmargin=10)
        self.detail_text.tag_configure("redacted", foreground=PALETTE["ok"], background=PALETTE["ok_bg"], font=(self.mono, 9, "bold"))

    def _on_mousewheel(self, event: Any) -> None:
        delta = -1 * int(event.delta / 120) if event.delta else 0
        if delta:
            self.list_canvas.yview_scroll(delta, "units")

    def _on_search_change(self, _event: Any = None) -> None:
        if self._search_after_id:
            self.win.after_cancel(self._search_after_id)
        self._search_after_id = self.win.after(120, self.run_search)

    def show(self) -> None:
        self.win.deiconify()
        self.win.lift()
        self.win.focus_force()
        self.refresh_all()

    def hide(self) -> None:
        self.win.withdraw()

    def update_header_from_snapshot(self, snap: HudSnapshot) -> None:
        if not self.win.winfo_exists():
            return
        dot_map = {
            "resident": (PALETTE["ok"], PALETTE["ok_bg"]),
            "ready": (PALETTE["sky"], PALETTE["sky_bg"]),
            "offline": (PALETTE["danger"], PALETTE["danger_bg"]),
        }
        fg, bg = dot_map.get(snap.state, (PALETTE["muted"], PALETTE["card"]))
        self.dot_lbl.configure(fg=fg)
        self.state_pill.configure(text=f"{snap.badge} · {snap.model[:22]}", fg=fg, bg=bg)
        self.kpi_left.configure(
            text=(
                f"文档 {snap.documents}  ·  事实 {snap.facts}  ·  写前纠错 {snap.revised_facts}  ·  "
                f"{snap.namespaces} 空间  ·  {format_bytes(snap.db_size_bytes)}"
            )
        )
        self.btn_agents.configure(text=f"⚡ Agent ({snap.connected_agents}/{snap.installed_agents})")

        values = ["全部空间", *snap.namespace_list]
        self.ns_combo["values"] = values
        if self.ns_var.get() not in values:
            self.ns_var.set("全部空间")

    def refresh_all(self) -> None:
        snap = collect_status(self.memory)
        with self.hud._lock:
            self.hud._snapshot = snap
        self.update_header_from_snapshot(snap)
        self.run_search()

    def run_search(self) -> None:
        query = self.search_var.get().strip()
        ns_sel = self.ns_var.get()
        ns = None if (not ns_sel or ns_sel == "全部空间") else ns_sel

        t0 = time.monotonic()
        try:
            hits = self.memory.recall(query, ns=ns, limit=40)
        except Exception:
            hits = []
        dt_ms = (time.monotonic() - t0) * 1000.0

        self._hits = hits
        if query:
            self.search_meta.configure(text=f"{len(hits)} 条 · {dt_ms:.1f}ms")
            self.list_title.configure(text=f"检索结果 ({len(hits)})")
        else:
            self.search_meta.configure(text=f"最新 {len(hits)} 条")
            self.list_title.configure(text=f"最新记忆 ({len(hits)})")

        self._render_list()

    def _render_list(self) -> None:
        tk = self.tk
        for child in self.list_inner.winfo_children():
            child.destroy()
        self._card_frames.clear()

        if not self._hits:
            tk.Label(
                self.list_inner,
                text="暂无匹配记忆\n\n点击右上角「+ 记一条」或运行 `zmem add` 写入第一条记录。",
                bg=PALETTE["bg"],
                fg=PALETTE["dim"],
                font=(self.font, 9),
                justify="center",
                pady=40,
            ).pack(fill="x")
            self._render_detail(None)
            return

        valid_ids = {h.fact_id for h in self._hits}
        if self._selected_id not in valid_ids:
            self._selected_id = self._hits[0].fact_id

        for hit in self._hits:
            selected = hit.fact_id == self._selected_id
            card_bg = PALETTE["card_selected"] if selected else PALETTE["card"]
            border_col = PALETTE["sky"] if selected else PALETTE["border"]
            accent_col = PALETTE["warn"] if hit.action == "revise" else PALETTE["ok"]

            outer = tk.Frame(
                self.list_inner,
                bg=card_bg,
                highlightthickness=1,
                highlightbackground=border_col,
                cursor="hand2",
            )
            outer.pack(fill="x", pady=(0, 7), padx=(2, 6))
            self._card_frames[hit.fact_id] = outer

            bar = tk.Frame(outer, bg=accent_col, width=4)
            bar.pack(side="left", fill="y")

            content = tk.Frame(outer, bg=card_bg)
            content.pack(side="left", fill="both", expand=True, padx=10, pady=8)

            # Meta row: namespace + action badge + date
            meta_row = tk.Frame(content, bg=card_bg)
            meta_row.pack(fill="x")

            tk.Label(
                meta_row,
                text=hit.namespace,
                bg=PALETTE["sky_bg"],
                fg=PALETTE["sky"],
                font=(self.font, 7, "bold"),
                padx=5,
                pady=1,
            ).pack(side="left", padx=(0, 5))

            act_txt = "已纠错" if hit.action == "revise" else "已验收"
            act_bg = PALETTE["warn_bg"] if hit.action == "revise" else PALETTE["ok_bg"]
            tk.Label(
                meta_row,
                text=act_txt,
                bg=act_bg,
                fg=accent_col,
                font=(self.font, 7, "bold"),
                padx=5,
                pady=1,
            ).pack(side="left")

            date_txt = (hit.occurred_start or hit.mentioned_at or "")[:10]
            tk.Label(
                meta_row,
                text=date_txt,
                bg=card_bg,
                fg=PALETTE["dim"],
                font=(self.mono, 8),
            ).pack(side="right")

            preview = hit.text.replace("\n", " ")
            if len(preview) > 78:
                preview = preview[:77] + "…"
            lbl = tk.Label(
                content,
                text=preview,
                bg=card_bg,
                fg=PALETTE["text"],
                font=(self.font, 9),
                anchor="w",
                justify="left",
                wraplength=350,
            )
            lbl.pack(fill="x", pady=(5, 0))

            def _bind_click(widget: Any, fid: int = hit.fact_id) -> None:
                widget.bind("<Button-1>", lambda _e, target=fid: self._select_fact(target))
                widget.bind("<MouseWheel>", self._on_mousewheel)

            for w in (outer, bar, content, meta_row, lbl):
                _bind_click(w)
            for child in meta_row.winfo_children():
                _bind_click(child)

        chosen = next((h for h in self._hits if h.fact_id == self._selected_id), self._hits[0])
        self._render_detail(chosen)

    def _select_fact(self, fact_id: int) -> None:
        self._selected_id = fact_id
        self._render_list()

    def _render_detail(self, hit: RecallHit | None) -> None:
        t = self.detail_text
        t.configure(state="normal")
        t.delete("1.0", "end")

        if hit is None:
            t.insert("end", "选择左侧任意一条记忆，即可核对提炼事实、时间边界与绑定原文证据。", "meta_key")
            t.configure(state="disabled")
            return

        # Badges line
        t.insert("end", f" {hit.namespace} ", "badge_sky")
        t.insert("end", "  ")
        if hit.action == "revise":
            t.insert("end", " V2 写前审校已纠错 ", "badge_warn")
        else:
            t.insert("end", " V2 写前审校通过 ", "badge_ok")
        t.insert("end", f"   #{hit.fact_id} · {hit.type}\n\n", "meta_key")

        # Main fact text
        t.insert("end", f"{hit.text}\n\n", "fact")

        # Time & source metadata
        t.insert("end", "记录时间：", "meta_key")
        t.insert("end", f"{hit.mentioned_at}\n", "meta_val")
        t.insert("end", "发生/生效：", "meta_key")
        t.insert("end", f"{hit.occurred_start or '未单独指定（与记录日隔离）'}\n", "meta_val")
        t.insert("end", "来源文档：", "meta_key")
        t.insert("end", f"{hit.doc_id}\n", "meta_val")
        if hit.entities:
            t.insert("end", "核心实体：", "meta_key")
            t.insert("end", " · ".join(hit.entities) + "\n", "meta_val")

        # Bound original evidence segments (with [REDACTED] highlight)
        t.insert("end", "\n绑定原文证据（零幻觉溯源）\n", "sec_hdr")
        if not hit.evidence:
            t.insert("end", "无绑定原文片段。\n", "meta_key")
        else:
            for ev in hit.evidence:
                role = f" [{ev['role']}]" if ev.get("role") else ""
                t.insert("end", f"\n▸ 原文片段 {ev['id']}{role}\n", "ev_id")
                body_txt = str(ev.get("text") or "")
                parts = body_txt.split("[REDACTED]")
                for idx, chunk in enumerate(parts):
                    if chunk:
                        t.insert("end", chunk, "ev_body")
                    if idx < len(parts) - 1:
                        t.insert("end", "[REDACTED]", "redacted")
                t.insert("end", "\n", "ev_body")

        t.configure(state="disabled")

    def _copy_selected(self) -> None:
        hit = next((h for h in self._hits if h.fact_id == self._selected_id), None)
        if not hit:
            return
        lines = [f"[{hit.namespace}] {hit.text}"]
        for ev in hit.evidence:
            lines.append(f"原文({ev['id']}): {ev['text']}")
        self.win.clipboard_clear()
        self.win.clipboard_append("\n".join(lines))
        self.btn_copy.configure(text="已复制 ✓", fg=PALETTE["ok"])
        self.win.after(1200, lambda: self.btn_copy.configure(text="复制", fg=PALETTE["text"]))

    def _delete_selected(self) -> None:
        hit = next((h for h in self._hits if h.fact_id == self._selected_id), None)
        if not hit:
            return
        if not self.messagebox.askyesno(
            "确认删除记忆",
            f"确定永久删除记忆 #{hit.fact_id}？\n\n{hit.text[:80]}",
            parent=self.win,
        ):
            return
        self.memory.delete_fact(hit.fact_id)
        self._selected_id = None
        self.refresh_all()

    def open_add_dialog(self) -> None:
        tk = self.tk
        dlg = tk.Toplevel(self.win)
        dlg.title("ZeroMem · 写入新记忆")
        dlg.configure(bg=PALETTE["bg"])
        dlg.geometry("560x380")
        dlg.transient(self.win)
        dlg.grab_set()
        _enable_windows_dark_titlebar(dlg)

        top = tk.Frame(dlg, bg=PALETTE["bg"])
        top.pack(fill="x", padx=18, pady=(16, 8))
        tk.Label(top, text="空间", bg=PALETTE["bg"], fg=PALETTE["muted"], font=(self.font, 9)).pack(side="left")
        ns_var = tk.StringVar(value=self.ns_var.get() if self.ns_var.get() != "全部空间" else "default")
        tk.Entry(
            top,
            textvariable=ns_var,
            bg=PALETTE["input_bg"],
            fg=PALETTE["text"],
            insertbackground=PALETTE["sky"],
            relief="flat",
            width=18,
            font=(self.font, 9),
        ).pack(side="left", padx=(8, 16), ipady=4)

        verify_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            top,
            text="开启 V2 写前审校（推荐）",
            variable=verify_var,
            bg=PALETTE["bg"],
            fg=PALETTE["sub"],
            selectcolor=PALETTE["input_bg"],
            activebackground=PALETTE["bg"],
            activeforeground=PALETTE["text"],
            font=(self.font, 9),
        ).pack(side="left")

        txt = tk.Text(
            dlg,
            bg=PALETTE["input_bg"],
            fg=PALETTE["text"],
            insertbackground=PALETTE["sky"],
            relief="flat",
            wrap="word",
            font=(self.font, 10),
            padx=12,
            pady=10,
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
        )
        txt.pack(fill="both", expand=True, padx=18, pady=6)
        txt.focus_set()

        status_lbl = tk.Label(dlg, text="本地 4B 抽取 + 自动脱敏，消耗 0 云端 Token", bg=PALETTE["bg"], fg=PALETTE["dim"], font=(self.font, 8))
        status_lbl.pack(anchor="w", padx=18, pady=(2, 6))

        foot = tk.Frame(dlg, bg=PALETTE["bg"])
        foot.pack(fill="x", padx=18, pady=(0, 14))

        busy = {"flag": False}

        def _submit() -> None:
            if busy["flag"]:
                return
            content = txt.get("1.0", "end").strip()
            if not content:
                status_lbl.configure(text="请先输入要记住的内容。", fg=PALETTE["warn"])
                return
            busy["flag"] = True
            status_lbl.configure(text="正在调用本地 Qwen3.5-4B-Memory-V2 抽取并验收…", fg=PALETTE["sky"])

            def _worker() -> None:
                try:
                    res = self.memory.remember(content, ns=ns_var.get().strip() or "default", verify=verify_var.get())
                    def _done() -> None:
                        dlg.destroy()
                        self.refresh_all()
                        self.messagebox.showinfo(
                            "写入完成",
                            f"已入库 {res.saved} 条事实（写前纠错 {res.revised} · 过滤 {res.dropped}）",
                            parent=self.win,
                        )
                    self.win.after(0, _done)
                except Exception as exc:
                    def _err() -> None:
                        busy["flag"] = False
                        status_lbl.configure(text=f"写入失败：{exc}", fg=PALETTE["danger"])
                    self.win.after(0, _err)

            threading.Thread(target=_worker, daemon=True).start()

        self._make_btn(
            foot,
            "写入记忆库",
            _submit,
            bg=PALETTE["ok_bg"],
            fg=PALETTE["ok"],
            hover_bg="#145344",
            border=PALETTE["ok"],
            bold=True,
        ).pack(side="right")
        self._make_btn(foot, "取消", dlg.destroy).pack(side="right", padx=(0, 8))

    def open_agents_dialog(self) -> None:
        tk = self.tk
        dlg = tk.Toplevel(self.win)
        dlg.title("ZeroMem · 主流 AI Agent 一键接入")
        dlg.configure(bg=PALETTE["bg_deep"])
        dlg.geometry("680x470")
        dlg.transient(self.win)
        dlg.grab_set()
        _enable_windows_dark_titlebar(dlg)

        hdr = tk.Frame(dlg, bg=PALETTE["bg_deep"])
        hdr.pack(fill="x", padx=18, pady=(14, 6))
        tk.Label(
            hdr,
            text="⚡ 一键接入主流 AI Agent（自动备份原配置 .bak）",
            bg=PALETTE["bg_deep"],
            fg=PALETTE["text"],
            font=(self.font, 11, "bold"),
        ).pack(side="left")

        list_box = tk.Frame(dlg, bg=PALETTE["bg_deep"])
        list_box.pack(fill="both", expand=True, padx=18, pady=6)

        msg_lbl = tk.Label(dlg, text="所有配置变更均在修改前自动创建 .bak 备份，随时可一键断开。", bg=PALETTE["bg_deep"], fg=PALETTE["dim"], font=(self.font, 8))
        msg_lbl.pack(anchor="w", padx=18, pady=(2, 6))

        def _render_rows() -> None:
            for w in list_box.winfo_children():
                w.destroy()
            clients: list[ClientStatus] = detect_clients()
            for c in clients:
                row = tk.Frame(
                    list_box,
                    bg=PALETTE["card"],
                    highlightthickness=1,
                    highlightbackground=PALETTE["ok"] if c.connected else PALETTE["border"],
                )
                row.pack(fill="x", pady=4)

                dot_col = PALETTE["ok"] if c.connected else (PALETTE["sky"] if c.installed else PALETTE["dim"])
                tk.Label(row, text="●", bg=PALETTE["card"], fg=dot_col, font=(self.font, 11)).pack(side="left", padx=(12, 8), pady=10)

                info = tk.Frame(row, bg=PALETTE["card"])
                info.pack(side="left", fill="x", expand=True, pady=6)
                status_txt = "已接入" if c.connected else ("已检测到" if c.installed else "未检测到（可强制生成）")
                tk.Label(
                    info,
                    text=f"{c.display_name}   ·   {status_txt}",
                    bg=PALETTE["card"],
                    fg=PALETTE["text"] if c.installed or c.connected else PALETTE["muted"],
                    font=(self.font, 9, "bold"),
                    anchor="w",
                ).pack(fill="x")
                tk.Label(
                    info,
                    text=c.config_path,
                    bg=PALETTE["card"],
                    fg=PALETTE["dim"],
                    font=(self.mono, 8),
                    anchor="w",
                ).pack(fill="x")

                def _toggle(cid: str = c.id, is_conn: bool = c.connected) -> None:
                    try:
                        if is_conn:
                            r = disconnect_client(cid)
                            msg_lbl.configure(text=f"已断开 {r.get('display_name', cid)}", fg=PALETTE["warn"])
                        else:
                            r = connect_client(cid, db_path=self.memory.db_path)
                            msg_lbl.configure(text=f"✓ 已接入 {r['display_name']} -> {r['config_path']}", fg=PALETTE["ok"])
                        _render_rows()
                        self.refresh_all()
                    except Exception as exc:
                        msg_lbl.configure(text=f"操作失败：{exc}", fg=PALETTE["danger"])

                if c.connected:
                    self._make_btn(
                        row,
                        "断开",
                        _toggle,
                        bg=PALETTE["danger_bg"],
                        fg=PALETTE["danger"],
                        hover_bg="#631b30",
                        border=PALETTE["danger"],
                        padx=12,
                        pady=4,
                        font_size=8,
                    ).pack(side="right", padx=12)
                else:
                    self._make_btn(
                        row,
                        "一键接入",
                        _toggle,
                        bg=PALETTE["ok_bg"],
                        fg=PALETTE["ok"],
                        hover_bg="#145344",
                        border=PALETTE["ok"],
                        padx=12,
                        pady=4,
                        font_size=8,
                        bold=True,
                    ).pack(side="right", padx=12)

        _render_rows()

        foot = tk.Frame(dlg, bg=PALETTE["bg_deep"])
        foot.pack(fill="x", padx=18, pady=(4, 14))

        def _copy_prompt() -> None:
            dlg.clipboard_clear()
            dlg.clipboard_append(MCP_PROMPT_COPY)
            msg_lbl.configure(text="✓ 已复制推荐并行拆查提示词，可贴入客户端 Rules / System Prompt。", fg=PALETTE["ok"])

        self._make_btn(
            foot,
            "复制推荐检索提示词",
            _copy_prompt,
            bg=PALETTE["sky_bg"],
            fg=PALETTE["sky"],
            hover_bg="#174463",
            border=PALETTE["sky"],
            bold=True,
        ).pack(side="left")
        self._make_btn(foot, "关闭", dlg.destroy).pack(side="right")


class FloatingStatusHud:
    """Compact always-on-top rounded status pill with expandable card & full Memory Workbench."""

    def __init__(
        self,
        memory: Memory,
        *,
        refresh_sec: float = 5.0,
        start_expanded: bool = False,
        start_browser: bool = False,
        topmost: bool = True,
    ) -> None:
        import tkinter as tk

        _enable_windows_dpi_awareness()
        self.tk = tk
        self.memory = memory
        self.refresh_sec = max(2.0, float(refresh_sec))
        self.expanded = bool(start_expanded)
        self.topmost = bool(topmost)
        self.hovered = False
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._workbench: MemoryWorkbenchWindow | None = None
        self._snapshot = HudSnapshot(
            state="ready",
            badge="检测",
            model="检测中…",
            documents=0,
            facts=0,
            revised_facts=0,
            namespaces=0,
            db_size_bytes=0,
        )
        self._pulse_phase = False
        self._drag_x = 0
        self._drag_y = 0
        self._dragged = False
        self._hitboxes: list[tuple[tuple[float, float, float, float], Any]] = []

        self.root = tk.Tk()
        self.root.title("ZeroMem")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", self.topmost)
        try:
            self.root.attributes("-alpha", 0.96)
        except Exception:
            pass

        self._use_transparent = False
        if sys.platform == "win32":
            try:
                self.root.configure(bg=TRANSPARENT_KEY)
                self.root.wm_attributes("-transparentcolor", TRANSPARENT_KEY)
                self._use_transparent = True
            except Exception:
                self.root.configure(bg=PALETTE["bg"])
        else:
            self.root.configure(bg=PALETTE["bg"])

        try:
            dpi = float(self.root.winfo_fpixels("1i"))
            self.scale = max(1.0, min(dpi / 96.0, 2.25))
        except Exception:
            self.scale = 1.0

        bg_canvas = TRANSPARENT_KEY if self._use_transparent else PALETTE["bg"]
        self.canvas = tk.Canvas(
            self.root,
            bg=bg_canvas,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.canvas.pack(fill="both", expand=True)

        self.menu = tk.Menu(self.root, tearoff=0)
        self.menu.add_command(label="打开记忆工作台", command=self.open_workbench)
        self.menu.add_command(label="一键接入主流 Agent", command=self.open_agents)
        self.menu.add_separator()
        self.menu.add_command(label="展开 / 收起小卡", command=self._toggle_expand)
        self.menu.add_command(label="置顶开关", command=self._toggle_topmost)
        self.menu.add_command(label="立即刷新", command=self._trigger_poll_now)
        self.menu.add_separator()
        self.menu.add_command(label="退出", command=self.close)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Double-Button-1>", lambda _e: self.open_workbench())
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Enter>", self._on_enter)
        self.canvas.bind("<Leave>", self._on_leave)

        self._place_initial()
        self._redraw()

        self._worker = threading.Thread(target=self._poll_loop, daemon=True)
        self._worker.start()
        self.root.after(800, self._tick_ui)
        if start_browser:
            self.root.after(150, self.open_workbench)

    def _s(self, px: float) -> int:
        return max(1, int(round(px * self.scale)))

    def _dims(self) -> tuple[int, int]:
        if self.expanded:
            return self._s(308), self._s(194)
        return self._s(268), self._s(40)

    def _place_initial(self) -> None:
        w, h = self._dims()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = max(16, sw - w - self._s(26))
        y = max(16, sh - h - self._s(72))
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _resize_anchor_br(self) -> None:
        self.root.update_idletasks()
        old_x = self.root.winfo_x()
        old_y = self.root.winfo_y()
        old_w = self.root.winfo_width()
        old_h = self.root.winfo_height()
        new_w, new_h = self._dims()
        new_x = max(8, old_x + old_w - new_w)
        new_y = max(8, old_y + old_h - new_h)
        self.root.geometry(f"{new_w}x{new_h}+{new_x}+{new_y}")

    def open_workbench(self) -> None:
        if self._workbench is None or not self._workbench.win.winfo_exists():
            self._workbench = MemoryWorkbenchWindow(self)
        else:
            self._workbench.show()

    def open_agents(self) -> None:
        self.open_workbench()
        if self._workbench is not None:
            self._workbench.open_agents_dialog()

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            snap = collect_status(self.memory)
            with self._lock:
                self._snapshot = snap
            self._stop_event.wait(self.refresh_sec)

    def _trigger_poll_now(self) -> None:
        def _once() -> None:
            snap = collect_status(self.memory)
            with self._lock:
                self._snapshot = snap
        threading.Thread(target=_once, daemon=True).start()

    def _tick_ui(self) -> None:
        if self._stop_event.is_set():
            return
        self._pulse_phase = not self._pulse_phase
        self._redraw()
        if self._workbench is not None and self._workbench.win.winfo_exists():
            with self._lock:
                snap = self._snapshot
            self._workbench.update_header_from_snapshot(snap)
        self.root.after(950, self._tick_ui)

    def _toggle_expand(self) -> None:
        self.expanded = not self.expanded
        self._resize_anchor_br()
        self._redraw()

    def _toggle_topmost(self) -> None:
        self.topmost = not self.topmost
        self.root.attributes("-topmost", self.topmost)
        self._redraw()

    def _on_enter(self, _event: Any) -> None:
        self.hovered = True
        self._redraw()

    def _on_leave(self, _event: Any) -> None:
        self.hovered = False
        self._redraw()

    def _on_press(self, event: Any) -> None:
        self._drag_x = event.x
        self._drag_y = event.y
        self._dragged = False

    def _on_drag(self, event: Any) -> None:
        dx = event.x - self._drag_x
        dy = event.y - self._drag_y
        if abs(dx) > 3 or abs(dy) > 3:
            self._dragged = True
        if self._dragged:
            x = self.root.winfo_x() + dx
            y = self.root.winfo_y() + dy
            self.root.geometry(f"+{x}+{y}")

    def _on_release(self, event: Any) -> None:
        if self._dragged:
            return
        for (x1, y1, x2, y2), cb in self._hitboxes:
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                cb()
                return

    def _on_right_click(self, event: Any) -> None:
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def _dot_colors(self, state: str) -> tuple[str, str]:
        if state == "resident":
            return PALETTE["resident_dot"], PALETTE["resident_ring"]
        if state == "ready":
            return PALETTE["ready_dot"], PALETTE["ready_ring"]
        return PALETTE["offline_dot"], PALETTE["offline_ring"]

    def _redraw(self) -> None:
        with self._lock:
            snap = self._snapshot

        c = self.canvas
        c.delete("all")
        self._hitboxes.clear()

        w, h = self._dims()
        c.config(width=w, height=h)
        pad = self._s(2)
        border = PALETTE["border_hover"] if self.hovered else PALETTE["border"]
        dot_color, ring_color = self._dot_colors(snap.state)
        font_family = _ui_font()
        radius = (h - 2 * pad) / 2.0 if self._use_transparent else 0

        if not self.expanded:
            if self._use_transparent:
                _draw_rounded_rect(
                    c, pad, pad, w - pad, h - pad, r=radius,
                    fill=PALETTE["bg"], outline=border, width=max(1, self._s(1.2)),
                )
            else:
                c.create_rectangle(pad, pad, w - pad, h - pad, fill=PALETTE["bg"], outline=border, width=max(1, self._s(1.2)))

            cy = h / 2.0
            cx = self._s(18)
            ring_r = self._s(7.5 if (snap.state == "resident" and self._pulse_phase) else 6.0)
            dot_r = self._s(3.8)
            c.create_oval(cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r, fill=ring_color, outline="")
            c.create_oval(cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r, fill=dot_color, outline="")

            c.create_text(
                self._s(32), cy, text=f"ZeroMem · {snap.badge}", anchor="w",
                fill=PALETTE["text"], font=(font_family, 9, "bold"),
            )

            count_text = f"{snap.facts} 事实" if snap.state != "offline" else "离线"
            count_color = PALETTE["ok"] if snap.state != "offline" else PALETTE["offline_dot"]
            c.create_text(
                self._s(136), cy, text=count_text, anchor="w",
                fill=count_color, font=(font_family, 9),
            )

            # Quick controls: [◈ Workbench] [▾ Card] [✕ Close]
            wb_x = w - self._s(66)
            ex_x = w - self._s(42)
            cl_x = w - self._s(20)
            c.create_text(wb_x, cy, text="◈", fill=PALETTE["sky"], font=(font_family, 10, "bold"))
            c.create_text(ex_x, cy, text="▾", fill=PALETTE["muted"], font=(font_family, 10))
            c.create_text(cl_x, cy, text="✕", fill=PALETTE["muted"], font=(font_family, 8))
            self._hitboxes.append(((wb_x - self._s(12), 0, wb_x + self._s(12), h), self.open_workbench))
            self._hitboxes.append(((ex_x - self._s(11), 0, ex_x + self._s(11), h), self._toggle_expand))
            self._hitboxes.append(((cl_x - self._s(11), 0, cl_x + self._s(11), h), self.close))
            return

        # Expanded compact status card
        if self._use_transparent:
            _draw_rounded_rect(
                c, pad, pad, w - pad, h - pad, r=self._s(14),
                fill=PALETTE["bg"], outline=border, width=max(1, self._s(1.2)),
            )
        else:
            c.create_rectangle(pad, pad, w - pad, h - pad, fill=PALETTE["bg"], outline=border, width=max(1, self._s(1.2)))

        top_y = self._s(22)
        cx = self._s(18)
        ring_r = self._s(7.5 if (snap.state == "resident" and self._pulse_phase) else 6.0)
        dot_r = self._s(3.8)
        c.create_oval(cx - ring_r, top_y - ring_r, cx + ring_r, top_y + ring_r, fill=ring_color, outline="")
        c.create_oval(cx - dot_r, top_y - dot_r, cx + dot_r, top_y + dot_r, fill=dot_color, outline="")

        c.create_text(
            self._s(32), top_y, text="ZeroMem · 零忆", anchor="w",
            fill=PALETTE["text"], font=(font_family, 10, "bold"),
        )

        pin_x = w - self._s(64)
        col_x = w - self._s(42)
        cls_x = w - self._s(20)
        pin_color = PALETTE["pin_on"] if self.topmost else PALETTE["muted"]
        c.create_text(pin_x, top_y, text="●" if self.topmost else "○", fill=pin_color, font=(font_family, 9))
        c.create_text(col_x, top_y, text="▴", fill=PALETTE["muted"], font=(font_family, 10))
        c.create_text(cls_x, top_y, text="✕", fill=PALETTE["muted"], font=(font_family, 8))
        self._hitboxes.append(((pin_x - self._s(10), self._s(6), pin_x + self._s(10), self._s(36)), self._toggle_topmost))
        self._hitboxes.append(((col_x - self._s(10), self._s(6), col_x + self._s(10), self._s(36)), self._toggle_expand))
        self._hitboxes.append(((cls_x - self._s(10), self._s(6), cls_x + self._s(10), self._s(36)), self.close))

        model_y = self._s(45)
        short_model = snap.model if len(snap.model) <= 28 else snap.model[:27] + "…"
        c.create_text(
            self._s(16), model_y, text=f"{snap.badge} · {short_model}", anchor="w",
            fill=dot_color, font=(font_family, 8),
        )

        # 3 compact metric tiles
        tile_top = self._s(62)
        tile_bot = self._s(114)
        gap = self._s(8)
        margin = self._s(14)
        tile_w = (w - 2 * margin - 2 * gap) / 3.0
        tiles = [
            ("文档", str(snap.documents), PALETTE["text"]),
            ("事实", str(snap.facts), PALETTE["ok"]),
            ("纠错", str(snap.revised_facts), PALETTE["warn"]),
        ]
        for idx, (label, val, val_color) in enumerate(tiles):
            x1 = margin + idx * (tile_w + gap)
            x2 = x1 + tile_w
            _draw_rounded_rect(c, x1, tile_top, x2, tile_bot, r=self._s(9), fill=PALETTE["card"], outline=PALETTE["border"])
            mid_x = (x1 + x2) / 2.0
            c.create_text(mid_x, tile_top + self._s(19), text=val, fill=val_color, font=(font_family, 12, "bold"))
            c.create_text(mid_x, tile_bot - self._s(12), text=label, fill=PALETTE["muted"], font=(font_family, 8))

        # 2 crisp action buttons: Open Workbench | Connect Agents
        btn_top = self._s(124)
        btn_bot = self._s(156)
        btn_w = (w - 2 * margin - gap) / 2.0
        b1_x1, b1_x2 = margin, margin + btn_w
        b2_x1, b2_x2 = b1_x2 + gap, w - margin

        _draw_rounded_rect(c, b1_x1, btn_top, b1_x2, btn_bot, r=self._s(8), fill=PALETTE["ok_bg"], outline=PALETTE["ok"])
        c.create_text((b1_x1 + b1_x2) / 2.0, (btn_top + btn_bot) / 2.0, text="◈ 记忆工作台", fill=PALETTE["ok"], font=(font_family, 9, "bold"))
        self._hitboxes.append(((b1_x1, btn_top, b1_x2, btn_bot), self.open_workbench))

        _draw_rounded_rect(c, b2_x1, btn_top, b2_x2, btn_bot, r=self._s(8), fill=PALETTE["sky_bg"], outline=PALETTE["sky"])
        ag_txt = f"⚡ Agent ({snap.connected_agents}/{snap.installed_agents})"
        c.create_text((b2_x1 + b2_x2) / 2.0, (btn_top + btn_bot) / 2.0, text=ag_txt, fill=PALETTE["sky"], font=(font_family, 9, "bold"))
        self._hitboxes.append(((b2_x1, btn_top, b2_x2, btn_bot), self.open_agents))

        # Footer row
        foot_y = h - self._s(17)
        foot_text = f"{format_bytes(snap.db_size_bytes)} · {snap.namespaces} 空间"
        c.create_text(margin, foot_y, text=foot_text, anchor="w", fill=PALETTE["dim"], font=(font_family, 8))
        ref_x = w - self._s(22)
        c.create_text(ref_x, foot_y, text="↻", fill=PALETTE["muted"], font=(font_family, 10, "bold"))
        self._hitboxes.append(((ref_x - self._s(12), foot_y - self._s(12), ref_x + self._s(12), foot_y + self._s(12)), self._trigger_poll_now))

    def close(self) -> None:
        self._stop_event.set()
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self) -> int:
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.close()
        finally:
            self._stop_event.set()
        return 0


def launch_hud(
    memory: Memory,
    *,
    refresh_sec: float = 5.0,
    start_expanded: bool = False,
    start_browser: bool = False,
    topmost: bool = True,
) -> int:
    """Start the compact floating status pill and visual Memory Workbench."""
    try:
        import tkinter  # noqa: F401
    except ImportError:
        print(
            "[zmem] 当前 Python 环境缺少 tkinter 图形库，无法启动桌面界面。\n"
            "  - Ubuntu/Debian: sudo apt install python3-tk\n"
            "  - Fedora/RHEL:   sudo dnf install python3-tkinter\n"
            "  - macOS (Brew):  brew install python-tk\n"
            "  命令行功能（zmem add / search / mcp / connect）不受影响。",
            file=sys.stderr,
        )
        return 1

    hud = FloatingStatusHud(
        memory,
        refresh_sec=refresh_sec,
        start_expanded=start_expanded,
        start_browser=start_browser,
        topmost=topmost,
    )
    return hud.run()


def main_gui(argv: list[str] | None = None) -> int:
    """GUI script entry point (`zmem-hud` / `v2mem-hud`) launching the desktop pill & workbench."""
    import argparse

    parser = argparse.ArgumentParser(prog="zmem-hud", description="ZeroMem (零忆) floating pill & desktop workbench")
    parser.add_argument("--db", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--refresh", type=float, default=5.0)
    parser.add_argument("--expanded", action="store_true")
    parser.add_argument("--browser", action="store_true")
    parser.add_argument("--no-topmost", action="store_true")
    args = parser.parse_args(argv)
    mem = Memory(db_path=args.db, model=args.model, ollama_url=args.ollama_url)
    return launch_hud(
        mem,
        refresh_sec=args.refresh,
        start_expanded=args.expanded,
        start_browser=args.browser,
        topmost=not args.no_topmost,
    )
