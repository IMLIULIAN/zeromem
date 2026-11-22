<p align="right">
  <a href="README.md">简体中文</a> | <strong>English</strong>
</p>

# ZeroMem (零忆)

> **by [ZeroZoneX](https://zerozonex.com) · Four-Zero Local Memory Engine Tailored for Open-Source [`Qwen3.5-4B-Memory-V2`](https://huggingface.co/IMLIULIAN/qwen3.5-4b-memory-v2)**  
> Created by **B站IM榴莲 (IMLIULIAN)** · Docs: `https://mem.zerozonex.com` · Apache-2.0 (backward-compatible with `v2-memory` / `v2mem` CLI & imports)

**Four-Zero Architecture**: **Zero** third-party Python dependencies (pure Python 3.9+ stdlib, including the MCP stdio server), **Zero** vector DB or embedding model (single SQLite file with dual-track FTS5), **Zero** cloud write API cost (local 4B extraction + pre-write admission & revision), and **Zero** cloud plaintext secret leaks (dual-layer local redaction to `[REDACTED]`).

---

## Live 52k-Char Benchmark (DeepSeek API: A / B / C / D 4-Group Comparison)

Evaluated on **71 engineering documents (51,947 chars / 385 extracted facts) × 12 multi-dimensional objective questions** against the live cloud DeepSeek API (`deepseek-flash`). All costs are computed directly from returned API usage logs (official pricing: cache miss `¥2/M`, cache hit `¥0.2/M`, output `¥3/M`; local V2 write cost `¥0`):

| Metric | Group A: Bare API (No Memory) | Group B: Full-Context Stuffing | **Plan 1 (Group C): ZeroMem Single-Recall** | **Plan 2 (Group D): ZeroMem MCP Multi-Query** |
| :--- | :---: | :---: | :---: | :---: |
| **Objective Accuracy** | 2/12 (16.7%) | 12/12 (100.0%) | **11/12 (91.7%)** | **12/12 (100.0%) Perfect** |
| **Avg Input Tokens / Q** | 1,629 | 31,720 | **1,821 (-94.3%)** | **4,296 (-86.5%)** |
| **12-Q Cold-Start Cost (No Cache)** | ¥0.053 | ¥0.770 | **¥0.054 (-93.0% / 14.3× less)** | **¥0.120 (-84.4% / 6.4× less)** |
| **12-Q Cached Cost (90.7% Hit on B)** | ¥0.026 | ¥0.149 | **¥0.033 (-78.0% / 4.5× less)** | **¥0.078 (-47.7% / 1.9× less)** |
| **Avg Latency / Q** | 9.27s | 7.20s | **6.46s (Fastest)** | 9.84s (avg 2.17 tool calls) |
| **Cloud Plaintext Secret Leaks** | 0 | **24×** ⚠️ | **0×** ✅ | **0×** ✅ |
| **Bound Evidence Traceability** | 0/12 | 0/12 | **12/12** | **12/12** |

![ZeroMem 52k-Char Live 4-Group Benchmark](docs/images/zeromem_abcd_50k_en.png)

> **Scaling Scissors Gap (15k → 52k Chars)**: When the corpus expands 3.5×, full-context stuffing (Group B) surges **3.38×** in prompt tokens (`9,394 → 31,720`), whereas ZeroMem Plan 1 stays virtually flat (`1,650 → 1,821`, only +10.4%) with zero accuracy decay.

![ZeroMem 15k vs 52k Corpus Scaling Benchmark](docs/images/zeromem_scaling_15k_vs_50k_en.png)

---

## Native Dark Desktop Memory Workbench (Zero Web Server, Zero Dependencies)

Run `zmem-hud` (or `zmem ui --browser`) to launch the **floating status pill + modern dark desktop workbench** built purely on Python's built-in `tkinter`:
- **Sub-millisecond `<1ms` live FTS5 search** with namespace filtering and clear `Verified` / `Revised` admission badges;
- **Bound original evidence panel** with automatic green-pill highlighting for locally redacted `[REDACTED]` credentials;
- **GUI remember & two-step delete** plus **one-click connect/disconnect for 7 mainstream AI Agents** (automatic `.bak` backups before every edit).

![ZeroMem Native Desktop Memory Workbench](docs/images/zeromem_workbench.png)

---

## Two Installation Plans: True One-Command Setup

Prerequisites: **Python 3.9+** and **Ollama** (pass `-AutoDeps` / `--auto-deps` to auto-install missing dependencies and silently start `ollama serve`; hardware/VRAM/unified memory is probed automatically to pick `Q8_0` vs `Q4_K_M` and safe `num_ctx`).

| Comparison | Plan 1: Ultra-Fast & Cost-Saving (`--mode fast`) | Plan 2: Full-Score Agent MCP (`--mode mcp`, Default) |
| :--- | :--- | :--- |
| **Integration Mode** | CLI / 5-line Python SDK single-recall injection | Zero-dep 4-tool MCP stdio server + 7 mainstream Agents + parallel split-query prompt |
| **Best For** | High-frequency single-topic Q&A, API proxies, scripts | Claude Desktop / Claude Code / Cursor / Windsurf / Codex CLI / Antigravity / DSH |
| **Live Benchmark** | **-94.3% tokens**, **6.46s fastest**, 91.7% accuracy, 0 leaks | **12/12 (100%) accuracy**, solves cross-doc queries, still **-86.5% tokens**, 0 leaks |
| **Auto-Generated Artifacts** | `~/.zeromem/configs/fast_sdk_example.py` | Auto-backed-up client configs + `MCP_SYSTEM_PROMPT.md` |

### Plan 2: Full-Score Agent MCP (Recommended — Auto Hardware Quant + Auto Pull + Auto Connect)

**Windows (PowerShell)**:

```powershell
# From local repo (auto deps + auto VRAM quant + connect all detected Agents + launch HUD)
.\install.ps1 -Mode mcp -AutoDeps -ConnectAll -LaunchHud

# Or remote one-liner (add $env:ZEROMEM_SOURCE="hf" outside mainland China for direct Hugging Face pull):
$env:ZEROMEM_MODE="mcp"; $env:ZEROMEM_SOURCE="hf"; $env:ZEROMEM_AUTO_DEPS="1"; $env:ZEROMEM_CONNECT_ALL="1"
irm https://raw.githubusercontent.com/IMLIULIAN/zeromem/main/install.ps1 | iex
```

**macOS / Linux**:

```bash
# From local repo (auto deps + Apple Silicon / GPU detection + connect all detected Agents)
./install.sh --mode mcp --source hf --auto-deps --connect-all

# Or remote one-liner:
ZEROMEM_MODE=mcp ZEROMEM_SOURCE=hf ZEROMEM_AUTO_DEPS=1 ZEROMEM_CONNECT_ALL=1 curl -fsSL https://raw.githubusercontent.com/IMLIULIAN/zeromem/main/install.sh | bash
```

### Plan 1: Ultra-Fast & Cost-Saving (CLI / Python SDK Single-Recall)

**Windows (PowerShell)**:

```powershell
.\install.ps1 -Mode fast -Source hf -AutoDeps

# Or remote one-liner:
$env:ZEROMEM_MODE="fast"; $env:ZEROMEM_SOURCE="hf"; $env:ZEROMEM_AUTO_DEPS="1"
irm https://raw.githubusercontent.com/IMLIULIAN/zeromem/main/install.ps1 | iex
```

**macOS / Linux**:

```bash
./install.sh --mode fast --source hf --auto-deps

# Or remote one-liner:
ZEROMEM_MODE=fast ZEROMEM_SOURCE=hf ZEROMEM_AUTO_DEPS=1 curl -fsSL https://raw.githubusercontent.com/IMLIULIAN/zeromem/main/install.sh | bash
```

### Alternative: `pip install` (Works for Both Plans)

```bash
pip install git+https://github.com/IMLIULIAN/zeromem.git
zmem setup --mode mcp --source hf --connect-all   # Plan 2: Full-score Agent MCP
zmem setup --mode fast --source hf                # Plan 1: Ultra-fast & cost-saving
```

---

## One-Click Integration for 7 Mainstream AI Agents (Backed Up & Reversible)

ZeroMem's MCP stdio server supports both `NDJSON` and `Content-Length` framing, exposes 4 tools (**`memory_remember` / `memory_recall` / `memory_stats` / `memory_delete`**), and embeds the 12/12 parallel split-query retrieval instructions directly in MCP `initialize` and tool schemas:

| AI Agent Client | Connect Command | Managed Config File |
| :--- | :--- | :--- |
| **Claude Desktop** | `zmem connect claude-desktop` | `claude_desktop_config.json` (JSON) |
| **Claude Code** | `zmem connect claude-code` | `~/.claude.json` (JSON) |
| **Cursor** | `zmem connect cursor` | `~/.cursor/mcp.json` (JSON) |
| **Windsurf** | `zmem connect windsurf` | `~/.codeium/windsurf/mcp_config.json` (JSON) |
| **Codex CLI** | `zmem connect codex` | `~/.codex/config.toml` (fenced TOML block) |
| **Antigravity** | `zmem connect antigravity` | `~/.gemini/antigravity/mcp_config.json` (JSON) |
| **DeepSeek Harness (DSH)** | `zmem connect dsh` | `~/.zeromem/configs/dsh_mcp_patch.yml` (YAML patch) |

```bash
zmem doctor                     # Inspect hardware/VRAM, Ollama status, DB stats & 7 Agents
zmem connect --list             # Detect installed AI Agents and connection status
zmem connect --all              # Auto-backup (.bak) and connect all detected Agents
zmem connect --project-rules .  # Export .cursor/rules/zeromem.mdc & AGENTS.md retrieval rules
zmem disconnect --all           # Cleanly remove ZeroMem without touching other MCP servers
```

---

## Open-Source Model Weights

| Platform | Repository / Command | Notes |
| :--- | :--- | :--- |
| **Hugging Face** | https://huggingface.co/IMLIULIAN/qwen3.5-4b-memory-v2 | **Global Recommended** (`--source hf`, `ollama pull hf.co/IMLIULIAN/qwen3.5-4b-memory-v2:Q8_0`) |
| **ModelScope** | https://modelscope.cn/models/IMLIULIAN/qwen3.5-4b-memory-v2 | **Mainland China Mirror** (`--source modelscope`, auto SHA-256 verification & Ollama registration) |
| **Offline Local GGUF** | `zmem setup --gguf ./qwen3.5-4b-memory-v2-Q8_0.gguf` | Air-gapped / intranet registration |

> **Quantization Guide**: GPUs with `>= 7.5 GB` VRAM (or Apple Silicon `>= 16 GB` unified memory) automatically pick **`Q8_0` (4.48 GB)** for peak pre-write admission accuracy; 4–7 GB GPUs, 8 GB Macs, and CPU-only hosts automatically pick **`Q4_K_M` (2.71 GB, `--quant q4_k_m`)** to prevent OOM fallback.

---

## Quick Start

### 1. Plan 1: CLI & Python SDK (3 Lines of Code)

```bash
# Remember a note or Markdown file (local 4B extract + pre-write admission + credential redaction)
zmem add "2026-03-01 Note: API gateway rate limit increases to 120 req/min on 2026-03-15." --ns demo
zmem add -f ./notes.md --ns my-project

# Sub-millisecond recall with bound original evidence segments
zmem search "gateway rate limit" --ns demo

# Launch desktop status pill & workbench (windowless)
zmem-hud

# View database statistics
zmem stats
```

```python
from zeromem import Memory  # Also compatible with: from v2mem import Memory

mem = Memory()  # Auto-detects local V2 model & ~/.zeromem/memory.db (migrates ~/.v2-memory seamlessly)
mem.remember("2026-11-20 Team decided to start with single-file SQLite WAL instead of Redis.", ns="arch")
for hit in mem.recall("why not Redis", ns="arch"):
    print(hit.text, "<- Evidence:", hit.evidence)
```

### 2. Plan 2: Manual MCP JSON Snippet (Claude Desktop / Cursor / Windsurf / Antigravity)

Running `zmem config --mode mcp` exports ready-to-use files with absolute paths into `~/.zeromem/configs/` (`mcp_config.json`, `dsh_mcp_patch.yml`, `MCP_SYSTEM_PROMPT.md`):

```json
{
  "mcpServers": {
    "zeromem": {
      "command": "python",
      "args": ["-m", "zeromem", "mcp"]
    }
  }
}
```

Environment variable overrides: `ZEROMEM_DB`, `ZEROMEM_MODEL`, `ZEROMEM_OLLAMA_URL`, `ZEROMEM_NUM_CTX` (fully backward-compatible with `V2MEM_*`).

---

## Why Purpose-Built for Qwen3.5-4B-Memory-V2?

Dropping a specialized 4B memory model into a bloated generic framework causes long-context recall collapse, repetition loops, and wastes V2's signature pre-write revision capability. ZeroMem focuses on **5 razor-sharp design pillars**:

1. **Short-Window Micro-Chunking (`< 1,500` chars + `s0..sN` evidence IDs)**: Keeps every input inside V2's peak recall sweet spot with 1-segment overlap.
2. **Native Dual-Contract Single-Round Loop (`Extract → Admit`)**: Strictly aligns with V2's trained `fact-extraction-v1` and `fact-admission-v1` contracts; `revise` replacements are accepted in one pass (zero deadlock), while `drop`/`hold` isolates bad items individually.
3. **0ms Deterministic Guardrails**: Validates evidence IDs, auto-repairs truncated 64-char SHA-256 hashes from bound source text, separates record date (`mentioned_at`) from event date (`occurred_start`), and double-redacts secrets to `[REDACTED]`.
4. **Anti-OTL Sampling & Truncation Salvage**: Disables thinking, pins `temperature=0`, `presence_penalty=0`, `repeat_penalty=1.08`, bounded `num_predict`, and salvages completed facts if output JSON is truncated.
5. **Dual-Track FTS5 (Facts + Bound Evidence)**: Indexes concise facts, entities, and bound original evidence segments with CJK 2-gram + alphanumeric tokenization — enabling cross-lingual Chinese/English search with **zero embedding model** and 100% source traceability.

---

## Project Layout

```text
zeromem/
├── install.ps1             # Windows one-command installer (-Mode fast | mcp)
├── install.sh              # macOS / Linux one-command installer (--mode fast | mcp)
├── pyproject.toml          # Zero-dependency package config (zmem / zmem-hud + v2mem aliases)
├── README.md               # Chinese README (简体中文)
├── README_EN.md            # English README
├── zeromem/                # Public package alias & module entry (python -m zeromem)
├── v2mem/                  # Core engine implementation (backward-compatible)
│   ├── contracts.py        # Native V2 dual contracts, micro-chunker, guardrails & salvage parser
│   ├── hardware.py         # Pure-stdlib OS / GPU VRAM / Apple Silicon probe & quant selector
│   ├── connectors.py       # Auto-discovery, backup & reversible connect/disconnect for 7 Agents
│   ├── ollama.py           # Stdlib Ollama client, anti-OTL options & SHA-256 weight bootstrap
│   ├── store.py            # Single-file SQLite WAL + dual-track FTS5 store
│   ├── memory.py           # Core remember() / recall() pipeline
│   ├── mcp_server.py       # Zero-dep 4-tool MCP stdio server (NDJSON + Content-Length)
│   ├── hud.py              # Native dark tkinter floating status pill & memory workbench
│   └── cli.py              # setup / connect / disconnect / doctor / config / add / search / stats / mcp / ui
├── docs/images/            # Live 52k benchmark charts (ZH/EN) & native workbench screenshots
├── modelfiles/             # Official Q8_0 / Q4_K_M Ollama Modelfiles
├── demo.py                 # Synthetic isolated end-to-end demo
└── tests/                  # Offline unit tests + live V2 smoke test + cloud E2E suite
```

---

## Testing

```bash
# Offline zero-dependency unit tests (no network / no Ollama required)
python -m unittest discover -s tests -p "test_core.py" -v

# Live end-to-end smoke test against a local Ollama V2 instance
python -m unittest discover -s tests -p "test_live_v2.py" -v
```

---

## License & Attribution

- **License**: **Apache License 2.0** (see [LICENSE](LICENSE) and [NOTICE](NOTICE))
- **Author**: **B站IM榴莲 (IMLIULIAN)** · **ZeroZoneX** ([zerozonex.com](https://zerozonex.com) / [mem.zerozonex.com](https://mem.zerozonex.com))
- **Companion Model**: [`Qwen3.5-4B-Memory-V2`](https://huggingface.co/IMLIULIAN/qwen3.5-4b-memory-v2) (derived from `Qwen/Qwen3.5-4B`)
