<p align="right">
  <a href="README.md">简体中文</a> | <strong>English</strong>
</p>

# ZeroMem (零忆)

> **by ZeroZoneX · A Lightweight Local Memory Engine for [`Qwen3.5-4B-Memory-V2`](https://huggingface.co/IMLIULIAN/qwen3.5-4b-memory-v2)**  
> Created by **B站IM榴莲 (IMLIULIAN)** · Apache-2.0 (backward-compatible with `v2-memory` / `v2mem` CLI & imports)

ZeroMem is built around four simple engineering constraints to keep local setup minimal: **no third-party Python dependencies** (Python 3.9+ stdlib only, including the MCP stdio server), **no external vector DB or embedding model** (single SQLite file with dual-track FTS5), **no cloud API cost on write** (local 4B extraction and pre-write admission), and **local credential redaction** before persistence (`[REDACTED]`).

---

## Controlled 52k-Char Benchmark (DeepSeek API: Groups A / B / C / D)

Evaluated on **71 synthetic engineering documents (51,947 chars / 385 extracted facts) × 12 objective questions** against the cloud DeepSeek API (`deepseek-flash`). Cost estimates are calculated directly from returned API usage logs (official pricing: cache miss `¥2/M`, cache hit `¥0.2/M`, output `¥3/M`; local V2 write cloud cost `¥0`):

| Metric | Group A: Bare API (No Memory) | Group B: Full-Context Stuffing | **Plan 1 (Group C): Single-Recall Mode** | **Plan 2 (Group D): Agent MCP Mode** |
| :--- | :---: | :---: | :---: | :---: |
| **Objective Accuracy** | 2/12 (16.7%) | 12/12 (100.0%) | **11/12 (91.7%)** | **12/12 (100.0%)** |
| **Avg Input Tokens / Q** | 1,629 | 31,720 | **1,821 (-94.3%)** | **4,296 (-86.5%)** |
| **12-Q Cold-Start Cost (No Cache)** | ¥0.053 | ¥0.770 | **¥0.054 (-93.0%)** | **¥0.120 (-84.4%)** |
| **12-Q Cached Cost (90.7% Hit on B)** | ¥0.026 | ¥0.149 | **¥0.033 (-78.0%)** | **¥0.078 (-47.7%)** |
| **Avg Latency / Q** | 9.27s | 7.20s | **6.46s** | 9.84s (avg 2.17 tool calls) |
| **Cloud Plaintext Secret Exposure** | 0 | **24×** ⚠️ | **0×** ✅ | **0×** ✅ |
| **Bound Evidence Traceability** | 0/12 | 0/12 | **12/12** | **12/12** |

> **Scope Note**: This is a small, controlled benchmark (71 synthetic engineering notes, 12 objective questions) intended to illustrate relative trade-offs in context size, estimated cost, and credential redaction—not a universal leaderboard. Real-world results will vary with document structure, language mix, and query complexity.

![ZeroMem 52k-Char Live 4-Group Benchmark](docs/images/zeromem_abcd_50k_en.png)

> **Scaling Behavior (15k → 52k Chars)**: When the corpus expands 3.5×, full-context stuffing (Group B) grows roughly **3.38×** in input tokens (`9,394 → 31,720`), whereas single-recall injection (Plan 1) stays relatively flat (`1,650 → 1,821`, +10.4%) while maintaining 11/12 on this test set.

![ZeroMem 15k vs 52k Corpus Scaling Benchmark](docs/images/zeromem_scaling_15k_vs_50k_en.png)

### 52k-Char Ablation Ladder & v0.2.1 Engine Fixes (`L0 -> L3 -> L3-FIX`)

On the same 71-document corpus (51,947 chars) × 12 questions, identical `tokenize_for_fts` FTS5 engine (fixed Top-6), and live `deepseek-flash`, we ran a single-variable ablation ladder from raw-segment FTS5 (`L0`) through 4B extraction (`L1`), 1-round native admission (`L2`), v0.2.0 in-admission `prior_facts` (`L3`), and **v0.2.1 upgraded ZeroMem (`L3-FIX`: pure native 3-field admission + gated standalone supersession + no-op `revise` guard + recall evidence dedup/folding)**, plus a full-database dual-layer audit (deterministic rules ∪ independent cloud judge across all 560 unique stored facts):

| Ablation Tier | Top-1 Hit (10 Answerable Qs) | Accuracy (12 Qs) | Rule Defect Rate | Combined Fact Error (Rule ∪ Judge) | Avg Context Chars | Avg Input Tokens (11 Single-Step / 12-Q) | Plaintext Secrets (DB / Cloud) | Local Write Time (LLM Calls) | Avg Search |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **L0: Raw Segment FTS5 (0 LLM)** | 10/10 (100%) | 11/12 (91.7%) | N/A | N/A | 797.7 | 1,156.9 / 1,149.9 (-96.4% vs B) | DB 2 / Cloud 8 ⚠️ | 0.015s (0) | 3.02ms |
| **L1: +4B Extract (No Admit/Guard)** | 10/10 (100%) | 11/12 (91.7%) | 1.34% (5/373) | 9.65% (36/373) | 2,635.7 | 1,886.0 / 1,925.3 (-93.9% vs B) | DB 4 / Cloud 8 ⚠️ | 558.19s (71) | 6.47ms |
| **L2: +1-Round Native Admit** | 9/10 (90%) | 11/12 (91.7%) | 0.54% (2/370) | 7.03% (26/370) | 2,275.7 | 1,722.6 / 1,747.3 (-94.5% vs B) | DB 4 / Cloud 8 ⚠️ | 692.39s (142) | 4.99ms |
| **L3: ZeroMem v0.2.0 (Before Fix)** | 9/10 (90%) | 11/12 (91.7%) | 1.07% (4/373) | 8.31% (31/373) | 2,313.4 | 1,742.1 / 1,777.9 (-94.4% vs B) | **DB 0 / Cloud 0** ✅ | 751.59s (142) | 4.63ms |
| **L3-FIX: ZeroMem v0.2.1 (Upgraded)** | **10/10 (100%)** | **11/12 (91.7%)** | **0.00% (0/368)** | **6.79% (25/368)** | **1,919.6 (-17.0%)** | **1,499.5 (-13.9%) / 1,720.2 (-94.6% vs B)** | **DB 0 / Cloud 0** ✅ | 853.09s (147, gated 5) | **3.73ms** |
| **L3-FIX-FO: Facts-Only (Same DB)** | **10/10 (100%)** | **11/12 (91.7%)** | **0.00% (0/368)** | **6.79% (25/368)** | 1,407.1 | 1,233.3 / 1,349.3 (-95.7% vs B) | **DB 0 / Cloud 0** ✅ | Reuses L3-FIX | 4.03ms |

> **Verified Takeaways & Honest Boundaries**:
> - **v0.2.1 Engine Fixes Verified**: Restoring `_admit_window` to the pure native 3-field contract, holding no-op `revise` echoes, and isolating supersession into a gated standalone pass (triggered on only 5/71 docs) drove deterministic rule defects to **0.00%**, eliminated prior template leaks (`4/6 -> 0/0` across rule & judge), cut full-DB combined error rate to **6.79%** (`-29.6%` vs L1, surpassing L2), reduced supersession false positives by `-83.3%` while linking the core architectural reversal (`sig_05 -> sig_02`, lifting Top-1 hit rate from `9/10` to **10/10**). Recall evidence deduplication & verbatim folding shrank context characters on **12/12** questions (`-17.0%` avg) and cut single-step cloud input tokens by `-13.9%` (note: both 11 single-step and 12-Q means are reported because `deepseek-flash` performed a spontaneous 2-step verification on unanswerable `Q10`).
> - **Honest Positioning**: For static single-topic FTS5 search alone, raw-segment `L0` already scores 11/12 at `1,150 tok/Q`. ZeroMem's real edge lies in **¥0 cloud write cost, zero cloud secret leaks (`8 -> 0`), structured event-date grounding, timeline supersession ranking, and autonomous MCP cross-doc decomposition (`12/12`)**. Dual-track tokens still sit slightly above `L0` mainly because `Qwen3.5-4B-Memory-V2` often expands Chinese notes into longer English facts (~`2.35×` char expansion), which is the primary target for the next-generation 4B SFT run.

![ZeroMem 52k-Char Ablation Ladder & v0.2.1 Fixes](docs/images/zeromem_ablation_50k_en.png)

---

## Native Dark Desktop Workbench (No Web Server, Zero Dependencies)

Run `zmem-hud` (or `zmem ui --browser`) to open a **floating status pill and dark desktop workbench** built solely on Python's standard `tkinter`:
- **Local sub-millisecond (`<1ms`) FTS5 search** with namespace filtering and `Verified` / `Revised` admission badges;
- **Bound original evidence viewer** with highlighting for locally redacted `[REDACTED]` credentials so stored facts stay inspectable;
- **In-app remember & two-step delete**, plus **one-click connect/disconnect for 7 common AI Agents** (creates `.bak` backups before modifying any config).

![ZeroMem Native Desktop Memory Workbench](docs/images/zeromem_workbench.png)

---

## Two Setup Modes: Pick What Fits Your Workflow

Prerequisites: **Python 3.9+** and **Ollama** (pass `-AutoDeps` / `--auto-deps` to install missing dependencies and start `ollama serve` in the background; local RAM/VRAM/unified memory is detected automatically to choose `Q8_0` vs `Q4_K_M` and a safe `num_ctx`).

| Comparison | Plan 1: Single-Recall Mode (`--mode fast`) | Plan 2: Agent MCP Mode (`--mode mcp`, Default) |
| :--- | :--- | :--- |
| **Integration Style** | CLI / 5-line Python SDK single-recall injection | Zero-dep 4-tool MCP stdio server + 7 AI Agent connectors + split-query prompt |
| **Best Fit** | Single-topic Q&A, API proxies, lightweight scripts | Claude Desktop / Claude Code / Cursor / Windsurf / Codex CLI / Antigravity / DSH |
| **Benchmark Reference** | **-94.3% tokens**, **6.46s avg**, 11/12, 0 secret leaks | **12/12** (handles multi-doc queries), **-86.5% tokens**, 0 secret leaks |
| **Generated Files** | `~/.zeromem/configs/fast_sdk_example.py` | Backed-up client configs + `MCP_SYSTEM_PROMPT.md` |

### Plan 2: Agent MCP Mode (Default — Hardware Detection + Model Pull + Agent Setup)

**Windows (PowerShell)**:

```powershell
# From local clone (optional: -AutoDeps, -ConnectAll, -LaunchHud)
.\install.ps1 -Mode mcp -AutoDeps -ConnectAll -LaunchHud

# Or remote installer (use ZEROMEM_SOURCE=hf outside mainland China for direct Hugging Face pull):
$env:ZEROMEM_MODE="mcp"; $env:ZEROMEM_SOURCE="hf"; $env:ZEROMEM_AUTO_DEPS="1"; $env:ZEROMEM_CONNECT_ALL="1"
irm https://raw.githubusercontent.com/IMLIULIAN/zeromem/main/install.ps1 | iex
```

**macOS / Linux**:

```bash
# From local clone
./install.sh --mode mcp --source hf --auto-deps --connect-all

# Or remote installer:
ZEROMEM_MODE=mcp ZEROMEM_SOURCE=hf ZEROMEM_AUTO_DEPS=1 ZEROMEM_CONNECT_ALL=1 curl -fsSL https://raw.githubusercontent.com/IMLIULIAN/zeromem/main/install.sh | bash
```

### Plan 1: Single-Recall Mode (Low Cost CLI / Python SDK)

**Windows (PowerShell)**:

```powershell
.\install.ps1 -Mode fast -Source hf -AutoDeps

# Or remote installer:
$env:ZEROMEM_MODE="fast"; $env:ZEROMEM_SOURCE="hf"; $env:ZEROMEM_AUTO_DEPS="1"
irm https://raw.githubusercontent.com/IMLIULIAN/zeromem/main/install.ps1 | iex
```

**macOS / Linux**:

```bash
./install.sh --mode fast --source hf --auto-deps

# Or remote installer:
ZEROMEM_MODE=fast ZEROMEM_SOURCE=hf ZEROMEM_AUTO_DEPS=1 curl -fsSL https://raw.githubusercontent.com/IMLIULIAN/zeromem/main/install.sh | bash
```

### Alternative: `pip install`

```bash
pip install git+https://github.com/IMLIULIAN/zeromem.git
zmem setup --mode mcp --source hf --connect-all   # Plan 2: Agent MCP Mode
zmem setup --mode fast --source hf                # Plan 1: Single-Recall Mode
```

---

## One-Click Setup for 7 Mainstream AI Agents (Backed Up & Reversible)

ZeroMem's MCP stdio server supports both `NDJSON` and `Content-Length` framing, exposes 4 tools (**`memory_remember` / `memory_recall` / `memory_stats` / `memory_delete`**), and includes concise parallel split-query guidance in MCP `initialize` and tool descriptions:

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
zmem doctor                     # Check hardware/VRAM, Ollama status, DB stats & 7 Agents
zmem connect --list             # Detect installed AI Agents and connection status
zmem connect --all              # Create .bak backups and connect detected Agents
zmem connect --project-rules .  # Export .cursor/rules/zeromem.mdc & AGENTS.md retrieval rules
zmem disconnect --all           # Cleanly remove ZeroMem without touching other MCP entries
```

---

## Companion Open-Source Model Weights

| Platform | Repository / Command | Notes |
| :--- | :--- | :--- |
| **Hugging Face** | https://huggingface.co/IMLIULIAN/qwen3.5-4b-memory-v2 | **Global Recommended** (`--source hf`, `ollama pull hf.co/IMLIULIAN/qwen3.5-4b-memory-v2:Q8_0`) |
| **ModelScope** | https://modelscope.cn/models/IMLIULIAN/qwen3.5-4b-memory-v2 | **Mainland China Mirror** (`--source modelscope`, auto SHA-256 verification & Ollama registration) |
| **Offline Local GGUF** | `zmem setup --gguf ./qwen3.5-4b-memory-v2-Q8_0.gguf` | Air-gapped / intranet registration |

> **Quantization Guide**: GPUs with `>= 7.5 GB` VRAM (or Apple Silicon with `>= 16 GB` unified memory) default to **`Q8_0` (4.48 GB)** for more reliable pre-write revision; smaller GPUs, 8 GB Macs, and CPU-only hosts default to **`Q4_K_M` (2.71 GB, or `--quant q4_k_m`)** to avoid memory fallback.

---

## Quick Start

### 1. Plan 1: CLI & Python SDK

```bash
# Remember a note or Markdown file (local 4B extract + pre-write admission + credential redaction)
zmem add "2026-03-01 Note: API gateway rate limit increases to 120 req/min on 2026-03-15." --ns demo
zmem add -f ./notes.md --ns my-project

# Local search with bound original evidence segments (supports event-time range filtering & supersession timeline)
zmem search "gateway rate limit" --ns demo
zmem search "Project A" --ns demo --occurred-from 2025-01-01 --occurred-to 2025-12-31

# Read-only Markdown export with bound citations (Obsidian / Git friendly)
zmem export --md -o ./zeromem-export.md
zmem export --md --dir ./obsidian-zeromem

# Safe single-file snapshot via SQLite Online Backup API (no -wal/-shm sidecars, cloud-drive safe)
zmem backup -o ./zeromem-snapshot.db --sha256

# Launch desktop status pill & workbench
zmem-hud

# View database statistics
zmem stats
```

```python
from zeromem import Memory  # Also compatible with: from v2mem import Memory

mem = Memory()  # Auto-detects local V2 model & ~/.zeromem/memory.db (migrates ~/.v2-memory seamlessly)
mem.remember("2026-11-20 Team decided to start with single-file SQLite WAL instead of Redis.", ns="arch")
for hit in mem.recall("why not Redis", ns="arch", occurred_from="2026-01-01"):
    print(hit.text, "<- Evidence:", hit.evidence)

mem.export_markdown(output_dir="./obsidian-zeromem")
mem.backup("./zeromem-snapshot.db", write_sha256=True)
```

### 2. Plan 2: Manual MCP JSON Snippet

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

Environment variable overrides: `ZEROMEM_DB`, `ZEROMEM_MODEL`, `ZEROMEM_OLLAMA_URL`, `ZEROMEM_NUM_CTX` (compatible with `V2MEM_*`).

---

## Why a Dedicated Lightweight Harness for Qwen3.5-4B-Memory-V2?

General-purpose memory frameworks are typically built around larger cloud models and vector databases, which does not always align well with a specialized 4B model's context limits and output schema. ZeroMem keeps the pipeline small with **5 practical design choices** tailored to V2:

1. **Short-Window Micro-Chunking (`< 1,500` chars + `s0..sN` evidence IDs)**: Splits long inputs into small overlapping windows to reduce missed facts on a 4B model.
2. **Pure Native Dual-Contract + Gated Standalone Supersession (`Extract -> Native Admit -> Gated Supersession`)**: Keeps `fact-extraction-v1` and 3-field `fact-admission-v1` 100% pure (zero OOD schema pollution), routes empty/unchanged no-op `revise` echoes into `hold`, and invokes a lightweight standalone supersession check (`num_predict=128`) only on admitted facts with explicit state-change/migration cues that pass boilerplate-echo filtering.
3. **Deterministic Guardrails**: Validates evidence IDs, repairs truncated 64-char SHA-256 hashes against bound source text, separates record date (`mentioned_at`) from event date (`occurred_start`, with conservative single-date backfill), and redacts common secret patterns to `[REDACTED]`.
4. **Conservative Sampling & Truncation Salvage**: Disables thinking, fixes `temperature=0`, `presence_penalty=0`, `repeat_penalty=1.08`, bounded `num_predict`, and salvages completed facts if output JSON is truncated.
5. **Dual-Track FTS5 + Evidence Dedup & Folding**: Indexes extracted facts, entities, and bound source segments with CJK 2-gram + alphanumeric tokenization; keeps underlying `RecallHit.evidence` lossless while deduplicating shared `(doc_id, seg_id)` segments and folding verbatim-contained segments into compact citations in SDK (`format_recall_context`), MCP, and CLI rendering.

---

## Project Layout

```text
zeromem/
├── install.ps1             # Windows installer (-Mode fast | mcp)
├── install.sh              # macOS / Linux installer (--mode fast | mcp)
├── pyproject.toml          # Zero-dependency package config (zmem / zmem-hud + v2mem aliases)
├── README.md               # Chinese README (简体中文)
├── README_EN.md            # English README
├── CHANGELOG.md            # Release changelog
├── zeromem/                # Public package alias & module entry (python -m zeromem)
├── v2mem/                  # Core engine implementation (backward-compatible)
│   ├── contracts.py        # Pure native dual contracts, gated supersession contract, no-op revise guard & salvage parser
│   ├── hardware.py         # Pure-stdlib OS / GPU VRAM / Apple Silicon probe & quant selector
│   ├── connectors.py       # Auto-discovery, backup & reversible connect/disconnect for 7 Agents
│   ├── ollama.py           # Stdlib Ollama client, sampling options & SHA-256 weight bootstrap
│   ├── store.py            # Single-file SQLite WAL + dual-track FTS5 + fact_supersessions timeline edges
│   ├── memory.py           # remember() (pure admission + gated supersession) / recall() (time-range + evidence dedup/folding)
│   ├── expiry.py           # Strict until:YYYY-MM-DD expiry demotion, adaptive 2-8 gating & occurred range helpers
│   ├── governance.py       # Fail-closed retraction ledger & lossless soft-eviction / SHA-256 restore
│   ├── portability.py      # 100% read-only Markdown export (Obsidian/Git) & SQLite Online Backup snapshots
│   ├── audit.py            # Read-only AGENTS.md / project rules vs. memory conflict auditor
│   ├── trace.py            # Opt-in body-free JSONL recall trace logger & summary analyzer
│   ├── mcp_server.py       # Zero-dep 4-tool MCP stdio server (NDJSON + Content-Length)
│   ├── hud.py              # Native dark tkinter floating status pill & memory workbench
│   └── cli.py              # setup / connect / disconnect / doctor / config / add / search / export / backup / stats / mcp / ui
├── docs/images/            # Controlled 52k benchmark & ablation charts (ZH/EN) & native workbench screenshots
├── modelfiles/             # Official Q8_0 / Q4_K_M Ollama Modelfiles
├── demo.py                 # Synthetic isolated end-to-end demo
└── tests/                  # Offline unit tests + live V2 smoke test + cloud E2E suite
```

---

## Testing

```bash
# Offline zero-dependency unit tests (no network / no Ollama required)
python -m unittest discover -s tests -p "test_core.py" -v
python -m unittest discover -s tests -p "test_governance.py" -v
python -m unittest discover -s tests -p "test_timeline_and_portability.py" -v

# Live end-to-end smoke test against a local Ollama V2 instance
python -m unittest discover -s tests -p "test_live_v2.py" -v
```

---

## License, Attribution & Disambiguation

- **License**: **Apache License 2.0** (see [LICENSE](LICENSE) and [NOTICE](NOTICE))
- **Author**: **B站IM榴莲 (IMLIULIAN)** · **ZeroZoneX**
- **Companion Model**: [`Qwen3.5-4B-Memory-V2`](https://huggingface.co/IMLIULIAN/qwen3.5-4b-memory-v2) (derived from `Qwen/Qwen3.5-4B`, Apache-2.0)
- **Schema Lineage & Compatibility**: The structured fact schema conventions (`bank`, `type: world | experience | observation`, `occurred_start` / `occurred_end` / `mentioned_at`, `where`, `entities`) are inspired by and kept conceptually compatible with [**Vectorize Hindsight**](https://github.com/vectorize-io/hindsight) (MIT License). ZeroMem's zero-dependency single-file SQLite/FTS5 store, pre-write admission loop, deterministic guardrails, timeline supersession chain, credential redaction, read-only export/online backup, MCP server, and native desktop workbench are independent implementations containing no Hindsight source code.
- **Name Disambiguation**: **ZeroMem (零忆)** is an independent local SQLite/FTS5 memory engine built around a local 4B specialist model, and is **not** affiliated with or derived from the academic paper *Zero-Mem: Zero-Token Memory Operations for LLM Agents* (Yilin Xiao et al., [arXiv:2607.29377](https://arxiv.org/abs/2607.29377), July 2026), which proposes a non-generative zero-LLM-token entity-context graph and temporal hierarchy framework.
