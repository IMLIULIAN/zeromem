# Changelog

All notable changes to **ZeroMem（零忆）** (`zeromem` / `v2mem`) are documented in this file.

## [0.2.1] - 2026-11-27

### Improved
- **Pure Native Pre-Write Admission + Gated Standalone Supersession (`contracts.py`, `memory.py`)**:
  - Restored `_admit_window` to the 100% pure native 3-field `FACT_ADMISSION_SYSTEM` contract (eliminating out-of-distribution `prior_facts` / 4-field JSON Schema pollution during admission).
  - Extracted supersession linking into a lightweight gated standalone pass (`FACT_SUPERSESSION_SYSTEM` + `build_supersession_schema`, `num_predict=128`), triggered only for admitted & guarded facts containing explicit state-change/migration cues (`has_supersession_cue`) and passing size-independent token-overlap and boilerplate-echo filters (`FACT_ADMISSION_SUPERSEDE_ADDENDUM` retained for compatibility).
- **No-Op Revision Guard (`is_noop_revision`)**:
  - Intercepts `decision == "revise"` responses whose replacement is empty or materially identical to the original candidate (ignoring whitespace, casing, or trailing `| Involving:` tags) and routes them into `hold` isolation instead of persisting stale/polluted candidates.
- **Recall Evidence Deduplication & Verbatim Folding (`deduplicate_recall_evidence`, `format_recall_context`)**:
  - Deduplicates shared `(doc_id, seg_id)` evidence segments across returned hits and folds segments already contained verbatim in the fact text into compact segment references across SDK (`format_recall_context`), MCP (`memory_recall`), and CLI (`zmem search`) rendering layers—while keeping underlying `RecallHit.evidence` lossless.

### Verified (52k-Char Ablation Re-Benchmark, `L3 v0.2.0 -> L3-FIX v0.2.1`)
- Schema and cross-runtime invariants unchanged: `schema_version = 1`, `SCHEMA_SIG = 42e7ab99494de6a6b4e88eb2`, `APP_FORMAT_ID = 0x494D4C4C` (100% interoperable with `@imliulian/dsh-zeromem@0.2.1`).
- Deterministic rule defect rate: `1.07% (4/373) -> 0.00% (0/368)`; prior template leaks: `4/6 -> 0/0`; full-DB combined error rate (rule ∪ judge across 560 unique facts): `8.31% -> 6.79%` (`-29.6%` vs L1).
- Supersession edges: `12 (0 TP / 12 FP) -> 3 (including core TP sig_05 -> sig_02, FP -83.3%)`, lifting Top-1 hit rate on answerable questions from `9/10 (90%)` to `10/10 (100%)` with only `5/71` docs triggering the gated pass.
- Context chars fell on `12/12` questions (`2,313.4 -> 1,919.6 ch`, `-17.0%`), reducing single-step cloud input tokens by `-13.9%` (`1,742.1 -> 1,499.5 tok`, `-94.6%` vs full-context stuffing across all 12 questions at `¥0` cloud write cost) and average local search latency from `4.63 ms` to `3.73 ms`.

## [0.2.0] - 2026-11-26

### Added
- **Timeline Evolution & Event-Time Filtering**: Added same-DB `fact_supersessions` edge table, `occurred_from` / `occurred_to` / `include_superseded` recall filtering, active-first ranking, and conservative single-date grounding backfill.
- **Read-Only Markdown Export & SQLite Online Backup (`portability.py`)**: Added `zmem export --md` (`mode=ro`, `PRAGMA query_only=ON`, stdout / single file / Obsidian per-namespace directory) and `zmem backup` (SQLite Online Backup API flattened to `journal_mode=DELETE` with `integrity_check` and optional `.sha256`).
- **Memory Governance (`expiry.py`, `governance.py`, `audit.py`, `trace.py`)**: Added `until:YYYY-MM-DD` expiry demotion, adaptive `2–8` relative-score gating, fail-closed retraction ledger, lossless soft-eviction with canonical SHA-256 restore, rule-vs-memory audit, and opt-in body-free recall trace summaries.
- **Attribution & Disambiguation (`NOTICE`, `README.md`, `README_EN.md`)**: Documented Vectorize Hindsight (MIT) schema lineage compatibility and disambiguation from `arXiv:2607.29377` (*Zero-Mem*).
