"""100% read-only AGENTS.md / project rules <-> ZeroMem SQLite memory audit.

Design invariants:
- 100% read-only: opens SQLite via `mode=ro` + `PRAGMA query_only = ON`, never mutates any
  rule file or database, and writes report files ONLY when `out_dir` is explicitly provided.
- Zero LLM calls: deterministic section parsing (`<!-- tag:begin/end -->` and Markdown headings),
  bilingual tokenization via `store.tokenize_for_fts`, concrete technical anchor extraction
  (models, ports, paths, scripts, env vars), and secret/PII redaction on every emitted snippet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from .contracts import redact_credentials
from .expiry import evaluate_expiry, parse_until_date, resolve_as_of_date
from .store import tokenize_for_fts

_MARKER_RE = re.compile(
    r"<!--\s*([A-Za-z0-9._-]+):begin\s*-->(.*?)<!--\s*\1:end\s*-->",
    re.DOTALL,
)
_HEADING_RE = re.compile(r"^#{1,3}\s+(.+?)\s*$", re.MULTILINE)
_PORT_RE = re.compile(r"(?:127\.0\.0\.1:|localhost:|端口\s*|port\s*[=:]?\s*)(\d{4,5})\b", re.IGNORECASE)
_MODEL_RE = re.compile(
    r"(?<![A-Za-z0-9_./\\-])(?:qwen\d[\w.:-]*|llama[\d.:-][\w.:-]*|gpt-[\w.:-]+|"
    r"claude-[\w.:-]+|gemini-[\w.:-]+|deepseek-[\w.:-]+|mistral[\w.:-]*|"
    r"whisper[\w.-]*|paraformer[\w.-]*)(?![A-Za-z0-9_/\\])",
    re.IGNORECASE,
)
_PATH_RE = re.compile(r"(?:[A-Za-z]:[/\\]|~/|/)[^\s`'\"()（）【】，,;；。]{3,90}")
_SCRIPT_RE = re.compile(r"\b[A-Za-z0-9_-]{3,48}\.(?:py|mjs|js|ts|ps1|sh|cmd|bat|json|toml|sqlite3|db)\b", re.IGNORECASE)
_ENV_RE = re.compile(r"\b[A-Z][A-Z0-9_]{4,40}\b")
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

_NEGATION_RE = re.compile(r"禁止|不得|不许|不要|禁用|不自动|切勿|严禁|勿|避免|\b(?:never|forbid|forbidden|disallow|disable|must not|do not)\b", re.IGNORECASE)
_AFFIRM_DEFAULT_RE = re.compile(r"长期默认|默认使用|固定使用|优先使用|一律|必须|常驻|强制|\b(?:always|default|must|require|mandatory|prefer)\b", re.IGNORECASE)
_STOP_ANCHORS = frozenset({
    "UTF", "JSON", "JSONL", "HTTP", "HTTPS", "SQLITE", "AGENTS", "SKILL",
    "README", "TRUE", "FALSE", "NULL", "NONE", "WINDOWS", "LINUX", "MACOS",
    "LOCAL", "GLOBAL",
})


@dataclass
class RuleSection:
    file_path: str
    section_id: str
    title: str
    text: str
    chars: int
    est_tokens: int
    tokens: set[str]
    anchors: set[str]
    models: set[str]
    ports: set[str]


def estimate_tokens(text: str) -> int:
    """Lightweight CJK-aware token estimate (~0.75 tok/CJK char, ~0.28 tok/ASCII char)."""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u3400" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, int(round(cjk * 0.75 + other * 0.28)))


def safe_snippet(text: str, limit: int = 120) -> str:
    cleaned = redact_credentials(str(text or ""))
    cleaned = _PHONE_RE.sub("[PHONE_REDACTED]", cleaned)
    cleaned = _EMAIL_RE.sub("[EMAIL_REDACTED]", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:limit] + "…" if len(cleaned) > limit else cleaned


def extract_anchors(text: str) -> tuple[set[str], set[str], set[str]]:
    anchors: set[str] = set()
    models: set[str] = set()
    ports: set[str] = set()
    if not text:
        return anchors, models, ports

    for m in _PORT_RE.finditer(text):
        p = m.group(1)
        ports.add(p)
        anchors.add(f"port:{p}")
    for m in _MODEL_RE.finditer(text):
        val = re.sub(r"\s+", " ", m.group(0).strip().casefold())
        models.add(val)
        anchors.add(f"model:{val}")
    for m in _SCRIPT_RE.finditer(text):
        anchors.add(f"file:{m.group(0).casefold()}")
    for m in _PATH_RE.finditer(text):
        norm = m.group(0).replace("\\", "/").rstrip(".,;:)").casefold()
        parts = [p for p in norm.split("/") if p]
        if len(parts) >= 2:
            anchors.add("path:" + "/".join(parts[-2:]))
    for m in _ENV_RE.finditer(text):
        val = m.group(0)
        if val not in _STOP_ANCHORS and "_" in val:
            anchors.add(f"env:{val.casefold()}")
    return anchors, models, ports


def _tokenize_set(text: str) -> set[str]:
    return set(tokenize_for_fts(text).split())


def parse_rule_file(path: Path) -> tuple[dict[str, Any], list[RuleSection]]:
    raw = path.read_text(encoding="utf-8-sig")
    sections: list[RuleSection] = []
    consumed_spans: list[tuple[int, int]] = []

    for m in _MARKER_RE.finditer(raw):
        tag = m.group(1).strip()
        body = m.group(2).strip()
        consumed_spans.append((m.start(), m.end()))
        if not body:
            continue
        heading_m = _HEADING_RE.search(body)
        title = heading_m.group(1).strip() if heading_m else tag
        toks = _tokenize_set(body)
        anchors, models, ports = extract_anchors(body)
        sections.append(
            RuleSection(
                file_path=str(path),
                section_id=tag,
                title=title,
                text=body,
                chars=len(body),
                est_tokens=estimate_tokens(body),
                tokens=toks,
                anchors=anchors,
                models=models,
                ports=ports,
            )
        )

    remaining_chars = list(raw)
    for start, end in consumed_spans:
        for i in range(start, end):
            remaining_chars[i] = " "
    remainder = "".join(remaining_chars)
    headings = list(_HEADING_RE.finditer(remainder))
    for idx, hm in enumerate(headings):
        start = hm.start()
        end = headings[idx + 1].start() if idx + 1 < len(headings) else len(remainder)
        chunk = remainder[start:end].strip()
        if len(chunk) < 30:
            continue
        title = hm.group(1).strip()
        toks = _tokenize_set(chunk)
        anchors, models, ports = extract_anchors(chunk)
        sections.append(
            RuleSection(
                file_path=str(path),
                section_id=f"heading:{idx + 1}:{title[:24]}",
                title=title,
                text=chunk,
                chars=len(chunk),
                est_tokens=estimate_tokens(chunk),
                tokens=toks,
                anchors=anchors,
                models=models,
                ports=ports,
            )
        )

    if not sections and raw.strip():
        toks = _tokenize_set(raw)
        anchors, models, ports = extract_anchors(raw)
        sections.append(
            RuleSection(
                file_path=str(path),
                section_id="whole-file",
                title=path.name,
                text=raw.strip(),
                chars=len(raw.strip()),
                est_tokens=estimate_tokens(raw),
                tokens=toks,
                anchors=anchors,
                models=models,
                ports=ports,
            )
        )

    summary = {
        "file": str(path),
        "chars": len(raw),
        "est_tokens": estimate_tokens(raw),
        "section_count": len(sections),
        "oversized_sections": [
            {"section_id": s.section_id, "title": s.title, "chars": s.chars, "est_tokens": s.est_tokens}
            for s in sections
            if s.chars > 1200
        ],
    }
    return summary, sections


def discover_rule_files(
    *,
    project_dir: str | Path | None = None,
    include_home_globals: bool = False,
    home_override: Path | None = None,
) -> list[Path]:
    """Discover Markdown rule files in a project (and optionally user global Agent configs)."""
    candidates: list[Path] = []
    if project_dir is not None:
        proj = Path(project_dir).expanduser().resolve()
        for rel in ("AGENTS.md", "CLAUDE.md", ".cursorrules"):
            p = proj / rel
            if p.is_file():
                candidates.append(p)
        cursor_rules = proj / ".cursor" / "rules"
        if cursor_rules.is_dir():
            candidates.extend(sorted(cursor_rules.glob("*.mdc")))
            candidates.extend(sorted(cursor_rules.glob("*.md")))

    if include_home_globals:
        home = (home_override or Path.home()).expanduser().resolve()
        plugin_rules_root = home / ".gemini" / "config" / "plugins"
        if plugin_rules_root.is_dir():
            candidates.extend(sorted(plugin_rules_root.glob("*/rules/AGENTS.md")))
        for fixed in (
            home / ".gemini" / "AGENTS.md",
            home / ".codex" / "AGENTS.md",
            home / ".dsh" / "AGENTS.md",
            home / ".claude" / "CLAUDE.md",
        ):
            if fixed.is_file():
                candidates.append(fixed)

    deduped: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve()).casefold()
        if key not in seen and p.is_file():
            seen.add(key)
            deduped.append(p.resolve())
    return deduped


def load_zeromem_memories(
    db_path: str | Path,
    *,
    namespace: str | None = None,
    as_of: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read active ZeroMem facts strictly read-only and compute `until:` expiry statistics."""
    db_file = Path(db_path).expanduser().resolve()
    today = resolve_as_of_date(as_of)
    if not db_file.is_file():
        return [], {
            "as_of": today.isoformat(),
            "total_facts": 0,
            "until_tagged_active": 0,
            "until_tagged_expired": 0,
            "expired_facts": [],
        }

    conn = sqlite3.connect(db_file.as_uri() + "?mode=ro", uri=True, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    try:
        if namespace:
            rows = conn.execute(
                "SELECT * FROM facts WHERE namespace = ? ORDER BY fact_id ASC", (namespace,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM facts ORDER BY fact_id ASC").fetchall()

        seg_cache: dict[str, dict[str, str]] = {}
        doc_ids = sorted({str(r["doc_id"]) for r in rows})
        for did in doc_ids:
            seg_cache[did] = {
                str(sr["seg_id"]): str(sr["text"])
                for sr in conn.execute(
                    "SELECT seg_id, text FROM segments WHERE doc_id = ?", (did,)
                ).fetchall()
            }
    finally:
        conn.close()

    until_active = 0
    until_expired = 0
    expired_list: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for r in rows:
        fid = int(r["fact_id"])
        did = str(r["doc_id"])
        ns = str(r["namespace"])
        text = str(r["text"] or "")
        entities = json.loads(r["entities_json"] or "[]")
        ev_ids = json.loads(r["evidence_ids_json"] or "[]")
        segs = [{"id": eid, "text": seg_cache.get(did, {}).get(eid, "")} for eid in ev_ids]
        ud = parse_until_date(entities, text=text, evidence=segs)
        if ud is not None:
            info = evaluate_expiry(ud, today)
            if info:
                until_expired += 1
                expired_list.append(
                    {"fact_id": fid, "namespace": ns, "title": safe_snippet(text, 60), "until": info["until"]}
                )
            else:
                until_active += 1

        ev_joined = " ".join(s["text"] for s in segs if s["text"])
        records.append(
            {
                "id": f"fact:{fid}",
                "fact_id": fid,
                "doc_id": did,
                "store": f"zeromem:{ns}",
                "kind": str(r["type"] or "world"),
                "title": safe_snippet(text, 80),
                "text": f"{text}\n{ev_joined[:1200]}",
            }
        )

    expiry_summary = {
        "as_of": today.isoformat(),
        "total_facts": len(rows),
        "until_tagged_active": until_active,
        "until_tagged_expired": until_expired,
        "expired_facts": expired_list[:20],
    }
    return records, expiry_summary


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b) if inter else 0.0


def _containment(smaller: set[str], larger: set[str]) -> float:
    if not smaller or not larger:
        return 0.0
    return len(smaller & larger) / len(smaller)


def _rule_host_scope(file_path: str) -> str:
    norm = file_path.replace("\\", "/").casefold()
    if "/.gemini/" in norm:
        return "antigravity-global"
    if "/.codex/" in norm:
        return "codex-global"
    if "/.dsh/" in norm:
        return "dsh-global"
    if "/.claude/" in norm:
        return "claude-global"
    if "/.cursor/" in norm and "/.cursor/rules/" not in norm:
        return "cursor-global"
    return f"workspace:{Path(file_path).parent.name}"


def _short_rule_label(file_path: str) -> str:
    p = Path(file_path)
    parts = p.parts
    if len(parts) >= 3 and parts[-2].casefold() == "rules":
        return f"{parts[-3]}/{parts[-1]}"
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return p.name


def audit_rules_vs_memory(
    *,
    rule_files: Sequence[Path],
    memory_records: Sequence[Mapping[str, Any]],
    expiry_summary: Mapping[str, Any] | None = None,
    overlap_threshold: float = 0.22,
    dup_threshold: float = 0.45,
) -> dict[str, Any]:
    """Run deterministic overlap, duplicate, cross-host drift, and conflict analysis."""
    file_summaries: list[dict[str, Any]] = []
    all_sections: list[RuleSection] = []
    for rf in rule_files:
        summary, secs = parse_rule_file(Path(rf))
        summary["host_scope"] = _rule_host_scope(summary["file"])
        summary["label"] = _short_rule_label(summary["file"])
        file_summaries.append(summary)
        all_sections.extend(secs)

    same_session_duplicates: list[dict[str, Any]] = []
    cross_host_drifts: list[dict[str, Any]] = []
    cross_host_mirrored_ok = 0

    for i in range(len(all_sections)):
        for j in range(i + 1, len(all_sections)):
            a, b = all_sections[i], all_sections[j]
            if a.file_path == b.file_path:
                continue
            same_marker = (
                not a.section_id.startswith("heading:")
                and a.section_id != "whole-file"
                and a.section_id == b.section_id
            )
            jac = _jaccard(a.tokens, b.tokens)
            cont = max(_containment(a.tokens, b.tokens), _containment(b.tokens, a.tokens))
            if not (same_marker or jac >= dup_threshold or cont >= 0.72):
                continue
            scope_a = _rule_host_scope(a.file_path)
            scope_b = _rule_host_scope(b.file_path)
            both_distinct_globals = (
                scope_a.endswith("-global") and scope_b.endswith("-global") and scope_a != scope_b
            )
            entry = {
                "kind": "same_marker_block" if same_marker else "high_similarity_block",
                "section_a": {
                    "file": a.file_path,
                    "label": _short_rule_label(a.file_path),
                    "scope": scope_a,
                    "id": a.section_id,
                    "title": a.title,
                    "est_tokens": a.est_tokens,
                },
                "section_b": {
                    "file": b.file_path,
                    "label": _short_rule_label(b.file_path),
                    "scope": scope_b,
                    "id": b.section_id,
                    "title": b.title,
                    "est_tokens": b.est_tokens,
                },
                "jaccard": round(jac, 4),
                "containment": round(cont, 4),
                "estimated_duplicate_tokens": min(a.est_tokens, b.est_tokens),
            }
            if both_distinct_globals:
                if jac < 0.85:
                    cross_host_drifts.append({**entry, "category": "cross_host_sync_drift"})
                else:
                    cross_host_mirrored_ok += 1
            else:
                same_session_duplicates.append({**entry, "category": "same_session_double_loaded"})

    prepped_mem = []
    for rec in memory_records:
        text = str(rec.get("text") or "")
        toks = _tokenize_set(text)
        if not toks:
            continue
        anchors, models, ports = extract_anchors(text)
        prepped_mem.append((rec, toks, anchors, models, ports))

    overlaps_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    conflicts_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for sec in all_sections:
        if len(sec.tokens) < 6:
            continue
        sec_has_neg = bool(_NEGATION_RE.search(sec.text))
        sec_has_default = bool(_AFFIRM_DEFAULT_RE.search(sec.text))
        best_for_sec: list[tuple[float, dict[str, Any]]] = []

        for rec, m_toks, m_anchors, m_models, m_ports in prepped_mem:
            shared_anchors = sorted(sec.anchors & m_anchors)
            jac = _jaccard(sec.tokens, m_toks)
            cont = _containment(sec.tokens, m_toks)
            score = max(jac, cont * 0.75) + min(0.25, 0.08 * len(shared_anchors))

            conflict_reasons: list[str] = []
            if shared_anchors and sec.models and m_models and not (sec.models >= m_models or m_models >= sec.models):
                diff_models = sorted(sec.models ^ m_models)
                if diff_models and (sec_has_default or _AFFIRM_DEFAULT_RE.search(str(rec.get("text") or ""))):
                    conflict_reasons.append(f"model_divergence:{','.join(diff_models[:4])}")
            if (
                shared_anchors
                and sec.ports
                and m_ports
                and not (sec.ports <= m_ports or m_ports <= sec.ports)
                and len(sec.tokens & m_toks) >= 6
            ):
                conflict_reasons.append(f"port_divergence:rule={sorted(sec.ports)} vs mem={sorted(m_ports)}")
            if len(shared_anchors) >= 2 and cont >= 0.20:
                mem_text = str(rec.get("text") or "")
                mem_has_neg = bool(_NEGATION_RE.search(mem_text))
                mem_has_def = bool(_AFFIRM_DEFAULT_RE.search(mem_text))
                if (sec_has_neg and not mem_has_neg and mem_has_def) or (
                    mem_has_neg and not sec_has_neg and sec_has_default
                ):
                    conflict_reasons.append("directive_polarity_contrast")

            dedupe_key = (sec.section_id, str(rec["id"]))
            if conflict_reasons and (score >= overlap_threshold or len(shared_anchors) >= 2):
                cand = {
                    "rule_file": sec.file_path,
                    "rule_label": _short_rule_label(sec.file_path),
                    "rule_section": sec.section_id,
                    "rule_title": sec.title,
                    "memory_id": str(rec["id"]),
                    "memory_store": str(rec.get("store") or "zeromem"),
                    "memory_kind": str(rec.get("kind") or "fact"),
                    "memory_title": str(rec.get("title") or ""),
                    "shared_anchors": shared_anchors[:8],
                    "reasons": conflict_reasons,
                    "overlap_score": round(score, 4),
                    "memory_snippet": safe_snippet(str(rec.get("text") or ""), 110),
                }
                prev = conflicts_by_key.get(dedupe_key)
                if prev is None or cand["overlap_score"] > prev["overlap_score"]:
                    conflicts_by_key[dedupe_key] = cand
            elif score >= overlap_threshold and (len(shared_anchors) >= 1 or jac >= 0.25 or cont >= 0.35):
                best_for_sec.append(
                    (
                        score,
                        {
                            "rule_file": sec.file_path,
                            "rule_label": _short_rule_label(sec.file_path),
                            "rule_section": sec.section_id,
                            "rule_title": sec.title,
                            "rule_est_tokens": sec.est_tokens,
                            "memory_id": str(rec["id"]),
                            "memory_store": str(rec.get("store") or "zeromem"),
                            "memory_kind": str(rec.get("kind") or "fact"),
                            "memory_title": str(rec.get("title") or ""),
                            "shared_anchors": shared_anchors[:8],
                            "jaccard": round(jac, 4),
                            "rule_containment": round(cont, 4),
                            "overlap_score": round(score, 4),
                        },
                    )
                )

        best_for_sec.sort(key=lambda pair: (-pair[0], pair[1]["memory_id"]))
        for _, row in best_for_sec[:3]:
            dedupe_key = (row["rule_section"], row["memory_id"])
            prev = overlaps_by_key.get(dedupe_key)
            if prev is None or row["overlap_score"] > prev["overlap_score"]:
                overlaps_by_key[dedupe_key] = row

    overlaps = sorted(
        overlaps_by_key.values(),
        key=lambda r: (-r["overlap_score"], r["rule_section"], r["memory_id"]),
    )
    conflicts = sorted(
        conflicts_by_key.values(),
        key=lambda r: (-r["overlap_score"], r["rule_section"], r["memory_id"]),
    )
    cross_host_drifts.sort(key=lambda r: (r["jaccard"], r["section_a"]["id"]))
    same_session_duplicates.sort(key=lambda r: (-r["estimated_duplicate_tokens"], r["section_a"]["id"]))

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "summary": {
            "rule_files_scanned": len(file_summaries),
            "total_rule_chars": sum(f["chars"] for f in file_summaries),
            "total_rule_est_tokens": sum(f["est_tokens"] for f in file_summaries),
            "total_rule_sections": len(all_sections),
            "memory_records_scanned": len(memory_records),
            "same_session_duplicate_count": len(same_session_duplicates),
            "estimated_same_session_dup_tokens": sum(
                d["estimated_duplicate_tokens"] for d in same_session_duplicates
            ),
            "cross_host_mirrored_ok_count": cross_host_mirrored_ok,
            "cross_host_drift_count": len(cross_host_drifts),
            "overlap_count": len(overlaps),
            "conflict_count": len(conflicts),
        },
        "rule_files": file_summaries,
        "expiry_summary": dict(expiry_summary) if expiry_summary else None,
        "same_session_duplicates": same_session_duplicates[:50],
        "cross_host_drifts": cross_host_drifts[:50],
        "potential_conflicts": conflicts[:40],
        "rule_memory_overlaps": overlaps[:50],
    }


def render_audit_markdown(report: Mapping[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# ZeroMem Rules & Memory Audit (Read-Only)",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Rule files: `{s['rule_files_scanned']}` (~`{s['total_rule_est_tokens']}` tokens, `{s['total_rule_sections']}` sections)",
        f"- Memory facts scanned: `{s['memory_records_scanned']}`",
        f"- Same-session duplicates: `{s['same_session_duplicate_count']}` (~`{s['estimated_same_session_dup_tokens']}` dup tokens)",
        f"- Cross-host mirrored OK / drifts: `{s['cross_host_mirrored_ok_count']}` / `{s['cross_host_drift_count']}`",
        f"- Rule <-> memory overlaps / potential conflicts: `{s['overlap_count']}` / `{s['conflict_count']}`",
    ]
    exp = report.get("expiry_summary")
    if isinstance(exp, Mapping):
        lines.append(
            f"- `until:` tags (as of `{exp.get('as_of')}`): active=`{exp.get('until_tagged_active', 0)}`, "
            f"expired=`{exp.get('until_tagged_expired', 0)}`"
        )
    if report.get("potential_conflicts"):
        lines.extend(["", "## Potential Conflicts"])
        for c in report["potential_conflicts"][:20]:
            lines.append(
                f"- `{c['rule_label']}` [{c['rule_section']}] vs `{c['memory_id']}` "
                f"({', '.join(c['reasons'])}; anchors={', '.join(c['shared_anchors'])})"
            )
    if report.get("same_session_duplicates"):
        lines.extend(["", "## Same-Session Duplicates"])
        for d in report["same_session_duplicates"][:20]:
            lines.append(
                f"- `{d['section_a']['label']}` vs `{d['section_b']['label']}` [{d['section_a']['id']}] "
                f"jaccard={d['jaccard']} (~{d['estimated_duplicate_tokens']} tok)"
            )
    if report.get("cross_host_drifts"):
        lines.extend(["", "## Cross-Host Sync Drifts"])
        for d in report["cross_host_drifts"][:20]:
            lines.append(
                f"- `{d['section_a']['label']}` vs `{d['section_b']['label']}` [{d['section_a']['id']}] "
                f"jaccard={d['jaccard']}"
            )
    return "\n".join(lines) + "\n"


def write_audit_report(report: Mapping[str, Any], out_dir: str | Path) -> dict[str, str]:
    target = Path(out_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "rules-memory-audit.json"
    md_path = target / "rules-memory-audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_audit_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}
