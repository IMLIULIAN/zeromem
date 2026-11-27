"""Core Memory facade: remember() (Extract -> 1-round Admit -> Guard -> SQLite) and recall()."""

from __future__ import annotations

import hashlib
import inspect
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import (
    FACT_ADMISSION_SYSTEM,
    FACT_EXTRACTION_SYSTEM,
    FACT_SUPERSESSION_SYSTEM,
    VALID_DECISIONS,
    Segment,
    _SUPERSESSION_STOPWORDS,
    build_segments,
    build_supersession_schema,
    guard_fact,
    has_supersession_cue,
    is_evidence_verbatim_in_fact,
    is_noop_revision,
    parse_json_object,
    window_segments,
)
from .expiry import ADAPTIVE_MIN_LIMIT, ADAPTIVE_SCORE_RATIO, resolve_default_limit
from .ollama import DEFAULT_NUM_CTX, DEFAULT_OLLAMA_URL, OllamaClient
from .portability import backup_database, export_markdown
from .store import SQLiteMemoryStore, tokenize_for_fts
from .trace import RecallTracer, build_trace_record, resolve_trace_dir


def default_home_dir() -> Path:
    """Return ZeroMem home dir, preserving legacy ~/.v2-memory if already populated."""
    new_dir = Path.home() / ".zeromem"
    legacy_dir = Path.home() / ".v2-memory"
    if (legacy_dir / "memory.db").is_file() and not (new_dir / "memory.db").is_file():
        return legacy_dir
    return new_dir


def default_db_path() -> Path:
    env_path = os.environ.get("ZEROMEM_DB") or os.environ.get("V2MEM_DB")
    if env_path:
        return Path(env_path).expanduser()
    return default_home_dir() / "memory.db"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class RememberResult:
    doc_id: str
    namespace: str
    deduplicated: bool = False
    windows: int = 0
    extracted: int = 0
    saved: int = 0
    revised: int = 0
    superseded: int = 0
    dropped: int = 0
    held: int = 0
    guard_rejected: int = 0
    model: str = ""
    fact_ids: list[int] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecallHit:
    fact_id: int
    doc_id: str
    namespace: str
    text: str
    type: str
    occurred_start: str | None
    occurred_end: str | None
    where: str | None
    entities: list[str]
    evidence_ids: list[str]
    evidence: list[dict[str, Any]]
    action: str
    mentioned_at: str
    score: float
    until_expiry: dict[str, Any] | None = None
    supersedes_fact_id: int | None = None
    superseded_by: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for opt_key in ("until_expiry", "supersedes_fact_id", "superseded_by"):
            if data.get(opt_key) is None:
                data.pop(opt_key, None)
        return data


def _hit_to_mapping(hit: RecallHit | dict[str, Any]) -> dict[str, Any]:
    if isinstance(hit, RecallHit):
        return hit.to_dict()
    return dict(hit)


def deduplicate_recall_evidence(
    hits: list[RecallHit | dict[str, Any]],
    *,
    fold_verbatim: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Deduplicate shared (doc_id, seg_id) evidence segments across hits and optionally fold verbatim repeats.

    Never mutates input RecallHit objects. Preserves segment IDs and speaker roles for full traceability.
    """
    seen_seg_first_hit: dict[tuple[str, str], int] = {}
    deduped_items: list[dict[str, Any]] = []
    shared_saved = 0
    verbatim_folded = 0

    for idx, hit in enumerate(hits, 1):
        item = _hit_to_mapping(hit)
        doc_id = str(item.get("doc_id") or "")
        fact_text = str(item.get("text") or "")
        raw_ev = item.get("evidence") or []
        compact_ev: list[dict[str, Any]] = []

        for ev in raw_ev:
            if not isinstance(ev, dict):
                continue
            seg_id = str(ev.get("id") or "")
            ev_text = str(ev.get("text") or "")
            role = ev.get("role")
            key = (doc_id, seg_id)

            if key in seen_seg_first_hit:
                ref_entry: dict[str, Any] = {
                    "id": seg_id,
                    "dedup_ref": f"#{seen_seg_first_hit[key]}:{doc_id}:{seg_id}" if doc_id else f"#{seen_seg_first_hit[key]}:{seg_id}",
                }
                if role:
                    ref_entry["role"] = role
                compact_ev.append(ref_entry)
                shared_saved += 1
                continue

            seen_seg_first_hit[key] = idx
            if fold_verbatim and is_evidence_verbatim_in_fact(fact_text, ev_text):
                fold_entry: dict[str, Any] = {"id": seg_id, "same_as_fact": True}
                if role:
                    fold_entry["role"] = role
                compact_ev.append(fold_entry)
                verbatim_folded += 1
                continue

            kept: dict[str, Any] = {"id": seg_id, "text": ev_text}
            if role:
                kept["role"] = role
            compact_ev.append(kept)

        item["evidence"] = compact_ev
        deduped_items.append(item)

    stats = {
        "shared_deduped": shared_saved,
        "verbatim_folded": verbatim_folded,
        "total_saved": shared_saved + verbatim_folded,
    }
    return deduped_items, stats


def format_recall_context(
    hits: list[RecallHit | dict[str, Any]],
    *,
    include_evidence: bool = True,
    dedup_evidence: bool = True,
    fold_verbatim: bool = True,
    header: str | None = None,
) -> str:
    """Format recalled hits into a concise, traceable dual-track prompt block with evidence deduplication."""
    if not hits:
        return "本地记忆系统未检索到与该问题相关的历史记录。"

    if header is None:
        header = (
            "以下是本地记忆系统检索到的最相关记忆（含去重原文证据）："
            if include_evidence
            else "以下是本地记忆系统检索到的最相关精炼事实（纯事实模式，不含展开原文）："
        )

    lines = [header]
    seen_seg_first: dict[tuple[str, str], int] = {}

    for idx, hit in enumerate(hits, 1):
        item = _hit_to_mapping(hit)
        t_parts = [f"记录于 {item.get('mentioned_at') or ''}"]
        if item.get("occurred_start"):
            t_parts.append(f"发生/生效于 {item['occurred_start']}")
        if item.get("supersedes_fact_id") is not None:
            t_parts.append(f"替代 #{item['supersedes_fact_id']}")
        if item.get("superseded_by"):
            t_parts.append("历史旧状态·已被新事实替代")
        if isinstance(item.get("until_expiry"), dict) and item["until_expiry"].get("label"):
            t_parts.append(str(item["until_expiry"]["label"]))

        fact_text = str(item.get("text") or "")
        lines.append(f"{idx}. [{' | '.join(t_parts)}] 事实：{fact_text}")

        if not include_evidence:
            continue

        doc_id = str(item.get("doc_id") or "")
        ref_tags: list[str] = []
        for ev in item.get("evidence") or []:
            if not isinstance(ev, dict):
                continue
            seg_id = str(ev.get("id") or "")
            ev_text = str(ev.get("text") or "")
            role = ev.get("role")
            role_prefix = f"[{role}] " if role else ""
            key = (doc_id, seg_id)

            if dedup_evidence and key in seen_seg_first:
                ref_tags.append(f"{seg_id}同#{seen_seg_first[key]}")
                continue

            seen_seg_first[key] = idx
            if dedup_evidence and fold_verbatim and is_evidence_verbatim_in_fact(fact_text, ev_text):
                role_note = f"·{role}" if role else ""
                ref_tags.append(f"{seg_id}{role_note}同正文")
                continue

            lines.append(f"   原文({seg_id}): {role_prefix}{ev_text}")

        if ref_tags:
            lines.append(f"   证据引用: {', '.join(ref_tags)}")

    return "\n".join(lines)


class Memory:
    """Out-of-the-box local memory engine tailored for Qwen3.5-4B-Memory-V2."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        model: str | None = None,
        ollama_url: str | None = None,
        num_ctx: int | None = None,
        timeout: float | None = None,
        client: OllamaClient | None = None,
    ) -> None:
        self.db_path = Path(db_path).expanduser() if db_path else default_db_path()
        self.store = SQLiteMemoryStore(self.db_path)
        resolved_timeout = (
            float(timeout)
            if timeout is not None
            else float(os.environ.get("ZEROMEM_TIMEOUT") or os.environ.get("V2MEM_TIMEOUT") or 90.0)
        )
        self.client = client or OllamaClient(
            base_url=ollama_url or os.environ.get("ZEROMEM_OLLAMA_URL") or os.environ.get("V2MEM_OLLAMA_URL") or DEFAULT_OLLAMA_URL,
            model=model or os.environ.get("ZEROMEM_MODEL") or os.environ.get("V2MEM_MODEL") or None,
            num_ctx=num_ctx or int(os.environ.get("ZEROMEM_NUM_CTX") or os.environ.get("V2MEM_NUM_CTX") or DEFAULT_NUM_CTX),
            timeout=resolved_timeout,
        )

    def remember(
        self,
        content: str | list[dict[str, Any]],
        *,
        ns: str = "default",
        source_id: str | None = None,
        mentioned_at: str | None = None,
        verify: bool = True,
        link_supersessions: bool = True,
        max_window_chars: int = 1400,
    ) -> RememberResult:
        """Ingest text or chat turns: short-window chunk -> extract -> prior FTS5 lookup -> 1-round admit -> guard -> store."""
        namespace = (ns or "default").strip() or "default"
        ts = (mentioned_at or utc_now_iso()).strip()
        created_at = utc_now_iso()

        segments = build_segments(content)
        if not segments:
            return RememberResult(doc_id="", namespace=namespace)

        canonical_body = "\n".join(f"{s.role or ''}:{s.text}" for s in segments)
        content_sha = hashlib.sha256(canonical_body.encode("utf-8")).hexdigest()
        existing_doc = self.store.has_document(namespace, content_sha)
        if existing_doc:
            return RememberResult(doc_id=existing_doc, namespace=namespace, deduplicated=True)

        doc_id = f"doc-{content_sha[:12]}"
        windows = window_segments(segments, max_window_chars=max_window_chars, overlap=1)
        seg_map = {s.id: s for s in segments}

        result = RememberResult(doc_id=doc_id, namespace=namespace, windows=len(windows))
        accepted_facts: list[dict[str, Any]] = []
        seen_fact_keys: set[str] = set()

        for win_idx, win in enumerate(windows):
            win_doc_id = doc_id if len(windows) == 1 else f"{doc_id}-w{win_idx}"
            candidates, used_model = self._extract_window(
                namespace=namespace,
                doc_id=win_doc_id,
                window=win,
            )
            if used_model:
                result.model = used_model
            result.extracted += len(candidates)
            if not candidates:
                continue

            if verify:
                reviewed = self._admit_window(
                    window=win,
                    candidates=candidates,
                    mentioned_at=ts,
                    namespace=namespace,
                    result=result,
                )
            else:
                reviewed = [{**c, "action": "bypass"} for c in candidates]

            window_accepted: list[dict[str, Any]] = []
            for cand in reviewed:
                action = str(cand.get("action") or "pass")
                guarded, _ = guard_fact(cand, seg_map, mentioned_at=ts)
                if not guarded:
                    result.guard_rejected += 1
                    continue
                dedup_key = guarded["text"].lower()
                if dedup_key in seen_fact_keys:
                    continue
                seen_fact_keys.add(dedup_key)
                guarded["action"] = action
                window_accepted.append(guarded)

            if verify and link_supersessions and window_accepted:
                self._link_supersessions_gated(
                    namespace=namespace,
                    accepted_facts=window_accepted,
                    max_prior_items=4,
                )

            accepted_facts.extend(window_accepted)

        saved_ids = self.store.save_document_and_facts(
            doc_id=doc_id,
            namespace=namespace,
            content_sha256=content_sha,
            source_id=source_id,
            mentioned_at=ts,
            created_at=created_at,
            segments=segments,
            facts=accepted_facts,
        )
        result.saved = len(saved_ids)
        result.superseded = sum(1 for f in accepted_facts if f.get("supersedes_fact_id") is not None)
        result.fact_ids = saved_ids
        result.facts = accepted_facts
        return result

    def recall(
        self,
        query: str,
        *,
        ns: str | None = None,
        limit: int | None = None,
        adaptive: bool = True,
        min_limit: int = ADAPTIVE_MIN_LIMIT,
        score_ratio: float = ADAPTIVE_SCORE_RATIO,
        as_of: str | date | None = None,
        occurred_from: str | date | None = None,
        occurred_to: str | date | None = None,
        include_superseded: bool = True,
        recall_log_dir: str | Path | None = None,
        trace_channel: str = "zeromem_recall",
    ) -> list[RecallHit]:
        """Dual-track FTS5 recall with occurred-time filter, supersession lineage, expiry demotion, and opt-in trace."""
        started = time.monotonic()
        eff_limit = resolve_default_limit(limit, adaptive=adaptive)
        rows = self.store.search(
            query,
            namespace=ns,
            limit=limit,
            adaptive=adaptive,
            min_limit=min_limit,
            score_ratio=score_ratio,
            as_of=as_of,
            occurred_from=occurred_from,
            occurred_to=occurred_to,
            include_superseded=include_superseded,
        )
        hits = [RecallHit(**r) for r in rows]
        trace_dir = resolve_trace_dir(recall_log_dir)
        if trace_dir is not None:
            try:
                RecallTracer(trace_dir).log(
                    build_trace_record(
                        query=query,
                        hits=rows,
                        elapsed_ms=(time.monotonic() - started) * 1000.0,
                        namespace=ns,
                        channel=trace_channel,
                        limit=eff_limit,
                        adaptive=adaptive,
                    )
                )
            except Exception:
                pass
        return hits

    def stats(self, ns: str | None = None) -> dict[str, Any]:
        return self.store.stats(namespace=ns)

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        return self.store.get_document(doc_id)

    def delete_fact(self, fact_id: int) -> bool:
        return self.store.delete_fact(fact_id)

    def delete_document(self, doc_id: str) -> bool:
        return self.store.delete_document(doc_id)

    def export_markdown(
        self,
        *,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        ns: str | None = None,
        occurred_from: str | date | None = None,
        occurred_to: str | date | None = None,
        as_of: str | date | None = None,
        include_superseded: bool = True,
        include_evicted: bool = False,
        frontmatter: bool = True,
    ) -> dict[str, Any]:
        return export_markdown(
            db_path=self.db_path,
            output_path=output_path,
            output_dir=output_dir,
            namespace=ns,
            occurred_from=occurred_from,
            occurred_to=occurred_to,
            as_of=as_of,
            include_superseded=include_superseded,
            include_evicted=include_evicted,
            frontmatter=frontmatter,
        )

    def backup(
        self,
        output_path: str | Path | None = None,
        *,
        overwrite: bool = False,
        verify: bool = True,
        write_sha256: bool = False,
    ) -> dict[str, Any]:
        return backup_database(
            db_path=self.db_path,
            output_path=output_path,
            overwrite=overwrite,
            verify=verify,
            write_sha256=write_sha256,
        )

    def _extract_window(
        self,
        *,
        namespace: str,
        doc_id: str,
        window: list[Segment],
    ) -> tuple[list[dict[str, Any]], str]:
        payload = {
            "bank": namespace,
            "document_id": doc_id,
            "evidence": [s.to_payload() for s in window],
        }
        raw, meta = self.client.chat_json(
            FACT_EXTRACTION_SYSTEM,
            payload,
            num_predict=1024,
        )
        parsed = parse_json_object(raw)
        if not parsed or not isinstance(parsed.get("facts"), list):
            return [], str(meta.get("model") or "")
        facts = [f for f in parsed["facts"] if isinstance(f, dict)]
        return facts, str(meta.get("model") or "")

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        return {
            t
            for t in tokenize_for_fts(text).split()
            if len(t) >= 2 and t not in _SUPERSESSION_STOPWORDS and not re.fullmatch(r"\d{1,2}", t)
        }

    @classmethod
    def _is_gated_prior_match(cls, cand_text: str, prior_text: str) -> bool:
        """Scale-invariant lexical overlap check excluding low-overlap hits and long boilerplate repeats."""
        a = cls._content_tokens(cand_text)
        b = cls._content_tokens(prior_text)
        if not a or not b:
            return False
        inter = len(a & b)
        jac = inter / max(1, len(a | b))
        cont = inter / max(1, min(len(a), len(b)))
        # Exclude near-identical long boilerplate repeats (e.g. routine checklists/minutes)
        if min(len(a), len(b)) >= 20 and jac >= 0.58:
            return False
        return inter >= 2 and cont >= 0.22 and jac >= 0.11

    def _find_gated_prior_facts(
        self,
        *,
        namespace: str,
        accepted_facts: list[dict[str, Any]],
        max_items: int = 4,
    ) -> tuple[list[int], list[dict[str, Any]]]:
        """Gated same-namespace FTS5 lookup returning only cue-bearing fact indices and high-signal prior facts."""
        if not accepted_facts or max_items <= 0:
            return [], []
        eligible_indices: list[int] = []
        collected: dict[int, tuple[float, dict[str, Any]]] = {}

        for idx, fact in enumerate(accepted_facts):
            text = str(fact.get("text") or "").strip()
            if not has_supersession_cue(text):
                continue
            entities = fact.get("entities") if isinstance(fact.get("entities"), list) else []
            parts = [str(e).strip() for e in entities[:4] if str(e).strip()]
            if text:
                parts.append(text[:120])
            query = " ".join(parts).strip()
            if not query:
                continue
            try:
                rows = self.store.search(
                    query,
                    namespace=namespace,
                    limit=3,
                    adaptive=False,
                    include_superseded=False,
                )
            except Exception:
                continue

            matched_any = False
            per_cand_added = 0
            for r in rows:
                fid = int(r["fact_id"])
                score = float(r.get("score") or 0.0)
                if not self._is_gated_prior_match(text, str(r.get("text") or "")):
                    continue
                matched_any = True
                per_cand_added += 1
                prev = collected.get(fid)
                if prev is None or score > prev[0]:
                    collected[fid] = (
                        score,
                        {
                            "fact_id": fid,
                            "text": str(r["text"]),
                            "occurred_start": r.get("occurred_start"),
                            "mentioned_at": str(r.get("mentioned_at") or ""),
                            "entities": list(r.get("entities") or [])[:6],
                        },
                    )
                if per_cand_added >= 2:
                    break
            if matched_any:
                eligible_indices.append(idx)

        if not eligible_indices or not collected:
            return [], []
        ranked = sorted(collected.values(), key=lambda pair: (-pair[0], pair[1]["fact_id"]))
        return eligible_indices, [item for _, item in ranked[:max_items]]

    @staticmethod
    def _parse_single_prior_id(raw: Any, offered_prior_ids: set[int]) -> int | None:
        if not offered_prior_ids or raw is None or isinstance(raw, bool):
            return None
        candidates: list[Any] = list(raw) if isinstance(raw, list) else [raw]
        for val in candidates:
            if isinstance(val, bool):
                continue
            parsed: int | None = None
            if isinstance(val, int):
                parsed = val
            elif isinstance(val, str) and val.strip().isdigit():
                parsed = int(val.strip())
            elif isinstance(val, dict) and "supersedes_fact_id" in val:
                inner = val.get("supersedes_fact_id")
                if isinstance(inner, int) and not isinstance(inner, bool):
                    parsed = inner
                elif isinstance(inner, str) and inner.strip().isdigit():
                    parsed = int(inner.strip())
            if parsed is not None and parsed in offered_prior_ids:
                return parsed
        return None

    def _link_supersessions_gated(
        self,
        *,
        namespace: str,
        accepted_facts: list[dict[str, Any]],
        max_prior_items: int = 4,
    ) -> None:
        """Gated post-admission supersession linker; never touches or pollutes native admission."""
        eligible_indices, prior_facts = self._find_gated_prior_facts(
            namespace=namespace,
            accepted_facts=accepted_facts,
            max_items=max_prior_items,
        )
        if not eligible_indices or not prior_facts:
            return

        offered_prior_ids = {int(pf["fact_id"]) for pf in prior_facts if isinstance(pf.get("fact_id"), int)}
        if not offered_prior_ids:
            return

        new_facts_map: dict[str, dict[str, Any]] = {}
        slot_keys: list[str] = []
        for idx in eligible_indices:
            slot = str(idx)
            slot_keys.append(slot)
            f = accepted_facts[idx]
            new_facts_map[slot] = {
                "text": str(f.get("text") or ""),
                "occurred_start": f.get("occurred_start"),
                "entities": list(f.get("entities") or [])[:6],
            }

        payload: dict[str, Any] = {
            "new_facts": new_facts_map,
            "prior_facts": prior_facts,
        }
        chat_kwargs: dict[str, Any] = {"num_predict": 128}
        sup_schema = build_supersession_schema(slot_keys, sorted(offered_prior_ids))
        try:
            sig = inspect.signature(self.client.chat_json)
            if "response_format" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                chat_kwargs["response_format"] = sup_schema
        except (TypeError, ValueError):
            pass

        try:
            raw, _ = self.client.chat_json(FACT_SUPERSESSION_SYSTEM, payload, **chat_kwargs)
        except Exception:
            return

        mapping = parse_json_object(raw)
        if not mapping:
            return

        for idx in eligible_indices:
            slot = str(idx)
            if slot not in mapping:
                continue
            sup_id = self._parse_single_prior_id(mapping.get(slot), offered_prior_ids)
            if sup_id is not None:
                accepted_facts[idx]["supersedes_fact_id"] = sup_id

    def _admit_window(
        self,
        *,
        window: list[Segment],
        candidates: list[dict[str, Any]],
        mentioned_at: str,
        namespace: str,
        result: RememberResult,
    ) -> list[dict[str, Any]]:
        """Pure native single-round admission (3-field contract): pass/revise directly, isolate drop/hold/noop-revise."""
        facts_map: dict[str, dict[str, Any]] = {}
        for idx, cand in enumerate(candidates):
            facts_map[str(idx)] = {
                "text": str(cand.get("text") or ""),
                "type": str(cand.get("type") or "world"),
                "occurred_start": cand.get("occurred_start"),
                "occurred_end": cand.get("occurred_end"),
                "where": cand.get("where"),
                "entities": cand.get("entities") if isinstance(cand.get("entities"), list) else [],
                "mentioned_at": mentioned_at,
                "context": namespace,
                "tags": [],
            }

        payload: dict[str, Any] = {
            "evidence": [s.to_payload() for s in window],
            "facts": facts_map,
        }

        raw, _ = self.client.chat_json(
            FACT_ADMISSION_SYSTEM,
            payload,
            num_predict=768,
        )
        verdicts = parse_json_object(raw)
        if not verdicts:
            return [{**c, "action": "pass_fallback"} for c in candidates]

        approved: list[dict[str, Any]] = []
        for idx, cand in enumerate(candidates):
            verdict = verdicts.get(str(idx))
            if not isinstance(verdict, dict):
                result.held += 1
                continue
            decision = str(verdict.get("decision") or "").strip().lower()
            if decision not in VALID_DECISIONS:
                result.held += 1
                continue

            ev_override = verdict.get("evidence")
            ev = ev_override if isinstance(ev_override, list) and ev_override else cand.get("evidence")

            if decision == "pass":
                approved.append({**cand, "evidence": ev, "action": "pass"})
            elif decision == "revise":
                replacement = str(verdict.get("replacement") or "").strip()
                if replacement and not is_noop_revision(str(cand.get("text") or ""), replacement):
                    result.revised += 1
                    approved.append({**cand, "text": replacement, "evidence": ev, "action": "revise"})
                else:
                    # Intercept empty or no-op revise (replacement identical to rejected original text)
                    result.held += 1
            elif decision == "drop":
                result.dropped += 1
            else:
                result.held += 1

        return approved
