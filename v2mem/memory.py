"""Core Memory facade: remember() (Extract -> 1-round Admit -> Guard -> SQLite) and recall()."""

from __future__ import annotations

import hashlib
import inspect
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import (
    FACT_ADMISSION_SUPERSEDE_ADDENDUM,
    FACT_ADMISSION_SYSTEM,
    FACT_EXTRACTION_SYSTEM,
    VALID_DECISIONS,
    Segment,
    build_admission_supersede_schema,
    build_segments,
    guard_fact,
    parse_json_object,
    window_segments,
)
from .expiry import ADAPTIVE_MIN_LIMIT, ADAPTIVE_SCORE_RATIO, resolve_default_limit
from .ollama import DEFAULT_NUM_CTX, DEFAULT_OLLAMA_URL, OllamaClient
from .portability import backup_database, export_markdown
from .store import SQLiteMemoryStore
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
                prior_facts = (
                    self._find_prior_facts(namespace=namespace, candidates=candidates, max_items=6)
                    if link_supersessions
                    else []
                )
                reviewed = self._admit_window(
                    window=win,
                    candidates=candidates,
                    mentioned_at=ts,
                    namespace=namespace,
                    result=result,
                    prior_facts=prior_facts,
                )
            else:
                reviewed = [{**c, "action": "bypass"} for c in candidates]

            for cand in reviewed:
                action = str(cand.get("action") or "pass")
                supersedes_fid = cand.get("supersedes_fact_id")
                guarded, _ = guard_fact(cand, seg_map, mentioned_at=ts)
                if not guarded:
                    result.guard_rejected += 1
                    continue
                dedup_key = guarded["text"].lower()
                if dedup_key in seen_fact_keys:
                    continue
                seen_fact_keys.add(dedup_key)
                guarded["action"] = action
                if isinstance(supersedes_fid, int) and not isinstance(supersedes_fid, bool) and supersedes_fid > 0:
                    guarded["supersedes_fact_id"] = supersedes_fid
                accepted_facts.append(guarded)

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

    def _find_prior_facts(
        self,
        *,
        namespace: str,
        candidates: list[dict[str, Any]],
        max_items: int = 6,
    ) -> list[dict[str, Any]]:
        """Sub-millisecond same-namespace FTS5 lookup for active historical facts relevant to candidates."""
        if not candidates or max_items <= 0:
            return []
        collected: dict[int, tuple[float, dict[str, Any]]] = {}
        for cand in candidates:
            entities = cand.get("entities") if isinstance(cand.get("entities"), list) else []
            parts = [str(e).strip() for e in entities[:4] if str(e).strip()]
            text = str(cand.get("text") or "").strip()
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
            for r in rows:
                fid = int(r["fact_id"])
                score = float(r.get("score") or 0.0)
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
        ranked = sorted(collected.values(), key=lambda pair: (-pair[0], pair[1]["fact_id"]))
        return [item for _, item in ranked[:max_items]]

    @staticmethod
    def _parse_supersedes_id(verdict: dict[str, Any], offered_prior_ids: set[int]) -> int | None:
        if not offered_prior_ids:
            return None
        raw = verdict.get("supersedes_fact_id")
        raw_candidates: list[Any] = []
        if isinstance(raw, list):
            raw_candidates.extend(raw)
        elif raw is not None:
            raw_candidates.append(raw)
        elif isinstance(verdict.get("supersedes_fact_ids"), list):
            raw_candidates.extend(verdict["supersedes_fact_ids"])
        for val in raw_candidates:
            if isinstance(val, bool):
                continue
            parsed: int | None = None
            if isinstance(val, int):
                parsed = val
            elif isinstance(val, str) and val.strip().isdigit():
                parsed = int(val.strip())
            if parsed is not None and parsed in offered_prior_ids:
                return parsed
        return None

    def _admit_window(
        self,
        *,
        window: list[Segment],
        candidates: list[dict[str, Any]],
        mentioned_at: str,
        namespace: str,
        result: RememberResult,
        prior_facts: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Bounded single-round admission: pass/revise directly, optional supersedes_fact_id, isolate drop/hold."""
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
        offered_prior_ids: set[int] = set()
        admit_schema: dict[str, Any] | None = None
        if prior_facts:
            payload["prior_facts"] = prior_facts
            offered_prior_ids = {int(pf["fact_id"]) for pf in prior_facts if isinstance(pf.get("fact_id"), int)}
            system_prompt = f"{FACT_ADMISSION_SYSTEM} {FACT_ADMISSION_SUPERSEDE_ADDENDUM}"
            if offered_prior_ids:
                admit_schema = build_admission_supersede_schema(len(candidates), sorted(offered_prior_ids))
        else:
            system_prompt = FACT_ADMISSION_SYSTEM

        chat_kwargs: dict[str, Any] = {"num_predict": 768}
        if admit_schema is not None:
            try:
                sig = inspect.signature(self.client.chat_json)
                if "response_format" in sig.parameters or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                ):
                    chat_kwargs["response_format"] = admit_schema
            except (TypeError, ValueError):
                pass

        raw, _ = self.client.chat_json(
            system_prompt,
            payload,
            **chat_kwargs,
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
            supersedes_id = self._parse_supersedes_id(verdict, offered_prior_ids)

            if decision == "pass":
                entry = {**cand, "evidence": ev, "action": "pass"}
                if supersedes_id is not None:
                    entry["supersedes_fact_id"] = supersedes_id
                approved.append(entry)
            elif decision == "revise":
                replacement = str(verdict.get("replacement") or "").strip()
                if replacement:
                    result.revised += 1
                    entry = {**cand, "text": replacement, "evidence": ev, "action": "revise"}
                    if supersedes_id is not None:
                        entry["supersedes_fact_id"] = supersedes_id
                    approved.append(entry)
                else:
                    result.held += 1
            elif decision == "drop":
                result.dropped += 1
            else:
                result.held += 1

        return approved
