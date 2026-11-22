"""Core Memory facade: remember() (Extract -> 1-round Admit -> Guard -> SQLite) and recall()."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import (
    FACT_ADMISSION_SYSTEM,
    FACT_EXTRACTION_SYSTEM,
    VALID_DECISIONS,
    Segment,
    build_segments,
    guard_fact,
    parse_json_object,
    window_segments,
)
from .ollama import DEFAULT_NUM_CTX, DEFAULT_OLLAMA_URL, OllamaClient
from .store import SQLiteMemoryStore


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        max_window_chars: int = 1400,
    ) -> RememberResult:
        """Ingest text or chat turns: short-window chunk -> extract -> 1-round admit -> guard -> store."""
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
        result.fact_ids = saved_ids
        result.facts = accepted_facts
        return result

    def recall(
        self,
        query: str,
        *,
        ns: str | None = None,
        limit: int = 6,
    ) -> list[RecallHit]:
        """Dual-track FTS5 recall over facts and bound original evidence segments."""
        rows = self.store.search(query, namespace=ns, limit=limit)
        return [RecallHit(**r) for r in rows]

    def stats(self, ns: str | None = None) -> dict[str, Any]:
        return self.store.stats(namespace=ns)

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        return self.store.get_document(doc_id)

    def delete_fact(self, fact_id: int) -> bool:
        return self.store.delete_fact(fact_id)

    def delete_document(self, doc_id: str) -> bool:
        return self.store.delete_document(doc_id)

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

    def _admit_window(
        self,
        *,
        window: list[Segment],
        candidates: list[dict[str, Any]],
        mentioned_at: str,
        namespace: str,
        result: RememberResult,
    ) -> list[dict[str, Any]]:
        """Bounded single-round admission: pass/revise directly, isolate drop/hold per item."""
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

        payload = {
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
                if replacement:
                    result.revised += 1
                    approved.append({**cand, "text": replacement, "evidence": ev, "action": "revise"})
                else:
                    result.held += 1
            elif decision == "drop":
                result.dropped += 1
            else:
                result.held += 1

        return approved
