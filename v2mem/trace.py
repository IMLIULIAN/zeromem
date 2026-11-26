"""Opt-in sidecar JSONL recall trace logger and read-only summary analyzer.

Design invariants:
- Opt-in only: enabled via explicit directory or env `ZEROMEM_RECALL_LOG_DIR` /
  `MEMORY_RECALL_LOG_DIR`; disabled when `ZEROMEM_RECALL_LOG=0|off|false|no` or
  `MEMORY_RECALL_LOG=0|off|false|no` or unset -> zero side effects in tests.
- Zero body leakage: never persists fact text, document content, or evidence quotes;
  records only timing, namespace, query SHA-256/len, secret-checked short preview,
  hit IDs, types, numeric scores, and expiry flags.
- Rolling retention: 1 MiB x 3 rotated files x 30 days.
- Fail-open: logging errors are swallowed and never disrupt recall().
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from .contracts import redact_credentials

DEFAULT_TRACE_FILE = "zeromem-recall-trace.jsonl"
_DISABLED_VALUES = frozenset({"0", "off", "false", "no"})
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_EXTRA_SECRET_RE = re.compile(
    r"(?i)(?:\b(?:github_pat_[A-Za-z0-9_]{16,}|AIza[0-9A-Za-z_-]{20,})\b"
    r"|\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password|passwd|secret|token)\s*[:=]\s*\S{4,})"
)


def resolve_trace_dir(
    explicit_dir: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    source = os.environ if env is None else env
    for flag_key in ("ZEROMEM_RECALL_LOG", "MEMORY_RECALL_LOG"):
        flag = source.get(flag_key)
        if isinstance(flag, str) and flag.strip().lower() in _DISABLED_VALUES:
            return None
    if explicit_dir is not None and str(explicit_dir).strip():
        raw = str(explicit_dir).strip()
    else:
        raw = (
            source.get("ZEROMEM_RECALL_LOG_DIR")
            or source.get("MEMORY_RECALL_LOG_DIR")
            or ""
        ).strip()
    return Path(raw).expanduser().resolve() if raw else None


def hash_query(query: str) -> str:
    return hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:16]


def sanitize_query_preview(
    query: str,
    *,
    max_chars: int = 80,
    redact: bool = False,
) -> str:
    if not isinstance(query, str):
        return ""
    if redact:
        return "[REDACTED_BY_POLICY]"
    compact = re.sub(r"\s+", " ", query).strip()
    if not compact:
        return ""
    if redact_credentials(compact) != compact or _EXTRA_SECRET_RE.search(compact):
        return "[REDACTED_SENSITIVE_QUERY]"
    if _PHONE_RE.search(compact) or _EMAIL_RE.search(compact):
        return "[REDACTED_SENSITIVE_QUERY]"
    return compact[:max_chars] + "…" if len(compact) > max_chars else compact


class RecallTracer:
    """Rolling JSONL sidecar logger for recall metadata (never throws to callers)."""

    def __init__(
        self,
        directory: str | Path,
        *,
        filename: str = DEFAULT_TRACE_FILE,
        max_bytes: int = 1024 * 1024,
        keep_files: int = 3,
        keep_days: int = 30,
        now_fn=None,
    ) -> None:
        self.dir = Path(directory)
        self.filename = filename
        self.path = self.dir / filename
        self.max_bytes = max(128, int(max_bytes))
        self.keep_files = max(1, int(keep_files))
        self.keep_days = max(1, int(keep_days))
        self._now = now_fn or time.time
        self.written = 0
        self.errors = 0
        self.rotations = 0
        self.swept = 0
        p = Path(filename)
        self._stem = p.stem
        self._ext = p.suffix or ".jsonl"
        self._rotated_re = re.compile(
            r"^" + re.escape(self._stem) + r"\..+" + re.escape(self._ext) + r"$"
        )

    def _stamp(self, epoch_s: float) -> str:
        return datetime.fromtimestamp(epoch_s, tz=timezone.utc).strftime("%Y%m%d-%H%M%S")

    def _rotate_if_needed(self) -> None:
        if not self.path.exists():
            return
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if size < self.max_bytes:
            return
        base = f"{self._stem}.{self._stamp(self._now())}"
        target = self.dir / f"{base}{self._ext}"
        if target.exists():
            target = self.dir / f"{base}.{os.getpid()}.{self.rotations + 1}{self._ext}"
            if target.exists():
                return
        os.replace(self.path, target)
        self.rotations += 1

    def sweep(self) -> None:
        try:
            items: list[tuple[float, Path]] = []
            for entry in self.dir.iterdir():
                if not entry.is_file() or not self._rotated_re.fullmatch(entry.name):
                    continue
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    mtime = 0.0
                items.append((mtime, entry))
        except OSError:
            return
        items.sort(key=lambda pair: pair[0], reverse=True)
        cutoff = self._now() - self.keep_days * 86400.0
        kept: list[Path] = []
        for mtime, entry in items:
            if mtime and mtime < cutoff:
                try:
                    entry.unlink(missing_ok=True)
                    self.swept += 1
                except OSError:
                    pass
                continue
            kept.append(entry)
        for entry in kept[self.keep_files :]:
            try:
                entry.unlink(missing_ok=True)
                self.swept += 1
            except OSError:
                pass

    def log(self, record: Mapping[str, Any]) -> bool:
        try:
            ts = datetime.fromtimestamp(self._now(), tz=timezone.utc).isoformat()
            line = json.dumps({"t": ts, **dict(record)}, ensure_ascii=False, separators=(",", ":"))
            self.dir.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed()
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self.written += 1
            if self.written % 200 == 0 or self.rotations > 0:
                self.sweep()
            return True
        except Exception:
            self.errors += 1
            return False


def build_trace_record(
    *,
    query: str,
    hits: Sequence[Mapping[str, Any]],
    elapsed_ms: float,
    namespace: str | None = None,
    channel: str = "zeromem_recall",
    limit: int | None = None,
    adaptive: bool = True,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    source_env = os.environ if env is None else env
    redact = (
        source_env.get("ZEROMEM_RECALL_LOG_REDACT_QUERY") == "1"
        or source_env.get("MEMORY_RECALL_LOG_REDACT_QUERY") == "1"
    )
    compact_hits: list[dict[str, Any]] = []
    expired_count = 0
    for item in list(hits)[:32]:
        exp = item.get("until_expiry") if isinstance(item, Mapping) else None
        is_exp = isinstance(exp, Mapping) and exp.get("expired") is True
        if is_exp:
            expired_count += 1
        row: dict[str, Any] = {
            "id": item.get("fact_id"),
            "doc_id": item.get("doc_id"),
            "namespace": item.get("namespace"),
            "kind": item.get("type"),
            "action": item.get("action"),
        }
        score = item.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            row["score"] = round(float(score), 6)
        if is_exp:
            row["expired"] = True
            row["until"] = exp.get("until")
        compact_hits.append(row)

    return {
        "channel": channel,
        "namespace": namespace,
        "status": "ok" if compact_hits else "no_hits",
        "elapsed_ms": round(max(0.0, float(elapsed_ms)), 2),
        "query_sha256": hash_query(query),
        "query_len": len(query) if isinstance(query, str) else 0,
        "query_preview": sanitize_query_preview(query, redact=redact),
        "adaptive": bool(adaptive),
        "limit": limit,
        "hit_count": len(compact_hits),
        "expired_count": expired_count,
        "hits": compact_hits,
    }


def summarize_traces(directory: str | Path, *, days: int = 30) -> dict[str, Any]:
    """Read-only aggregation over `*-recall-trace*.jsonl` sidecar logs."""
    root = Path(directory).expanduser()
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))
    files = sorted(root.glob("*-recall-trace*.jsonl")) if root.is_dir() else []
    total = 0
    by_channel: Counter[str] = Counter()
    by_status: Counter[str] = Counter()
    latencies: list[float] = []
    zero_hit_queries: list[dict[str, Any]] = []
    hit_id_counts: Counter[str] = Counter()
    expired_hits_total = 0

    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            ts_str = rec.get("t")
            if isinstance(ts_str, str):
                try:
                    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if dt < cutoff:
                        continue
                except ValueError:
                    pass
            total += 1
            by_channel[str(rec.get("channel") or "unknown")] += 1
            status = str(rec.get("status") or "unknown")
            by_status[status] += 1
            elapsed = rec.get("elapsed_ms")
            if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
                latencies.append(float(elapsed))
            expired_hits_total += int(rec.get("expired_count") or 0)
            if int(rec.get("hit_count") or 0) == 0 and status in {"no_hits", "ok"}:
                zero_hit_queries.append(
                    {
                        "t": ts_str,
                        "channel": rec.get("channel"),
                        "namespace": rec.get("namespace"),
                        "query_sha256": rec.get("query_sha256"),
                        "query_preview": rec.get("query_preview"),
                    }
                )
            for h in rec.get("hits") or []:
                if isinstance(h, Mapping) and h.get("id") is not None:
                    key = f"{h.get('namespace') or h.get('kind') or 'fact'}:{h['id']}"
                    hit_id_counts[key] += 1

    latencies.sort()

    def pct(p: float) -> float | None:
        if not latencies:
            return None
        idx = min(len(latencies) - 1, max(0, int(round((len(latencies) - 1) * p))))
        return round(latencies[idx], 2)

    return {
        "schema_version": 1,
        "trace_dir": str(root),
        "days": int(days),
        "files_scanned": len(files),
        "total_calls": total,
        "by_channel": dict(by_channel),
        "by_status": dict(by_status),
        "hit_rate": round(by_status.get("ok", 0) / total, 4) if total else None,
        "zero_hit_count": len(zero_hit_queries),
        "expired_hits_total": expired_hits_total,
        "latency_ms": {
            "p50": pct(0.5),
            "p95": pct(0.95),
            "max": round(latencies[-1], 2) if latencies else None,
        },
        "top_recalled_ids": [{"id": k, "count": v} for k, v in hit_id_counts.most_common(15)],
        "recent_zero_hit_queries": zero_hit_queries[-15:],
    }
