"""Time-expiry tags (`until:YYYY-MM-DD` / `until:YYYYMMDD`) and adaptive dual-lane gating.

Design invariants:
- Zero schema changes: inspects explicit `until:` markers in tags, entities, text, or bound evidence.
- Strict calendar validation (fail-open: invalid dates like 2026-02-30 are ignored as no expiry).
- Negative-control parity: when disabled (`ZEROMEM_UNTIL=0|off|false` or `MEMORY_UNTIL=0|off|false`)
  or no record is expired, outputs remain unmodified (zero extra keys, zero order changes).
- Non-destructive annotation: never rewrites `text` or evidence quotes; expired records are
  stable-partitioned to the tail with structured `until_expiry` metadata.
- Adaptive relative-score gating: keeps hits within `[min_limit, limit]` above
  `score >= score_ratio * leading_score` (anchored on the top active hit so a high-scoring
  expired tail hit never suppresses active hits), with `ZEROMEM_ADAPTIVE_LIMIT_DISABLED=1`
  rollback to fixed `limit=6`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import calendar
from datetime import date
import os
import re
from typing import Any

ADAPTIVE_MIN_LIMIT = 2
ADAPTIVE_MAX_LIMIT = 8
LEGACY_FIXED_LIMIT = 6
ADAPTIVE_SCORE_RATIO = 0.35

_UNTIL_TOKEN_RE = re.compile(r"^until:(\d{4})-?(\d{2})-?(\d{2})$", re.IGNORECASE)
_UNTIL_INLINE_RE = re.compile(r"(?<![A-Za-z0-9_])until:(\d{4})-?(\d{2})-?(\d{2})(?![0-9])", re.IGNORECASE)
_OCCURRED_FULL_RE = re.compile(r"^(\d{4})-?(\d{2})-?(\d{2})(?:$|T|\s)")
_OCCURRED_YM_RE = re.compile(r"^(\d{4})-(\d{2})$")
_OCCURRED_Y_RE = re.compile(r"^(\d{4})$")
_DISABLED_VALUES = frozenset({"0", "off", "false", "no"})


def is_until_disabled(env: Mapping[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    for key in ("ZEROMEM_UNTIL", "MEMORY_UNTIL"):
        raw = source.get(key)
        if isinstance(raw, str) and raw.strip().lower() in _DISABLED_VALUES:
            return True
    return False


def is_adaptive_limit_disabled(env: Mapping[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    for key in ("ZEROMEM_ADAPTIVE_LIMIT_DISABLED", "EVENT_MEMORY_ADAPTIVE_LIMIT_DISABLED"):
        raw = source.get(key)
        if isinstance(raw, str) and raw.strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def resolve_default_limit(
    limit: int | None = None,
    *,
    adaptive: bool = True,
    env: Mapping[str, str] | None = None,
) -> int:
    if limit is not None:
        return max(1, min(int(limit), 50))
    if not adaptive or is_adaptive_limit_disabled(env):
        return LEGACY_FIXED_LIMIT
    return ADAPTIVE_MAX_LIMIT


def parse_calendar_date(year: int, month: int, day: int) -> date | None:
    if not (1970 <= year <= 9999):
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_occurred_bound(value: date | str | None, *, is_end: bool = False) -> date | None:
    """Parse YYYY, YYYY-MM, YYYY-MM-DD, YYYYMMDD, or ISO-8601 prefix into a calendar date bound."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    m = _OCCURRED_FULL_RE.match(s)
    if m:
        return parse_calendar_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = _OCCURRED_YM_RE.fullmatch(s)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if not (1970 <= year <= 9999 and 1 <= month <= 12):
            return None
        day = calendar.monthrange(year, month)[1] if is_end else 1
        return parse_calendar_date(year, month, day)
    m = _OCCURRED_Y_RE.fullmatch(s)
    if m:
        year = int(m.group(1))
        if not (1970 <= year <= 9999):
            return None
        return parse_calendar_date(year, 12, 31) if is_end else parse_calendar_date(year, 1, 1)
    return None


def normalize_occurred_range(
    occurred_from: date | str | None = None,
    occurred_to: date | str | None = None,
) -> tuple[date | None, date | None]:
    """Validate optional occurred_from / occurred_to filter bounds (fail-closed on bad dates)."""
    from_d: date | None = None
    to_d: date | None = None
    if occurred_from is not None and (not isinstance(occurred_from, str) or occurred_from.strip()):
        from_d = parse_occurred_bound(occurred_from, is_end=False)
        if from_d is None:
            raise ValueError(f"invalid-occurred-from: {occurred_from!r} (expected YYYY, YYYY-MM, or YYYY-MM-DD)")
    if occurred_to is not None and (not isinstance(occurred_to, str) or occurred_to.strip()):
        to_d = parse_occurred_bound(occurred_to, is_end=True)
        if to_d is None:
            raise ValueError(f"invalid-occurred-to: {occurred_to!r} (expected YYYY, YYYY-MM, or YYYY-MM-DD)")
    if from_d is not None and to_d is not None and from_d > to_d:
        raise ValueError(f"invalid-occurred-range: occurred_from ({from_d.isoformat()}) > occurred_to ({to_d.isoformat()})")
    return from_d, to_d


def fact_matches_occurred_range(
    occurred_start: str | None,
    occurred_end: str | None = None,
    *,
    from_date: date | None = None,
    to_date: date | None = None,
) -> bool:
    """Return True iff a fact's occurred interval overlaps [from_date, to_date]."""
    if from_date is None and to_date is None:
        return True
    f_start = parse_occurred_bound(occurred_start, is_end=False)
    if f_start is None:
        return False
    f_end = parse_occurred_bound(occurred_end, is_end=True) if occurred_end else None
    if f_end is None:
        f_end = parse_occurred_bound(occurred_start, is_end=True) or f_start
    if f_end < f_start:
        f_end = f_start
    if from_date is not None and f_end < from_date:
        return False
    if to_date is not None and f_start > to_date:
        return False
    return True


def partition_superseded(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Stable-partition non-superseded records ahead of superseded ones without dropping any item."""
    if not records:
        return [], 0
    current: list[dict[str, Any]] = []
    superseded: list[dict[str, Any]] = []
    for rec in records:
        item = rec if isinstance(rec, dict) else dict(rec)
        if item.get("superseded_by"):
            superseded.append(item)
        else:
            current.append(item)
    if not superseded:
        return current, 0
    return current + superseded, len(superseded)


def resolve_as_of_date(today: date | str | None = None) -> date:
    if isinstance(today, date):
        return today
    if isinstance(today, str):
        m = re.match(r"^\s*(\d{4})-?(\d{2})-?(\d{2})(?:$|T)", today)
        if m:
            parsed = parse_calendar_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if parsed is not None:
                return parsed
    return date.today()


def _extract_dates_from_text(text: str) -> list[date]:
    found: list[date] = []
    for m in _UNTIL_INLINE_RE.finditer(text or ""):
        parsed = parse_calendar_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if parsed is not None:
            found.append(parsed)
    return found


def parse_until_date(
    tags: Any = None,
    metadata: Mapping[str, Any] | None = None,
    *,
    text: str | None = None,
    evidence: Sequence[Mapping[str, Any]] | None = None,
) -> date | None:
    """Return the latest valid `until:` calendar date from tags, metadata, text, or bound evidence."""
    candidates: list[date] = []

    if isinstance(tags, str):
        tokens = re.split(r"[\s,]+", tags.strip())
    elif isinstance(tags, Sequence) and not isinstance(tags, (bytes, bytearray)):
        tokens = [t for t in tags if isinstance(t, str)]
    else:
        tokens = []

    for token in tokens:
        trimmed = token.strip().strip("[]()")
        m = _UNTIL_TOKEN_RE.fullmatch(trimmed)
        if m:
            parsed = parse_calendar_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if parsed is not None:
                candidates.append(parsed)
        else:
            candidates.extend(_extract_dates_from_text(trimmed))

    if isinstance(metadata, Mapping):
        raw_meta = metadata.get("until")
        if isinstance(raw_meta, str) and raw_meta.strip():
            trimmed = raw_meta.strip()
            prefixed = trimmed if trimmed.lower().startswith("until:") else f"until:{trimmed}"
            m = _UNTIL_TOKEN_RE.fullmatch(prefixed)
            if m:
                parsed = parse_calendar_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                if parsed is not None:
                    candidates.append(parsed)

    if isinstance(text, str) and "until:" in text.lower():
        candidates.extend(_extract_dates_from_text(text))

    if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes, bytearray)):
        for seg in evidence:
            if isinstance(seg, Mapping):
                seg_text = seg.get("text")
                if isinstance(seg_text, str) and "until:" in seg_text.lower():
                    candidates.extend(_extract_dates_from_text(seg_text))

    return max(candidates) if candidates else None


def evaluate_expiry(until_date: date | None, as_of: date) -> dict[str, Any] | None:
    if until_date is None:
        return None
    expired_days = (as_of - until_date).days
    if expired_days <= 0:
        return None
    iso = until_date.isoformat()
    return {
        "until": iso,
        "expired": True,
        "expired_days": expired_days,
        "label": f"已过期（until {iso}）",
    }


def record_expiry_info(
    record: Mapping[str, Any],
    *,
    today: date | str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Evaluate `until:` expiry on a ZeroMem hit dict without mutating text or evidence."""
    if is_until_disabled(env) or not isinstance(record, Mapping):
        return None
    as_of = resolve_as_of_date(today)
    tags: list[str] = []
    for key in ("tags", "entities"):
        val = record.get(key)
        if isinstance(val, list):
            tags.extend(str(x) for x in val if isinstance(x, str))
        elif isinstance(val, str):
            tags.append(val)
    meta = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else None
    ev = record.get("evidence") if isinstance(record.get("evidence"), Sequence) else None
    until_d = parse_until_date(
        tags,
        meta,
        text=str(record.get("text") or ""),
        evidence=ev if not isinstance(ev, (str, bytes, bytearray)) else None,
    )
    return evaluate_expiry(until_d, as_of)


def partition_expired(
    records: Sequence[Mapping[str, Any]],
    *,
    today: date | str | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Stable-partition records so expired items follow active ones without dropping any item."""
    if is_until_disabled(env) or not records:
        return [r if isinstance(r, dict) else dict(r) for r in records], 0

    as_of = resolve_as_of_date(today)
    active: list[dict[str, Any]] = []
    expired: list[dict[str, Any]] = []

    for rec in records:
        info = record_expiry_info(rec, today=as_of, env=env)
        if info is not None:
            expired.append({**rec, "until_expiry": info})
        else:
            active.append(rec if isinstance(rec, dict) else dict(rec))

    if not expired:
        return active, 0
    return active + expired, len(expired)


def _positive_score(item: Mapping[str, Any]) -> float | None:
    score = item.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool) and score > 0:
        return float(score)
    return None


def select_adaptive_hits(
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int | None = None,
    min_limit: int = ADAPTIVE_MIN_LIMIT,
    score_ratio: float = ADAPTIVE_SCORE_RATIO,
    adaptive: bool = True,
    min_distinct_docs: int = 2,
    env: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Adaptive relative-score dual-lane selector.

    - Expects `records` already ordered with active hits ahead of expired hits.
    - Drops weak-tail noise below `score_ratio * leading_score` while guaranteeing at
      least `min(min_limit, cap)` items and expanding up to `cap` (default 8) on dense hits.
    - Balances top fact score with distinct source-document coverage (dual-lane) while
      strictly keeping active items ahead of expired items.
    - When `adaptive=False` or `ZEROMEM_ADAPTIVE_LIMIT_DISABLED=1`, reverts to fixed cap.
    """
    items = [r if isinstance(r, dict) else dict(r) for r in records]
    if not items:
        return []

    cap = resolve_default_limit(limit, adaptive=adaptive, env=env)
    if not adaptive or is_adaptive_limit_disabled(env):
        return items[:cap]

    scores = [s for x in items if (s := _positive_score(x)) is not None]
    if not scores or score_ratio <= 0:
        return items[:cap]

    # Anchor on the leading candidate (top active hit after partition_expired) so a
    # higher-scoring expired record at the tail can never suppress active records.
    first_score = _positive_score(items[0])
    ref_score = first_score if first_score is not None else max(scores)
    cutoff = ref_score * float(score_ratio)
    floor = max(1, min(int(min_limit), cap))

    strong = [x for x in items if (s := _positive_score(x)) is not None and s >= cutoff]
    pool = strong if len(strong) >= floor else items[:floor]
    target_limit = min(cap, max(floor, len(pool)))

    if len(pool) <= target_limit:
        return pool

    # Dual-lane balancing within active non-demoted candidates: preserve top-ranked fact lane
    # while guaranteeing distinct document evidence coverage when multiple strong docs match.
    active_pool = [x for x in pool if not x.get("until_expiry") and not x.get("superseded_by")]
    demoted_pool = [x for x in pool if x.get("until_expiry") or x.get("superseded_by")]

    selected_active: list[dict[str, Any]] = []
    seen_fids: set[Any] = set()
    seen_docs: set[str] = set()
    doc_lane_target = min(int(min_distinct_docs), target_limit)

    # Pass 1: admit strongest hits from distinct documents up to doc_lane_target
    if doc_lane_target >= 2:
        for item in active_pool:
            doc_id = str(item.get("doc_id") or "")
            fid = item.get("fact_id")
            if doc_id and doc_id not in seen_docs:
                selected_active.append(item)
                seen_fids.add(fid)
                seen_docs.add(doc_id)
                if len(selected_active) >= doc_lane_target:
                    break

    # Pass 2: fill remaining active slots strictly by original score order
    for item in active_pool:
        if len(selected_active) >= target_limit:
            break
        fid = item.get("fact_id")
        if fid not in seen_fids:
            selected_active.append(item)
            seen_fids.add(fid)

    # Restore original rank order among selected active items
    rank_index = {id(x): idx for idx, x in enumerate(active_pool)}
    selected_active.sort(key=lambda x: rank_index.get(id(x), 0))

    # Pass 3: fill any remaining slots with strong superseded/expired contrast hits at the tail
    remaining = target_limit - len(selected_active)
    if remaining > 0 and demoted_pool:
        return selected_active + demoted_pool[:remaining]
    return selected_active[:target_limit]
