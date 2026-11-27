"""ZeroMem (零忆) by ZeroZoneX — zero-dependency local memory engine for Qwen3.5-4B-Memory-V2."""

from v2mem import (
    Memory,
    RecallHit,
    RememberResult,
    __version__,
    audit_rules_vs_memory,
    backup_database,
    deduplicate_recall_evidence,
    export_markdown,
    format_recall_context,
    parse_until_date,
    restore_evicted,
    retract_record,
    soft_evict,
    summarize_traces,
)

__all__ = [
    "Memory",
    "RecallHit",
    "RememberResult",
    "__version__",
    "audit_rules_vs_memory",
    "backup_database",
    "deduplicate_recall_evidence",
    "export_markdown",
    "format_recall_context",
    "parse_until_date",
    "restore_evicted",
    "retract_record",
    "soft_evict",
    "summarize_traces",
]
