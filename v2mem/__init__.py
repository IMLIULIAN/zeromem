"""ZeroMem (零忆) core package tailored for Qwen3.5-4B-Memory-V2."""

from .audit import audit_rules_vs_memory
from .expiry import parse_until_date
from .governance import restore_evicted, retract_record, soft_evict
from .memory import Memory, RecallHit, RememberResult, deduplicate_recall_evidence, format_recall_context
from .portability import backup_database, export_markdown
from .trace import summarize_traces

__version__ = "0.2.1"

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
