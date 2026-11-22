"""ZeroMem (零忆) by ZeroZoneX — zero-dependency local memory engine for Qwen3.5-4B-Memory-V2."""

from v2mem import Memory, RecallHit, RememberResult, __version__

__all__ = [
    "Memory",
    "RecallHit",
    "RememberResult",
    "__version__",
]
