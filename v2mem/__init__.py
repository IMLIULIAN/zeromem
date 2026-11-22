"""ZeroMem (零忆) core package tailored for Qwen3.5-4B-Memory-V2."""

from .memory import Memory, RecallHit, RememberResult

__version__ = "0.2.0"

__all__ = [
    "Memory",
    "RecallHit",
    "RememberResult",
    "__version__",
]
