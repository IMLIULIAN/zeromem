"""Live end-to-end smoke test against a local Ollama Qwen3.5-4B-Memory-V2 instance."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from v2mem import Memory
from v2mem.ollama import OllamaClient, OllamaError


class TestLiveV2(unittest.TestCase):
    def setUp(self) -> None:
        self.client = OllamaClient()
        try:
            self.model = self.client.resolve_model()
        except OllamaError as exc:
            self.skipTest(f"Local Ollama V2 not available: {exc}")

    def test_live_extract_admit_and_bilingual_recall(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2mem_live_") as tmpdir:
            mem = Memory(db_path=Path(tmpdir) / "live.db", model=self.model)
            note = (
                "2026-03-01 记录：API 网关速率上限将于 2026-03-15 起调整为每分钟 120 次，"
                "原上限为每分钟 60 次。本次仅调整公开查询接口，不改变内部管理端口。"
            )
            res = mem.remember(
                note,
                ns="live-smoke",
                mentioned_at="2026-03-01T08:00:00Z",
                verify=True,
            )
            self.assertGreaterEqual(res.extracted, 1)
            self.assertGreaterEqual(res.saved, 1)
            self.assertEqual(res.guard_rejected, 0)

            # Verify no record-date leak into occurred_start
            for f in res.facts:
                if f.get("occurred_start"):
                    self.assertNotIn("2026-03-01", str(f["occurred_start"]))

            # Recall using Chinese keywords works even if V2 extracted English prose
            hits = mem.recall("网关 120", ns="live-smoke", limit=5)
            self.assertGreaterEqual(len(hits), 1)
            self.assertTrue(any("120" in h.text for h in hits))
            self.assertTrue(all(len(h.evidence) >= 1 for h in hits))


if __name__ == "__main__":
    unittest.main()
