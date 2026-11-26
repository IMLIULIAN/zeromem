"""Zero-dependency MCP stdio JSON-RPC 2.0 server exposing memory_remember and memory_recall."""

from __future__ import annotations

import json
import sys
from typing import Any

from .memory import Memory

TOOLS_SPEC = [
    {
        "name": "memory_remember",
        "description": (
            "Extract evidence-grounded facts from text using local Qwen3.5-4B-Memory-V2, "
            "run 1-round pre-write fact admission/revision, and save to local SQLite."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Text or note to remember. Long content is auto-chunked into <1500 char windows.",
                },
                "namespace": {
                    "type": "string",
                    "description": "Optional project or space namespace (default: 'default').",
                    "default": "default",
                },
                "source_id": {
                    "type": "string",
                    "description": "Optional source identifier or file path.",
                },
                "mentioned_at": {
                    "type": "string",
                    "description": "Optional ISO-8601 record timestamp (kept separate from event occurred_start).",
                },
                "verify": {
                    "type": "boolean",
                    "description": "Run V2 pre-write admission/revision gate (default: true).",
                    "default": True,
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "memory_recall",
        "description": (
            "Search local SQLite memory by keywords across both facts and bound original "
            "evidence segments (supports Chinese/English without an embedding model). "
            "For compound questions spanning multiple modules or metrics (e.g. API rate limit AND "
            "database connection pool), split into 2 concise keyword queries and call this tool "
            "in parallel instead of passing one long sentence."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Concise keyword query for one topic (empty string returns most recent facts).",
                },
                "namespace": {
                    "type": "string",
                    "description": "Optional namespace filter. Omit to search all namespaces.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Optional maximum number of facts to return (1-20; omit for adaptive 2-8 gating).",
                },
                "adaptive": {
                    "type": "boolean",
                    "description": "Enable adaptive relative-score tail pruning (default: true).",
                    "default": True,
                },
                "as_of": {
                    "type": "string",
                    "description": "Optional ISO date (YYYY-MM-DD) for evaluating until: expiry demotion.",
                },
                "occurred_from": {
                    "type": "string",
                    "description": "Optional inclusive event-time lower bound (YYYY, YYYY-MM, or YYYY-MM-DD).",
                },
                "occurred_to": {
                    "type": "string",
                    "description": "Optional inclusive event-time upper bound (YYYY, YYYY-MM, or YYYY-MM-DD).",
                },
                "include_superseded": {
                    "type": "boolean",
                    "description": "Include historical facts superseded by newer facts (default: true; demoted after current facts).",
                    "default": True,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_stats",
        "description": (
            "Return local ZeroMem database counts (documents, segments, facts, revised facts, "
            "and available namespaces) without invoking an LLM."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Optional namespace filter.",
                },
            },
        },
    },
    {
        "name": "memory_delete",
        "description": (
            "Delete an incorrect or outdated memory by fact_id or delete an entire source "
            "document by doc_id from local SQLite."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "fact_id": {
                    "type": "integer",
                    "description": "Numeric fact_id returned by memory_recall.",
                },
                "doc_id": {
                    "type": "string",
                    "description": "Document ID (e.g. doc-xxxxxxxxxxxx) to remove all its facts and segments.",
                },
            },
        },
    },
]


class MCPStdioServer:
    """Minimal standard-library JSON-RPC 2.0 MCP stdio server."""

    def __init__(self, memory: Memory) -> None:
        self.memory = memory
        self._use_content_length = False

    def handle_request(self, req: dict[str, Any]) -> dict[str, Any] | None:
        method = req.get("method")
        req_id = req.get("id")
        params = req.get("params") or {}

        if req_id is None:
            return None

        if method == "initialize":
            proto = str(params.get("protocolVersion") or "2024-11-05")
            return self._ok(
                req_id,
                {
                    "protocolVersion": proto,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": "zeromem",
                        "version": "0.2.0",
                    },
                    "instructions": (
                        "ZeroMem (零忆) local evidence-grounded memory powered by Qwen3.5-4B-Memory-V2. "
                        "Use memory_remember to store verified notes and memory_recall to retrieve facts with bound original evidence. "
                        "Retrieval discipline: (1) Single-topic question -> 1 focused keyword memory_recall call; "
                        "(2) Multi-topic / cross-document question -> issue 2 concise keyword memory_recall calls in parallel; "
                        "(3) Answer directly once evidence is retrieved; if nothing matches, retry at most once with alternate terms then report no record."
                    ),
                },
            )

        if method == "ping":
            return self._ok(req_id, {})

        if method == "tools/list":
            return self._ok(req_id, {"tools": TOOLS_SPEC})

        if method == "tools/call":
            return self._handle_tool_call(req_id, params)

        return self._err(req_id, -32601, f"Method not found: {method}")

    def _handle_tool_call(self, req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = str(params.get("name") or "")
        args = params.get("arguments") or {}
        try:
            if name == "memory_remember":
                res = self.memory.remember(
                    content=str(args.get("content") or ""),
                    ns=str(args.get("namespace") or "default"),
                    source_id=args.get("source_id"),
                    mentioned_at=args.get("mentioned_at"),
                    verify=bool(args.get("verify", True)),
                )
                return self._tool_text(req_id, res.to_dict())

            if name == "memory_recall":
                raw_limit = args.get("limit")
                hits = self.memory.recall(
                    query=str(args.get("query") or ""),
                    ns=args.get("namespace") or None,
                    limit=int(raw_limit) if raw_limit is not None else None,
                    adaptive=bool(args.get("adaptive", True)),
                    as_of=args.get("as_of") or None,
                    occurred_from=args.get("occurred_from") or None,
                    occurred_to=args.get("occurred_to") or None,
                    include_superseded=bool(args.get("include_superseded", True)),
                    trace_channel="zeromem_mcp",
                )
                expired_count = sum(1 for h in hits if h.until_expiry)
                payload: dict[str, Any] = {
                    "count": len(hits),
                    "items": [h.to_dict() for h in hits],
                }
                if expired_count > 0:
                    payload["expired_count"] = expired_count
                return self._tool_text(req_id, payload)

            if name == "memory_stats":
                ns = args.get("namespace") or None
                return self._tool_text(req_id, self.memory.stats(ns=ns))

            if name == "memory_delete":
                if args.get("fact_id") is not None:
                    deleted = self.memory.delete_fact(int(args["fact_id"]))
                    return self._tool_text(req_id, {"deleted": deleted, "target": "fact", "fact_id": int(args["fact_id"])})
                if args.get("doc_id"):
                    doc_id = str(args["doc_id"])
                    deleted = self.memory.delete_document(doc_id)
                    return self._tool_text(req_id, {"deleted": deleted, "target": "document", "doc_id": doc_id})
                return self._err(req_id, -32602, "Provide either fact_id or doc_id")

            return self._err(req_id, -32602, f"Unknown tool: {name}")
        except Exception as exc:
            return self._ok(
                req_id,
                {
                    "content": [{"type": "text", "text": f"Error: {exc}"}],
                    "isError": True,
                },
            )

    @staticmethod
    def _tool_text(req_id: Any, obj: dict[str, Any]) -> dict[str, Any]:
        return MCPStdioServer._ok(
            req_id,
            {
                "content": [{"type": "text", "text": json.dumps(obj, ensure_ascii=False, indent=2)}],
                "isError": False,
            },
        )

    @staticmethod
    def _ok(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _err(req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

    def _read_message(self) -> tuple[bool, dict[str, Any] | None]:
        stdin = sys.stdin.buffer
        while True:
            line = stdin.readline()
            if not line:
                return False, None
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(b"{"):
                self._use_content_length = False
                return True, json.loads(stripped.decode("utf-8"))
            headers: dict[str, str] = {}
            cur = line
            while cur and cur.strip():
                text = cur.decode("ascii", errors="ignore")
                if ":" in text:
                    k, v = text.split(":", 1)
                    headers[k.strip().lower()] = v.strip()
                cur = stdin.readline()
            length = int(headers.get("content-length", "0"))
            if length <= 0:
                continue
            body = stdin.read(length)
            if not body:
                return False, None
            self._use_content_length = True
            return True, json.loads(body.decode("utf-8"))

    def _write_message(self, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        out = sys.stdout.buffer
        if self._use_content_length:
            out.write(f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii"))
            out.write(raw)
        else:
            out.write(raw + b"\n")
        out.flush()

    def serve_forever(self) -> int:
        print("[zmem] ZeroMem MCP stdio server ready.", file=sys.stderr, flush=True)
        while True:
            try:
                ok, req = self._read_message()
            except json.JSONDecodeError as exc:
                self._write_message(self._err(None, -32700, f"Parse error: {exc}"))
                continue
            if not ok:
                return 0
            if not isinstance(req, dict):
                continue
            resp = self.handle_request(req)
            if resp is not None:
                self._write_message(resp)
