"""Zero-dependency Ollama HTTP client with anti-OTL guardrails and model bootstrap."""

from __future__ import annotations

import hashlib
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_NUM_CTX = 8192

HF_REPO_ID = "IMLIULIAN/qwen3.5-4b-memory-v2"
MODELSCOPE_BASE_URL = "https://modelscope.cn/models/IMLIULIAN/qwen3.5-4b-memory-v2/resolve/master"
HF_BASE_URL = "https://huggingface.co/IMLIULIAN/qwen3.5-4b-memory-v2/resolve/main"

OFFICIAL_WEIGHTS: dict[str, dict[str, Any]] = {
    "q8_0": {
        "filename": "qwen3.5-4b-memory-v2-Q8_0.gguf",
        "sha256": "be3259d5b1e2e78a49ee889247504188dd721a03c3df420a902b754d14d08ead",
        "size_bytes": 4_482_402_624,
        "ollama_ref": "imliulian/qwen3.5-4b-memory:v2",
        "ollama_hf_ref": f"hf.co/{HF_REPO_ID}:Q8_0",
        "local_tag": "qwen3.5:4b-memory-v2",
    },
    "q4_k_m": {
        "filename": "qwen3.5-4b-memory-v2-Q4_K_M.gguf",
        "sha256": "58d2b7939b43069cc2c610107a2f802f6c869ac30707bd183fc36dd670784e66",
        "size_bytes": 2_708_803_904,
        "ollama_ref": "imliulian/qwen3.5-4b-memory:v2-q4_k_m",
        "ollama_hf_ref": f"hf.co/{HF_REPO_ID}:Q4_K_M",
        "local_tag": "qwen3.5:4b-memory-v2-q4",
    },
}

PREFERRED_LOCAL_MODELS = (
    "imliulian/qwen3.5-4b-memory:v2",
    "imliulian/qwen3.5-4b-memory:v2-q8_0",
    "qwen3.5:4b-memory-v2",
    f"hf.co/{HF_REPO_ID}:Q8_0",
    f"hf.co/{HF_REPO_ID}:q8_0",
    "imliulian/qwen3.5-4b-memory:v2-q4_k_m",
    "qwen3.5:4b-memory-v2-q4",
    f"hf.co/{HF_REPO_ID}:Q4_K_M",
    f"hf.co/{HF_REPO_ID}:q4_k_m",
)


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an invalid response."""


def _is_v2_model_name(name: str) -> bool:
    low = name.lower()
    return ("memory-v2" in low) or ("qwen3.5-4b-memory:v2" in low)


class OllamaClient:
    """Standard-library Ollama /api/chat client with deterministic anti-OTL parameters."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str | None = None,
        num_ctx: int = DEFAULT_NUM_CTX,
        timeout: float = 90.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._explicit_model = model
        self._resolved_model: str | None = model
        self.num_ctx = int(num_ctx)
        self.timeout = float(timeout)

    def list_models(self) -> list[str]:
        url = f"{self.base_url}/api/tags"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=min(self.timeout, 10.0)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise OllamaError(f"无法连接 Ollama ({self.base_url}): {exc}") from exc
        models = data.get("models") or []
        return [str(m.get("name") or "") for m in models if m.get("name")]

    def list_running_models(self) -> list[str]:
        url = f"{self.base_url}/api/ps"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=min(self.timeout, 5.0)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return []
        models = data.get("models") or []
        return [str(m.get("name") or "") for m in models if m.get("name")]

    def resolve_model(self) -> str:
        """Prefer an already-resident V2 model to avoid VRAM reload churn, then check /api/tags."""
        if self._resolved_model:
            return self._resolved_model

        for running in self.list_running_models():
            if _is_v2_model_name(running):
                self._resolved_model = running
                return running

        available = self.list_models()
        avail_set = set(available)
        for cand in PREFERRED_LOCAL_MODELS:
            if cand in avail_set:
                self._resolved_model = cand
                return cand
        for name in available:
            if _is_v2_model_name(name):
                self._resolved_model = name
                return name
        raise OllamaError(
            "本机 Ollama 未检测到 Qwen3.5-4B-Memory-V2 模型。\n"
            "请运行以下任一命令一键安装：\n"
            "  1) zmem setup --source modelscope  # 国内推荐（魔搭高速镜像）\n"
            "  2) zmem setup --source hf          # HuggingFace GGUF 直拉\n"
            "  3) zmem setup --source ollama      # Ollama Hub 拉取\n"
            "  4) zmem setup --gguf <本地gguf路径> # 已有 GGUF 一键注册"
        )

    def chat_json(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        *,
        num_predict: int = 1024,
        retries: int = 1,
    ) -> tuple[str, dict[str, Any]]:
        """Invoke /api/chat with thinking disabled and anti-OTL sampling bounds."""
        model = self.resolve_model()
        url = f"{self.base_url}/api/chat"
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "stream": False,
            "think": False,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": int(num_predict),
                "temperature": 0.0,
                "top_k": 20,
                "top_p": 0.95,
                "presence_penalty": 0.0,
                "repeat_penalty": 1.08,
            },
        }
        payload_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")

        last_err: Exception | None = None
        for attempt in range(retries + 1):
            req = urllib.request.Request(
                url,
                data=payload_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t0 = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                elapsed = round(time.monotonic() - t0, 3)
                content = str(((res.get("message") or {}).get("content") or "")).strip()
                meta = {
                    "model": model,
                    "elapsed_sec": elapsed,
                    "done_reason": res.get("done_reason"),
                    "prompt_eval_count": res.get("prompt_eval_count"),
                    "eval_count": res.get("eval_count"),
                }
                return content, meta
            except (urllib.error.URLError, TimeoutError, socket.timeout, OSError, json.JSONDecodeError) as exc:
                last_err = exc
                if attempt < retries:
                    time.sleep(1.0)
        raise OllamaError(f"Ollama 调用失败 ({model}): {last_err}") from last_err


def build_modelfile_text(gguf_path: Path, num_ctx: int = DEFAULT_NUM_CTX) -> str:
    """Build an Ollama Modelfile matching official V2 inference parameters."""
    resolved = gguf_path.resolve().as_posix()
    return (
        f"FROM {resolved}\n"
        f"TEMPLATE {{{{ .Prompt }}}}\n"
        f"RENDERER qwen3.5\n"
        f"PARSER qwen3.5\n"
        f"PARAMETER num_ctx {int(num_ctx)}\n"
        f"PARAMETER temperature 0\n"
        f"PARAMETER top_k 20\n"
        f"PARAMETER top_p 0.95\n"
        f"PARAMETER presence_penalty 0\n"
    )


def register_gguf_in_ollama(
    gguf_path: Path,
    tag: str,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> str:
    """Register a local GGUF into Ollama using an auto-generated Modelfile."""
    if not shutil.which("ollama"):
        raise OllamaError("未在 PATH 中找到 `ollama` 命令，请先安装 Ollama (https://ollama.com)。")
    if not gguf_path.is_file():
        raise OllamaError(f"找不到 GGUF 文件: {gguf_path}")

    with tempfile.TemporaryDirectory(prefix="v2mem_mf_") as tmpdir:
        mf = Path(tmpdir) / "Modelfile"
        mf.write_text(build_modelfile_text(gguf_path, num_ctx=num_ctx), encoding="utf-8")
        proc = subprocess.run(["ollama", "create", tag, "-f", str(mf)], check=False)
        if proc.returncode != 0:
            raise OllamaError(f"`ollama create {tag}` 退出码为 {proc.returncode}")
    return tag


def pull_ollama_model(quant: str = "q8_0", use_hf: bool = False) -> str:
    """Pull V2 model directly via Ollama Registry or hf.co GGUF reference."""
    if not shutil.which("ollama"):
        raise OllamaError("未在 PATH 中找到 `ollama` 命令，请先安装 Ollama (https://ollama.com)。")
    spec = OFFICIAL_WEIGHTS[quant]
    ref = spec["ollama_hf_ref"] if use_hf else spec["ollama_ref"]
    proc = subprocess.run(["ollama", "pull", ref], check=False)
    if proc.returncode != 0:
        raise OllamaError(f"`ollama pull {ref}` 失败，退出码 {proc.returncode}")
    return ref


def download_gguf(
    quant: str = "q8_0",
    source: str = "modelscope",
    dest_dir: Path | None = None,
    verify_sha256: bool = True,
) -> Path:
    """Stream-download official GGUF from ModelScope or HuggingFace and verify SHA-256."""
    spec = OFFICIAL_WEIGHTS[quant]
    filename = spec["filename"]
    expected_sha = spec["sha256"]
    base = MODELSCOPE_BASE_URL if source == "modelscope" else HF_BASE_URL
    url = f"{base}/{filename}"

    legacy_weights = Path.home() / ".v2-memory" / "weights"
    if dest_dir is not None:
        target_dir = dest_dir
    elif (legacy_weights / filename).is_file():
        target_dir = legacy_weights
    else:
        target_dir = Path.home() / ".zeromem" / "weights"
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / filename
    part = target_dir / f"{filename}.part"

    if dest.is_file() and dest.stat().st_size == spec["size_bytes"]:
        if not verify_sha256 or _file_sha256(dest) == expected_sha:
            print(f"[zmem] 检测到已校验权重，复用本地文件: {dest}", file=sys.stderr)
            return dest

    print(f"[zmem] 开始从 {source} 下载 {filename} ...", file=sys.stderr)
    print(f"  URL: {url}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "ZeroMem/0.2"})
    hasher = hashlib.sha256()
    downloaded = 0
    total = spec["size_bytes"]
    last_print = 0.0

    with urllib.request.urlopen(req, timeout=60) as resp, part.open("wb") as out:
        content_len = resp.headers.get("Content-Length")
        if content_len and content_len.isdigit():
            total = int(content_len)
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            hasher.update(chunk)
            downloaded += len(chunk)
            now = time.monotonic()
            if now - last_print >= 1.5 or downloaded == total:
                pct = (downloaded / total * 100.0) if total else 0.0
                mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                print(f"\r  已下载 {mb:7.1f} / {total_mb:7.1f} MB ({pct:5.1f}%)", end="", file=sys.stderr)
                last_print = now
    print("", file=sys.stderr)

    actual_sha = hasher.hexdigest()
    if verify_sha256 and actual_sha != expected_sha:
        part.unlink(missing_ok=True)
        raise OllamaError(f"SHA-256 校验失败: 期望 {expected_sha}，实际 {actual_sha}")

    part.replace(dest)
    print(f"[v2mem] 权重校验通过: {dest}", file=sys.stderr)
    return dest


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
