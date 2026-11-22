#!/usr/bin/env bash
# One-command installer for ZeroMem (零忆) by ZeroZoneX (macOS / Linux)
#
# Features:
#   - Automatic GPU/VRAM & Apple Silicon unified memory detection (--quant auto)
#   - Optional --auto-deps to auto-install Ollama (brew on macOS / official script on Linux)
#   - Auto-starts Ollama daemon in background if installed but stopped
#   - Two installation modes:
#       --mode mcp  : Plan 2 (Default — 4-tool MCP + 7 mainstream Agents)
#       --mode fast : Plan 1 (CLI / Python SDK single-recall — low cost & latency)
#   - Optional --connect-all to auto-backup & connect detected Agents
#
# Usage:
#   ./install.sh --mode mcp --auto-deps --connect-all
#   ./install.sh --mode fast --source modelscope

set -euo pipefail

MODE="${ZEROMEM_MODE:-mcp}"
REPO="${ZEROMEM_REPO:-${V2MEM_REPO:-IMLIULIAN/zeromem}}"
SOURCE="${ZEROMEM_SOURCE:-${V2MEM_SOURCE:-auto}}"
QUANT="${ZEROMEM_QUANT:-${V2MEM_QUANT:-auto}}"
SKIP_SETUP="${ZEROMEM_SKIP_SETUP:-${V2MEM_SKIP_SETUP:-0}}"
AUTO_DEPS="${ZEROMEM_AUTO_DEPS:-0}"
CONNECT_ALL="${ZEROMEM_CONNECT_ALL:-0}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode)
      MODE="$2"; shift 2 ;;
    --source)
      SOURCE="$2"; shift 2 ;;
    --quant)
      QUANT="$2"; shift 2 ;;
    --auto-deps)
      AUTO_DEPS="1"; shift ;;
    --connect-all)
      CONNECT_ALL="1"; shift ;;
    --skip-setup)
      SKIP_SETUP="1"; shift ;;
    -h|--help)
      echo "ZeroMem (零忆) Installer"
      echo "  --mode fast|mcp             选择方案：fast（轻量省钱 CLI/SDK）或 mcp（Agent MCP，默认）"
      echo "  --source auto|modelscope|hf|ollama"
      echo "  --quant auto|q8_0|q4_k_m    量化档位（默认 auto 按显存/内存自动选）"
      echo "  --auto-deps                 缺少 Ollama 时自动安装并后台拉起服务"
      echo "  --connect-all               自动备份并一键接入本机检测到的主流 Agent"
      echo "  --skip-setup                跳过 Ollama 模型检查"
      exit 0 ;;
    *)
      echo "[ZeroMem] 未知参数: $1" >&2; exit 2 ;;
  esac
done

if [ "${MODE}" != "fast" ] && [ "${MODE}" != "mcp" ]; then
  echo "[ZeroMem] 错误：--mode 仅支持 fast 或 mcp（收到: ${MODE}）" >&2
  exit 2
fi

BASE_DIR="${HOME}/.zeromem"
INSTALL_DIR="${ZEROMEM_INSTALL_DIR:-${V2MEM_INSTALL_DIR:-${BASE_DIR}/app}}"
VENV_DIR="${ZEROMEM_VENV:-${V2MEM_VENV:-${BASE_DIR}/venv}}"
BIN_DIR="${HOME}/.local/bin"

log() { printf "\033[36m[ZeroMem]\033[0m %s\n" "$1"; }

ensure_ollama_running() {
  if ! command -v ollama >/dev/null 2>&1; then
    if [ "${AUTO_DEPS}" = "1" ]; then
      if [ "$(uname -s)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
        log "未检测到 Ollama，正在通过 Homebrew 自动安装..."
        brew install ollama
      elif [ "$(uname -s)" = "Linux" ]; then
        log "未检测到 Ollama，正在通过官方脚本自动安装..."
        curl -fsSL https://ollama.com/install.sh | sh
      else
        echo "[ZeroMem] 请先手动安装 Ollama: https://ollama.com/download" >&2
        return
      fi
    else
      log "提示：未检测到 Ollama。可追加 --auto-deps 自动安装，或访问 https://ollama.com/download"
      return
    fi
  fi

  if command -v ollama >/dev/null 2>&1; then
    if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      log "正在后台启动 Ollama 服务..."
      nohup ollama serve >/dev/null 2>&1 &
      for _ in $(seq 1 15); do
        sleep 1
        if curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
          log "Ollama 服务已就绪。"
          break
        fi
      done
    fi
  fi
}

if [ "${MODE}" = "fast" ]; then
  MODE_LABEL="方案一：轻量省钱版（CLI / Python SDK 单次预注）"
else
  MODE_LABEL="方案二：Agent MCP 版（自主拆查 + 主流 Agent 接入）"
fi
log "启动安装 · ${MODE_LABEL}"

# 1. Locate Python >= 3.9
PY_BIN="${ZEROMEM_PYTHON:-${V2MEM_PYTHON:-}}"
if [ -z "${PY_BIN}" ]; then
  for cand in python3 python; do
    if command -v "${cand}" >/dev/null 2>&1; then
      if "${cand}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
        PY_BIN="$(command -v "${cand}")"
        break
      fi
    fi
  done
fi
if [ -z "${PY_BIN}" ]; then
  echo "[ZeroMem] 错误：未检测到 Python 3.9+，请先安装 Python 3（macOS: brew install python；Ubuntu: sudo apt install python3 python3-venv）。" >&2
  exit 1
fi
log "使用 Python: ${PY_BIN}"

# 2. Detect local repo or download tarball
SCRIPT_DIR=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

if [ -n "${SCRIPT_DIR}" ] && [ -f "${SCRIPT_DIR}/pyproject.toml" ] && [ -f "${SCRIPT_DIR}/v2mem/__init__.py" ]; then
  PKG_ROOT="${SCRIPT_DIR}"
  log "检测到本地源码目录: ${PKG_ROOT}"
else
  mkdir -p "${BASE_DIR}"
  TAR_URL="https://github.com/${REPO}/archive/refs/heads/main.tar.gz"
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "${TMP_DIR}"' EXIT
  log "正在下载源码包: ${TAR_URL}"
  curl -fsSL "${TAR_URL}" | tar -xz -C "${TMP_DIR}"
  EXTRACTED="$(find "${TMP_DIR}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  rm -rf "${INSTALL_DIR}"
  mv "${EXTRACTED}" "${INSTALL_DIR}"
  PKG_ROOT="${INSTALL_DIR}"
fi

# 3. Install into isolated venv (zero third-party dependencies)
if [ ! -x "${VENV_DIR}/bin/python" ]; then
  log "创建轻量运行环境: ${VENV_DIR}"
  "${PY_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PY="${VENV_DIR}/bin/python"
log "安装 zmem / zmem-hud（兼容 v2mem，零第三方依赖）..."
if ! "${VENV_PY}" -m pip install --no-deps --no-build-isolation --quiet --upgrade "${PKG_ROOT}" 2>/dev/null; then
  SITE_PKGS="$("${VENV_PY}" -c 'import site; print(site.getsitepackages()[0])')"
  printf "%s\n" "${PKG_ROOT}" > "${SITE_PKGS}/zeromem.pth"
  for cli_name in zmem v2mem; do
    cat > "${VENV_DIR}/bin/${cli_name}" <<EOF
#!/usr/bin/env bash
exec "${VENV_PY}" -m zeromem "\$@"
EOF
    chmod +x "${VENV_DIR}/bin/${cli_name}"
  done
  for hud_name in zmem-hud v2mem-hud; do
    cat > "${VENV_DIR}/bin/${hud_name}" <<EOF
#!/usr/bin/env bash
exec "${VENV_PY}" -m zeromem ui "\$@"
EOF
    chmod +x "${VENV_DIR}/bin/${hud_name}"
  done
fi

mkdir -p "${BIN_DIR}"
for link_name in zmem zmem-hud v2mem v2mem-hud; do
  if [ -e "${VENV_DIR}/bin/${link_name}" ]; then
    ln -sf "${VENV_DIR}/bin/${link_name}" "${BIN_DIR}/${link_name}"
  fi
done
log "已链接命令到 ${BIN_DIR}/zmem、${BIN_DIR}/zmem-hud（兼容 v2mem）"

if [ "${SKIP_SETUP}" != "1" ]; then
  ensure_ollama_running
  log "硬件探测、检查/拉取 V2 模型并生成 [${MODE}] 方案配置..."
  EXTRA_FLAG=""
  if [ "${CONNECT_ALL}" = "1" ]; then EXTRA_FLAG="--connect-all"; fi
  # shellcheck disable=SC2086
  "${VENV_PY}" -m zeromem setup --mode "${MODE}" --source "${SOURCE}" --quant "${QUANT}" ${EXTRA_FLAG}
else
  log "已跳过模型检查，正在导出 [${MODE}] 方案配置模板..."
  "${VENV_PY}" -m zeromem config --mode "${MODE}" >/dev/null
fi

printf "\n\033[32m[ZeroMem · 零忆] 安装完成！（%s）\033[0m\n" "${MODE_LABEL}"
printf "  zmem doctor             # 一键体检硬件显存、模型与 Agent 接入状态\n"
printf "  zmem connect --all      # 一键自动接入本机主流 AI Agent（自动备份）\n"
printf "  zmem-hud                # 启动桌面悬浮胶囊（点 🔍 打开可视化记忆工作台）\n"
printf "  zmem add \"要记住的内容\"\n"
printf "  zmem search \"关键词\"\n"
