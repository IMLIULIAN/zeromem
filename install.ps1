<#
.SYNOPSIS
    True one-command installer for ZeroMem (零忆) by ZeroZoneX (Windows PowerShell).
    Features:
      - Automatic GPU/VRAM detection (auto-picks Q8_0 vs Q4_K_M + safe num_ctx)
      - Optional -AutoDeps to auto-install missing Python / Ollama via winget
      - Auto-starts Ollama daemon silently if installed but stopped
      - Two install modes:
          -Mode mcp  : Plan 2 (Recommended — MCP 4 tools + 7 mainstream Agents + 100% accuracy)
          -Mode fast : Plan 1 (CLI / Python SDK single-recall — 94.3% fewer tokens, fastest)
      - Optional -ConnectAll to auto-backup & connect all detected Agents (Claude/Cursor/Codex/Antigravity/DSH)
.EXAMPLE
    # True one-liner (auto hardware quant + auto model pull + MCP configs)
    .\install.ps1 -AutoDeps -ConnectAll -LaunchHud

    # Plan 1 (Fastest CLI / Python SDK)
    .\install.ps1 -Mode fast -Source modelscope
#>
param(
    [ValidateSet("mcp", "fast")]
    [string]$Mode = $(if ($env:ZEROMEM_MODE) { $env:ZEROMEM_MODE } else { "mcp" }),
    [ValidateSet("auto", "modelscope", "hf", "ollama")]
    [string]$Source = $(if ($env:ZEROMEM_SOURCE) { $env:ZEROMEM_SOURCE } elseif ($env:V2MEM_SOURCE) { $env:V2MEM_SOURCE } else { "auto" }),
    [ValidateSet("auto", "q8_0", "q4_k_m")]
    [string]$Quant = $(if ($env:ZEROMEM_QUANT) { $env:ZEROMEM_QUANT } elseif ($env:V2MEM_QUANT) { $env:V2MEM_QUANT } else { "auto" }),
    [string]$Repo = $(if ($env:ZEROMEM_REPO) { $env:ZEROMEM_REPO } elseif ($env:V2MEM_REPO) { $env:V2MEM_REPO } else { "IMLIULIAN/zeromem" }),
    [string]$InstallDir = $(if ($env:ZEROMEM_INSTALL_DIR) { $env:ZEROMEM_INSTALL_DIR } else { Join-Path $env:USERPROFILE ".zeromem\app" }),
    [string]$PythonExe = "",
    [switch]$AutoDeps,
    [switch]$ConnectAll,
    [switch]$SkipSetup,
    [switch]$LaunchHud
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Msg) {
    Write-Host "[ZeroMem] $Msg" -ForegroundColor Cyan
}

function Find-Python39 {
    foreach ($cand in @("python", "py")) {
        $cmd = Get-Command $cand -ErrorAction SilentlyContinue
        if ($cmd) {
            $ver = & $cmd.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($ver -and ([version]$ver -ge [version]"3.9")) {
                return $cmd.Source
            }
        }
    }
    return ""
}

function Ensure-OllamaRunning {
    $ollamaCmd = Get-Command "ollama" -ErrorAction SilentlyContinue
    if (-not $ollamaCmd) {
        $defaultBin = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
        if (Test-Path $defaultBin) { $ollamaCmd = Get-Command $defaultBin -ErrorAction SilentlyContinue }
    }
    if (-not $ollamaCmd) {
        if ($AutoDeps -or $env:ZEROMEM_AUTO_DEPS -eq "1") {
            $winget = Get-Command "winget" -ErrorAction SilentlyContinue
            if (-not $winget) {
                throw "未检测到 Ollama 且本机无 winget，请手动安装：https://ollama.com/download"
            }
            Write-Step "未检测到 Ollama，正在通过 winget 自动安装..."
            & $winget.Source install --id Ollama.Ollama --exact --accept-source-agreements --accept-package-agreements
            $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
            $ollamaCmd = Get-Command "ollama" -ErrorAction SilentlyContinue
            if (-not $ollamaCmd) {
                $defaultBin = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
                if (Test-Path $defaultBin) { $ollamaCmd = Get-Command $defaultBin -ErrorAction SilentlyContinue }
            }
        } else {
            Write-Host "[ZeroMem] 提示：未检测到 Ollama。可加 -AutoDeps 自动安装，或前往 https://ollama.com/download" -ForegroundColor Yellow
            return
        }
    }
    if (-not $ollamaCmd) { return }

    # Check if Ollama HTTP API is alive; if not, silently start `ollama serve`
    $alive = $false
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
        $alive = $true
    } catch { }

    if (-not $alive) {
        Write-Step "正在后台静默启动 Ollama 服务..."
        Start-Process -FilePath $ollamaCmd.Source -ArgumentList "serve" -WindowStyle Hidden
        for ($i = 0; $i -lt 15; $i++) {
            Start-Sleep -Seconds 1
            try {
                $null = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
                $alive = $true
                break
            } catch { }
        }
        if ($alive) { Write-Step "Ollama 服务已就绪。" }
    }
}

$ModeLabel = if ($Mode -eq "fast") {
    "方案一：极速省钱版（CLI / Python SDK 单次预注）"
} else {
    "方案二：满分 Agent 版（MCP 自主拆查 + 主流 Agent 接入）"
}
Write-Step "启动安装 · $ModeLabel"

# 1. Locate (or auto-install) Python >= 3.9
if (-not $PythonExe) { $PythonExe = Find-Python39 }
if (-not $PythonExe -and ($AutoDeps -or $env:ZEROMEM_AUTO_DEPS -eq "1")) {
    $winget = Get-Command "winget" -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Step "未检测到 Python 3.9+，正在通过 winget 自动安装 Python 3.12..."
        & $winget.Source install --id Python.Python.3.12 --exact --accept-source-agreements --accept-package-agreements
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
        $PythonExe = Find-Python39
    }
}
if (-not $PythonExe) {
    throw "未检测到 Python 3.9+。可加 -AutoDeps 自动安装，或手动安装 https://www.python.org/downloads/ 并勾选 Add to PATH。"
}
Write-Step "使用 Python: $PythonExe"

# 2. Detect local repo or download archive to InstallDir
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { "" }
if ($ScriptDir -and (Test-Path (Join-Path $ScriptDir "pyproject.toml")) -and (Test-Path (Join-Path $ScriptDir "v2mem\__init__.py"))) {
    $PkgRoot = $ScriptDir
    Write-Step "检测到本地源码目录: $PkgRoot"
} else {
    $PkgRoot = $InstallDir
    $ParentDir = Split-Path -Parent $PkgRoot
    New-Item -ItemType Directory -Force -Path $ParentDir | Out-Null

    $zipUrl = "https://github.com/$Repo/archive/refs/heads/main.zip"
    $tmpZip = Join-Path ([System.IO.Path]::GetTempPath()) ("zeromem_" + [guid]::NewGuid().ToString("N") + ".zip")
    $tmpExtract = Join-Path ([System.IO.Path]::GetTempPath()) ("zeromem_ext_" + [guid]::NewGuid().ToString("N"))

    Write-Step "正在下载源码包: $zipUrl"
    Invoke-WebRequest -Uri $zipUrl -OutFile $tmpZip -UseBasicParsing
    Expand-Archive -Path $tmpZip -DestinationPath $tmpExtract -Force
    $extracted = Get-ChildItem -Path $tmpExtract -Directory | Select-Object -First 1
    if (-not $extracted) { throw "解压源码包失败。" }
    if (Test-Path $PkgRoot) { Remove-Item -Recurse -Force $PkgRoot }
    Move-Item -Path $extracted.FullName -Destination $PkgRoot
    Remove-Item -Force -ErrorAction SilentlyContinue $tmpZip
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $tmpExtract
}

# 3. Install into an isolated venv (zero pollution to global Python)
$VenvDir = Join-Path $env:USERPROFILE ".zeromem\venv"
if ($env:ZEROMEM_VENV) {
    $VenvDir = $env:ZEROMEM_VENV
} elseif ($env:V2MEM_VENV) {
    $VenvDir = $env:V2MEM_VENV
}
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"
$VenvPyw = Join-Path $VenvDir "Scripts\pythonw.exe"

if (-not (Test-Path $VenvPy)) {
    Write-Step "创建轻量运行环境: $VenvDir"
    & $PythonExe -m venv $VenvDir
}

Write-Step "安装 zmem / zmem-hud（兼容 v2mem，零第三方依赖）..."
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $VenvPy -m pip install --no-deps --no-build-isolation --quiet --upgrade $PkgRoot *> $null
$pipRc = $LASTEXITCODE
$ErrorActionPreference = $prevEap

if ($pipRc -ne 0) {
    $sitePkgs = (& $VenvPy -c "import site; print(site.getsitepackages()[-1])").Trim()
    [System.IO.File]::WriteAllText((Join-Path $sitePkgs "zeromem.pth"), "$PkgRoot`n", [System.Text.UTF8Encoding]::new($false))
    $scriptsDir = Join-Path $VenvDir "Scripts"
    foreach ($cliName in @("zmem", "v2mem")) {
        Set-Content -Path (Join-Path $scriptsDir "$cliName.cmd") -Value "@echo off`r`n`"$VenvPy`" -m zeromem %*`r`n" -Encoding ASCII
    }
    foreach ($hudName in @("zmem-hud", "v2mem-hud")) {
        Set-Content -Path (Join-Path $scriptsDir "$hudName.cmd") -Value "@echo off`r`nstart `"`" `"$VenvPyw`" -m zeromem ui %*`r`n" -Encoding ASCII
    }
}

$BinDir = Join-Path $VenvDir "Scripts"
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$skipPath = ($env:ZEROMEM_NO_PATH -or $env:V2MEM_NO_PATH)
if (-not $skipPath -and ($userPath -notlike "*$BinDir*")) {
    $newPath = if ($userPath) { "$userPath;$BinDir" } else { $BinDir }
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    $env:Path = "$env:Path;$BinDir"
    Write-Step "已将命令目录写入用户 PATH: $BinDir"
}

# 4. Ensure Ollama is available & run hardware-aware model setup
if (-not $SkipSetup) {
    Ensure-OllamaRunning
    Write-Step "硬件探测、检查/拉取 V2 模型并生成 [$Mode] 方案配置..."
    $setupArgs = @("-m", "zeromem", "setup", "--mode", $Mode, "--source", $Source, "--quant", $Quant)
    if ($ConnectAll -or $env:ZEROMEM_CONNECT_ALL -eq "1") { $setupArgs += "--connect-all" }
    & $VenvPy @setupArgs
} else {
    Write-Step "已跳过模型检查，正在导出 [$Mode] 方案配置模板..."
    & $VenvPy -m zeromem config --mode $Mode | Out-Null
}

# 5. Optional: launch windowless status pill & workbench
if ($LaunchHud -and (Test-Path $VenvPyw)) {
    Start-Process -FilePath $VenvPyw -ArgumentList "-m","zeromem","ui" -WorkingDirectory $PkgRoot
    Write-Step "已启动 ZeroMem 桌面状态胶囊与记忆工作台。"
}

Write-Host "`n[ZeroMem · 零忆] 安装完成！（$ModeLabel）" -ForegroundColor Green
Write-Host "  zmem doctor                    # 一键体检硬件显存、模型与 Agent 接入状态"
Write-Host "  zmem connect --all             # 一键自动接入本机主流 AI Agent（自动备份）"
Write-Host "  zmem-hud                       # 启动桌面悬浮胶囊（点 🔍 打开可视化记忆工作台）"
Write-Host "  zmem add `"要记住的内容`"         # 写入记忆（本地 4B 写前验收 + 密钥脱敏）"
Write-Host "  zmem search `"关键词`"           # 毫秒级检索 + 原文证据溯源"
