# voice-lab 启动器（文字版）：实时替换、试听页面，以及运行环境的检查、配置与修复、快照和恢复。
# 双击仓库根目录的 一键启动.cmd 打开的是窗口版（scripts/ui_server.py，背后调用的就是这个脚本）；
# 一键启动.cmd -Menu 打开这个文字菜单。也可以带参数直接做一件事：
#   .\scripts\launch.ps1 -Live [-Voice <声线>] [-NoRoute] [-Play <设备名>]   启动前检查（残留进程、STOP 文件、内存），再开实时替换；
#                                                  游戏的输出自动切到虚拟设备、退出时切回（-NoRoute：不自动切换）
#   .\scripts\launch.ps1 -GameDefault               把游戏的输出切回 Windows 默认（游戏没开时，等它下次启动）
#   .\scripts\launch.ps1 -Pages                    打开试听页面（页面服务没开就先开）
#   .\scripts\launch.ps1 -Check [-Deep] [-JsonOut <文件>]   环境检查；-Deep 还会在每个环境里导入主要模块、确认显卡可用
#                                                  （约 1 分钟）；-JsonOut 把结果另存为 JSON（窗口版用）
#   .\scripts\launch.ps1 -Shortcut                 在桌面创建快捷方式
#   .\scripts\launch.ps1 -Setup [-All] [-Yes]      配置与修复：补齐缺的工具、仓库、Python 环境和模型（-All 连可选的一起）
#   .\scripts\launch.ps1 -Snapshot [-Label 说明]   保存环境快照：各 Python 环境的精确包版本、第三方仓库的 commit
#   .\scripts\launch.ps1 -Restore [-From <快照名>] [-Envs tools,applio] [-Yes]   把环境恢复到快照（默认最新一份）
#                                                  加 -DryRun 只列出 uv 会改哪些包，不动环境
# 快照存在 env-snapshots/<时间>/，格式与复现包的 env/ 相同；复现包和随代码附带的依赖清单也可以直接作为恢复来源。
# 恢复只改有差异的包；Python 本身坏了或版本不对时整个重建，旧环境改名保留，新环境可用后再问你删不删。
# 恢复前会自动把当前状态存一份快照，恢复错了可以再恢复回去。
# 只动本仓库的环境、第三方仓库和缓存；游戏的输出设备由实时工具自动切换和切回（app_route.py）。
# 其他 Windows 设置（页面文件）、虚拟声卡只检查和提示，由你来改。
param(
  [switch]$Live, [switch]$Pages, [switch]$Check, [switch]$Deep, [switch]$Setup, [switch]$Snapshot, [switch]$Restore,
  [string]$Voice = '', [string]$From = '', [string[]]$Envs = @(), [string]$Label = '', [switch]$All, [switch]$Yes,
  [switch]$DryRun, [switch]$NoRoute, [switch]$GameDefault, [string]$JsonOut = '', [switch]$Shortcut,
  [string]$Play = '', [switch]$Menu
)
. "$PSScriptRoot\env.ps1"
try { $Host.UI.RawUI.WindowTitle = 'voice-lab 启动器' } catch { }
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }   # Python prints UTF-8 (env.ps1): read it as such
New-Item -ItemType Directory -Force "$VL\cache\tmp" | Out-Null       # TEMP points here (env.ps1)
# with FORCE_COLOR set, uv wraps package names in colour codes and the package lists cannot be compared
$env:NO_COLOR = '1'
Remove-Item Env:FORCE_COLOR, Env:CLICOLOR_FORCE -ErrorAction SilentlyContinue
# downloads (uv, git, curl, Hugging Face) take the Windows proxy when one is on and the environment names none
if (-not $env:HTTPS_PROXY -and -not $env:HTTP_PROXY) {
  $is = Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' -ErrorAction SilentlyContinue
  if ($is -and $is.ProxyEnable -eq 1 -and $is.ProxyServer) {
    $px = "$($is.ProxyServer)"
    if ($px -match '(?:^|;)https=([^;]+)') { $px = $Matches[1] } elseif ($px -match '(?:^|;)http=([^;]+)') { $px = $Matches[1] }
    if ($px -notmatch '[=;]') {
      if ($px -notmatch '^\w+://') { $px = "http://$px" }
      $env:HTTP_PROXY = $px; $env:HTTPS_PROXY = $px
      if (-not $env:NO_PROXY) { $env:NO_PROXY = 'localhost,127.0.0.1,::1' }
    }
  }
}

$T = "$VL\envs\tools\Scripts\python.exe"
$SNAP = "$VL\env-snapshots"
$SETTINGS = "$VL\logs\launcher.json"
$STOP = "$VL\logs\live\STOP"
$ROUTE = "$(if ($env:VL_ROUTE_DIR) { $env:VL_ROUTE_DIR } else { "$VL\logs\live" })\route.json"   # app_route.py's state
$VIRTUAL = 'Steam Streaming|CABLE|VoiceMeeter|VB-Audio|Virtual'
$PORT = 8765
$UTF8 = New-Object System.Text.UTF8Encoding $false

# The Python environments: where, whether the pipeline needs it, what for, the Python a fresh one gets (a snapshot's
# own version wins), and what -Deep imports to prove it works
$PYENVS = [ordered]@{
  tools    = @{ Dir = "$VL\envs\tools";              Need = $true;  Py = '3.11'; Use = '处理、评估、实时工具、页面服务'; Deep = 'soundcard,pycaw,faster_whisper,audio_separator' }
  indextts = @{ Dir = "$env:VL_CACHE\envs\indextts"; Need = $true;  Py = '3.11'; Use = 'IndexTTS 重配';                Deep = 'indextts' }
  applio   = @{ Dir = "$env:VL_CACHE\envs\applio";   Need = $true;  Py = '3.12'; Use = 'RVC 训练和转换';               Deep = 'torchfcpe' }
  audiosr  = @{ Dir = "$VL\envs\audiosr";            Need = $true;  Py = '3.10'; Use = '超分到 48 kHz';                Deep = 'audiosr' }
  seedvc   = @{ Dir = "$VL\envs\seedvc";             Need = $true;  Py = '3.10'; Use = '情绪参考、精修候选';            Deep = 'funasr' }
  vevo2    = @{ Dir = "$VL\envs\vevo2";              Need = $false; Py = '3.10'; Use = '只在早期实验中用到';            Deep = 'librosa' }
}
# Third-party code at the commits the results were made with (a snapshot's third_party.json wins)
$REPOS = [ordered]@{
  'index-tts' = @{ Dir = "$VL\repos\index-tts";  Need = $true;  Url = 'https://github.com/index-tts/index-tts.git'; Commit = 'd9e41aac89fd00b3d71497fddb287b7f24613712' }
  'seed-vc'   = @{ Dir = "$VL\repos\seed-vc";    Need = $true;  Url = 'https://github.com/Plachtaa/seed-vc.git';    Commit = '51383efd921027683c89e5348211d93ff12ac2a8' }
  'Applio'    = @{ Dir = "$env:VL_CACHE\Applio"; Need = $true;  Url = 'https://github.com/IAHispano/Applio.git';    Commit = '324f4d89c3e0e8dc0e555a3d53784e3282c16e09' }
  'Amphion'   = @{ Dir = "$VL\repos\Amphion";    Need = $false; Url = 'https://github.com/open-mmlab/Amphion.git';  Commit = '26f6883110181f1dbfe95c70a7c7dbaf4de5f42a' }
}
$PATCH = @("$VL\patches\applio-train-workers.patch", "$VL\release_tools\public\patches\applio-train-workers.patch") |
  Where-Object { Test-Path $_ } | Select-Object -First 1
# Model files. Fix: tools = prefetch_models.py, indextts = the IndexTTS-2.5 checkpoints (Hugging Face),
# applio = Applio's own prerequisites download, auto = fetched by the step that uses it
$MODELS = @(
  @{ What = 'Whisper large-v3（转写）';     Path = "$VL\cache\whisper\models--Systran--faster-whisper-large-v3"; Fix = 'tools' }
  @{ What = '人声分离（Mel-Band RoFormer）'; Path = "$VL\cache\separator\vocals_mel_band_roformer.ckpt"; Fix = 'tools' }
  @{ What = '去混响';                       Path = "$VL\cache\separator\dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt"; Fix = 'tools' }
  @{ What = '降噪';                         Path = "$VL\cache\separator\UVR-DeNoise.pth"; Fix = 'tools' }
  @{ What = 'ECAPA 声纹';                   Path = "$VL\cache\ecapa\embedding_model.ckpt"; Fix = 'tools' }
  @{ What = 'IndexTTS 模型';                Path = "$VL\repos\index-tts\checkpoints\gpt.pth"; Fix = 'indextts' }
  @{ What = 'Applio 预训练（HiFi-GAN）';    Path = "$env:VL_CACHE\Applio\rvc\models\pretraineds\hifi-gan\f0G40k.pth"; Fix = 'applio' }
  @{ What = 'Applio 音高（RMVPE）';         Path = "$env:VL_CACHE\Applio\rvc\models\predictors\rmvpe.pt"; Fix = 'applio' }
  @{ What = 'Applio ContentVec';            Path = "$env:VL_CACHE\Applio\rvc\models\embedders\contentvec\pytorch_model.bin"; Fix = 'applio' }
  @{ What = '情绪识别（emotion2vec）';      Path = "$VL\cache\emotion2vec_plus_large"; Fix = 'auto' }
  @{ What = 'AudioSR';                      Path = "$VL\cache\hf\hub\models--haoheliu--audiosr_speech"; Fix = 'auto' }
)

# ---------------------------------------------------------------- small helpers
function Say([string]$t, [string]$c = 'Gray') { Write-Host $t -ForegroundColor $c }
function Head([string]$t) { Write-Host ''; Write-Host "== $t" -ForegroundColor Cyan }
function Ask([string]$q) {
  if ($Yes) { Say "$q 是（-Yes）" DarkGray; return $true }
  return ("$(Read-Host "$q [y/N]")" -match '^\s*(y|yes|是)\s*$')
}
function Read-Json([string]$p) {
  if (Test-Path -LiteralPath $p) { try { return (Get-Content -LiteralPath $p -Raw -Encoding UTF8 | ConvertFrom-Json) } catch { } }
  $null
}
function Write-Utf8([string]$p, $lines) { [IO.File]::WriteAllLines($p, [string[]]@($lines), $UTF8) }
function Get-FreeGB([string]$p) { $d = Get-PSDrive $p.Substring(0, 1) -ErrorAction SilentlyContinue; if ($d) { $d.Free / 1GB } else { 0 } }
function Get-DirGB([string]$p) { (Get-ChildItem -LiteralPath $p -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1GB }
function Get-Head([string]$dir) { $h = & git -C $dir rev-parse HEAD 2>$null; if ($LASTEXITCODE) { $null } else { "$h".Trim() } }
function Short([string]$c) { if ($c -and $c.Length -ge 8) { $c.Substring(0, 8) } else { "$c" } }
function Has([string]$cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

# ---------------------------------------------------------------- voices
function Get-Voices {
  # a voice = data/<voice>/voice.json, or finished lines in exp/all[_<voice>]/final_48k
  foreach ($d in Get-ChildItem "$VL\data" -Directory -ErrorAction SilentlyContinue) {
    $fin = "$(Get-ExpDir 'all' $d.Name)\final_48k"
    if (-not (Test-Path "$($d.FullName)\voice.json") -and -not (Test-Path $fin)) { continue }
    $cfg = Get-VoiceCfg $d.Name
    $n = if (Test-Path $fin) { @(Get-ChildItem $fin -Filter *.wav).Count } else { 0 }
    [pscustomobject]@{ Name = $d.Name; Label = "$($cfg.label)"; Lines = $n; Model = $(if ($cfg.rvc_model) { "$($cfg.rvc_model)" } else { "$($d.Name)_v1" }) }
  }
}
function Get-DefaultVoice {
  $s = Read-Json $SETTINGS
  if ($s -and $s.voice -and (Test-Path "$VL\data\$($s.voice)")) { return "$($s.voice)" }
  if (Test-Path "$PSScriptRoot\live.ps1") {                     # else the live tool's own default
    $m = Select-String -LiteralPath "$PSScriptRoot\live.ps1" -Pattern "\[string\]\`$Voice\s*=\s*'([^']+)'" | Select-Object -First 1
    if ($m) { return $m.Matches[0].Groups[1].Value }
  }
  "$(@(Get-Voices) | Select-Object -First 1 -ExpandProperty Name)"
}
function Set-DefaultVoice([string]$v) {
  New-Item -ItemType Directory -Force (Split-Path $SETTINGS) | Out-Null
  [IO.File]::WriteAllText($SETTINGS, (@{ voice = $v } | ConvertTo-Json), $UTF8)
}
function Select-Voice([string]$cur) {
  $vs = @(Get-Voices)
  if (-not $vs) { Say '还没有任何声线（data/<声线>/voice.json）' Yellow; return $cur }
  for ($i = 0; $i -lt $vs.Count; $i++) {
    Say ("  {0}  {1,-12} {2}，成品 {3} 句{4}" -f ($i + 1), $vs[$i].Name, $vs[$i].Label, $vs[$i].Lines, $(if ($vs[$i].Name -eq $cur) { '   <- 当前' } else { '' }))
  }
  $c = "$(Read-Host '选哪一个（回车不变）')".Trim()
  if ($c -match '^\d+$' -and [int]$c -ge 1 -and [int]$c -le $vs.Count) { $cur = $vs[[int]$c - 1].Name; Set-DefaultVoice $cur; Say "已切换到 $cur（下次打开也用它）" Green }
  $cur
}

# ---------------------------------------------------------------- processes, page server
function Get-Tree($all, [int[]]$roots) {
  # the given processes and everything they started (a venv's python.exe starts the real interpreter as a child)
  $ids = @($roots)
  do {
    $more = @($all | Where-Object { $ids -contains $_.ParentProcessId -and $ids -notcontains $_.ProcessId } | ForEach-Object { [int]$_.ProcessId })
    $ids += $more
  } while ($more.Count)
  @($all | Where-Object { $ids -contains $_.ProcessId })
}
function Get-LiveProcs {
  $all = @(Get-CimInstance Win32_Process)
  $roots = @($all | Where-Object { "$($_.CommandLine)" -like '*live_replace.py*' } | ForEach-Object { [int]$_.ProcessId })
  if ($roots) { Get-Tree $all $roots } else { @() }
}
function Get-EnvProcs([string]$dir) {
  $all = @(Get-CimInstance Win32_Process)
  $roots = @($all | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith("$dir\", [StringComparison]::OrdinalIgnoreCase) } |
    ForEach-Object { [int]$_.ProcessId })
  if ($roots) { Get-Tree $all $roots } else { @() }
}
function Get-PageServer { Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 }
function Start-PageServer {
  if (Get-PageServer) { return $true }
  if (-not (Test-Path $T)) { Say 'tools 环境还没装好，开不了页面服务（先选“环境配置与修复”）' Red; return $false }
  Start-Process -FilePath $T -ArgumentList @('-m', 'http.server', "$PORT", '--bind', '127.0.0.1') -WorkingDirectory $VL -WindowStyle Hidden
  for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 250
    if (Get-PageServer) { Say "页面服务已启动：http://localhost:$PORT/" Green; return $true }
  }
  Say '页面服务没有起来' Red
  $false
}

# ---------------------------------------------------------------- the game's output (app_route.py)
function Invoke-Route([string]$cmd) {
  # app_route.py status | restore | reset -> its JSON answer
  if (-not (Test-Path $T)) { return $null }
  $out = & $T "$PSScriptRoot\app_route.py" $cmd 2>$null
  try { return ("$out" | ConvertFrom-Json) } catch { return $null }
}
function Format-Back([string]$b) { if (-not $b -or $b -eq 'Windows default') { 'Windows 默认设备' } else { $b } }
function Resume-Route {
  # a switch-back the live tool could not finish: it was closed hard, or the PC restarted while the watcher waited
  $st = Read-Json $ROUTE
  if (-not $st -or -not $st.active -or @(Get-LiveProcs).Count) { return }
  $s = Invoke-Route 'status'
  if (-not $s -or $s.live -or ($s.pending -and $s.watcher)) { return }   # the watcher already waits for the game
  $r = Invoke-Route 'restore'
  if (-not $r) { return }
  switch ($r.result) {
    'restored' { Say "上次没切回的游戏声音已切回：$(Format-Back $r.back)" Green }
    'pending' { Say "游戏声音还没切回（工具被强制关闭过，或电脑重启过）：已重新开始等待，游戏下次启动时切回 $(Format-Back $r.back)" Yellow }
    'failed' { Say '游戏声音没能切回：选菜单 9，或在“应用音量和设备首选项”里手动改回' Red }
  }
}
function Set-GameDefault {
  $r = Invoke-Route 'reset'
  if (-not $r) { Say 'tools 环境没有装好，做不了' Red; return }
  switch ($r.result) {
    'restored' { Say '游戏的输出已切回 Windows 默认设备' Green }
    'pending' { Say '游戏没开：它下次启动时切回 Windows 默认设备（后台小程序在等，切完就退出）' Green }
    default { Say '没有切成功：在“应用音量和设备首选项”里手动改' Red }
  }
}

# ---------------------------------------------------------------- requirement lists, snapshots
function Get-Sources {
  # where an environment can be restored from: own snapshots (newest first), the reproducible packages' env/, the lists
  # shipped with the code; snapshots saved automatically before a restore come last (never the default)
  $snaps = @(Get-ChildItem $SNAP -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | ForEach-Object {
      $m = Read-Json "$($_.FullName)\snapshot.json"
      [pscustomobject]@{ Name = $_.Name; Dir = $_.FullName; Kind = '快照'; Note = $(if ($m) { "$($m.label)" } else { '' }); Auto = [bool]($m -and $m.auto) }
    })
  $out = @($snaps | Where-Object { -not $_.Auto })
  foreach ($d in @(Get-ChildItem "$VL\release" -Directory -Filter 'repro-*' -ErrorAction SilentlyContinue | Sort-Object Name -Descending)) {
    if (Test-Path "$($d.FullName)\env\requirements-tools.txt") {
      $out += [pscustomobject]@{ Name = $d.Name; Dir = "$($d.FullName)\env"; Kind = '复现包'; Note = ''; Auto = $false }
    }
  }
  foreach ($d in @("$VL\requirements", "$VL\release_tools\public\requirements")) {
    if (Test-Path "$d\requirements-tools.txt") {
      $out += [pscustomobject]@{ Name = 'requirements'; Dir = $d; Kind = '依赖清单'; Note = '随代码附带'; Auto = $false }
      break
    }
  }
  $out + @($snaps | Where-Object { $_.Auto })
}
function Get-SourceFor([string]$n) {
  # the first (non-automatic) source that has this environment
  @(Get-Sources) | Where-Object { -not $_.Auto -and (Test-Path "$($_.Dir)\requirements-$n.txt") } | Select-Object -First 1
}
function Get-SourceList($src, [string]$n) {
  if (-not $src) { return $null }
  $f = "$($src.Dir)\requirements-$n.txt"
  if (-not (Test-Path $f)) { return $null }
  $lines = @(Get-Content $f -Encoding UTF8)
  $ver = ''
  if ($lines.Count -and $lines[0] -match 'Python (\d+\.\d+\.\d+)') { $ver = $Matches[1] }
  [pscustomobject]@{ File = $f; Python = $ver; Lines = @($lines | ForEach-Object { $_.Trim() } | Where-Object { $_ -and -not $_.StartsWith('#') }) }
}
function Get-ThirdParty($src) {
  $m = @{}
  foreach ($n in $REPOS.Keys) { $m[$n] = [pscustomobject]@{ url = $REPOS[$n].Url; commit = $REPOS[$n].Commit } }
  if ($src) {
    $j = Read-Json "$($src.Dir)\third_party.json"
    if ($j) { foreach ($p in $j.PSObject.Properties) { if ($m.ContainsKey($p.Name)) { $m[$p.Name] = $p.Value } } }
  }
  $m
}
function ConvertTo-PkgMap($lines) {
  # name -> version (or URL); editable local checkouts by folder name
  $m = @{}
  foreach ($l in @($lines)) {
    $l = "$l".Trim()
    if (-not $l -or $l.StartsWith('#') -or $l.StartsWith('--')) { continue }
    if ($l -match '^-e\s+(.+)$') { $m['-e ' + ([uri]::UnescapeDataString($Matches[1]) -split '[\\/]')[-1]] = '(editable)'; continue }
    if ($l -match '^([A-Za-z0-9_.\-]+)\s*(==|@)\s*(.+)$') { $m[($Matches[1].ToLower() -replace '[_.]', '-')] = $Matches[3].Trim() }
  }
  $m
}
function Compare-Pkgs($want, $have) {
  $changed = @(); $missing = @(); $extra = @()
  $editables = @($want.Keys | Where-Object { $_ -like '-e *' }).Count
  foreach ($k in $want.Keys) {
    if (-not $have.ContainsKey($k)) { $missing += $k; continue }
    $w = $want[$k]; $h = $have[$k]
    # "0.26.0" in the list is the same release as an installed "0.26.0+cu128"; the reverse (a CPU build) is not
    if ($w -ne $h -and -not ($w -notmatch '\+' -and $h -like "$w+*")) { $changed += "$k $h（应为 $w）" }
  }
  foreach ($k in $have.Keys) {
    if ($want.ContainsKey($k)) { continue }
    if ($k -in 'pip', 'setuptools', 'wheel') { continue }       # lists made by pip freeze leave the installers out
    if ($k -like '-e *' -and -not $editables) { continue }      # shareable lists leave the local checkouts out
    $extra += $k
  }
  [pscustomobject]@{ Changed = $changed; Missing = $missing; Extra = $extra; Count = $changed.Count + $missing.Count + $extra.Count }
}
function Format-Diff($d, [int]$max = 5) {
  $parts = @()
  if ($d.Changed.Count) { $parts += "版本不同 $($d.Changed.Count) 个" }
  if ($d.Missing.Count) { $parts += "缺 $($d.Missing.Count) 个" }
  if ($d.Extra.Count) { $parts += "多出 $($d.Extra.Count) 个" }
  $items = @($d.Changed) + @($d.Missing | ForEach-Object { "缺 $_" }) + @($d.Extra | ForEach-Object { "多 $_" })
  $shown = ($items | Select-Object -First $max) -join '；'
  if ($items.Count -gt $max) { $shown += " 等" }
  "$($parts -join '、')：$shown"
}
function Get-EnvState([string]$n) {
  $dir = $PYENVS[$n].Dir
  $py = "$dir\Scripts\python.exe"
  $s = [pscustomobject]@{ Name = $n; Dir = $dir; Py = $py; Exists = (Test-Path $py); Works = $false; Version = ''; Freeze = @() }
  if ($s.Exists) {
    $v = & $py -c "import sys; print('%d.%d.%d' % sys.version_info[:3])" 2>$null
    if ($LASTEXITCODE -eq 0 -and $v) { $s.Works = $true; $s.Version = "$v".Trim() }
    if ($s.Works -and (Has uv)) {
      $s.Freeze = @(& uv pip freeze --python $py 2>$null | ForEach-Object { ("$_" -replace '\x1b\[[0-9;]*[A-Za-z]', '').Trim() } | Where-Object { $_ })
    }
  }
  $s
}
function Get-EnvPlan([string]$n, $src) {
  $st = Get-EnvState $n
  $list = Get-SourceList $src $n
  $p = [pscustomobject]@{ Name = $n; State = $st; List = $list; Action = 'none'; Why = '包和 Python 版本都相同'; Diff = $null }
  $mm = { param($v) (("$v" -split '\.')[0..1]) -join '.' }
  if (-not $list) { $p.Action = 'skip'; $p.Why = '来源里没有这个环境'; return $p }
  if (-not $st.Exists) {
    $partial = (Test-Path $st.Dir) -and (Get-ChildItem -LiteralPath $st.Dir -Force | Select-Object -First 1)
    $p.Action = $(if ($partial) { 'rebuild' } else { 'create' })
    $p.Why = $(if ($partial) { '环境目录不完整：重建' } else { "新建（Python $(if ($list.Python) { $list.Python } else { $PYENVS[$n].Py })）" })
    return $p
  }
  if (-not $st.Works) { $p.Action = 'rebuild'; $p.Why = 'Python 无法运行（它依赖的基础 Python 可能被删了）：重建'; return $p }
  if ($list.Python -and (& $mm $st.Version) -ne (& $mm $list.Python)) {
    $p.Action = 'rebuild'; $p.Why = "Python $($st.Version) 与来源的 $($list.Python) 不同：重建"; return $p
  }
  if (-not $st.Freeze) { $p.Action = 'sync'; $p.Why = '读不出已装的包：按来源同步'; return $p }
  $p.Diff = Compare-Pkgs (ConvertTo-PkgMap $list.Lines) (ConvertTo-PkgMap $st.Freeze)
  if ($p.Diff.Count) { $p.Action = 'sync'; $p.Why = Format-Diff $p.Diff }
  $p
}
function Write-SyncList($p) {
  $lines = New-Object System.Collections.Generic.List[string]
  foreach ($l in $p.List.Lines) {
    if ($l -match '^-e\s+(.+)$') {                              # an editable checkout: this machine's copy in repos/
      $leaf = ([uri]::UnescapeDataString($Matches[1]) -split '[\\/]')[-1]
      if (Test-Path "$VL\repos\$leaf") { $lines.Add("-e $VL\repos\$leaf") } else { Say "  跳过本地包 $leaf（没有 repos\$leaf）" Yellow }
    } elseif ($l -match '@\s*file:') { Say "  跳过本机文件包：$l" Yellow }
    else { $lines.Add($l) }
  }
  if ($p.Name -eq 'indextts' -and -not ($lines | Where-Object { $_ -like '-e *' }) -and (Test-Path "$VL\repos\index-tts")) {
    $lines.Add("-e $VL\repos\index-tts")                         # shareable lists leave IndexTTS itself out
  }
  foreach ($k in 'pip', 'setuptools', 'wheel') {                # keep the installers a pip-freeze list does not name
    if (-not ($lines | Where-Object { $_ -match "^$k==" })) {
      $cur = @($p.State.Freeze) | Where-Object { $_ -match "^$k==" } | Select-Object -First 1
      if ($cur) { $lines.Add($cur) }
    }
  }
  $f = "$env:TEMP\vl-sync-$($p.Name).txt"
  Write-Utf8 $f $lines
  $f
}
function Get-IndexArgs($lines) {
  # CUDA builds of torch (+cu128) only exist on the PyTorch index
  $cu = @($lines) | ForEach-Object { if ($_ -match '\+(cu\d+)\b') { $Matches[1] } } | Select-Object -First 1
  if ($cu) { @('--extra-index-url', "https://download.pytorch.org/whl/$cu", '--index-strategy', 'unsafe-best-match') } else { @() }
}
function Invoke-EnvPlan($p, $src) {
  $n = $p.Name; $dir = $PYENVS[$n].Dir; $py = "$dir\Scripts\python.exe"
  if ($p.Action -notin 'sync', 'create', 'rebuild') { Say "$n：$($p.Why)" Green; return $true }
  Head "$n：$($p.Why)"
  if (-not (Has uv)) { Say '需要 uv：先在“环境配置与修复”里装上' Red; return $false }
  if ($DryRun) {
    if ($p.Action -eq 'sync') { & uv pip sync (Write-SyncList $p) --python $py @(Get-IndexArgs $p.List.Lines) --dry-run }
    else { Say "（预览）会$(if ($p.Action -eq 'create') { '新建' } else { '改名保留旧的并重建' })这个环境，再装 $($p.List.Lines.Count) 个包" DarkGray }
    return $true
  }
  $procs = @(Get-EnvProcs $dir)
  $server = @($procs | Where-Object { "$($_.CommandLine)" -like '*http.server*' })
  $others = @($procs | Where-Object { "$($_.CommandLine)" -notlike '*http.server*' })
  if ($others) {
    Say '这个环境里有程序正在运行：' Yellow
    $others | ForEach-Object { $c = "$($_.CommandLine)"; Say ("  PID {0}  {1}" -f $_.ProcessId, $c.Substring(0, [Math]::Min(110, $c.Length))) DarkGray }
    if (-not (Ask '结束这些进程再继续？')) { Say "跳过 $n" Yellow; return $false }
    $others | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 1
  }
  # the page server only uses the standard library: package changes do not disturb it, moving the folder does
  $restartServer = $false
  if ($server -and $p.Action -ne 'sync') {
    $server | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    $restartServer = $true
    Start-Sleep -Seconds 1
  }
  $old = ''
  if ($p.Action -eq 'rebuild') {
    $old = "$dir.old-$(Get-Date -Format yyyyMMdd-HHmmss)"
    try { Rename-Item -LiteralPath $dir -NewName (Split-Path $old -Leaf) -ErrorAction Stop }
    catch { Say "旧环境改名失败（$($_.Exception.Message)），跳过 $n" Red; return $false }
    Say "旧环境先改名保留：$old" DarkGray
  }
  if ($p.Action -in 'create', 'rebuild') {
    $free = Get-FreeGB $dir
    if ($free -lt 8) {
      Say ("{0} 盘只剩 {1:N1} GB，一个带 torch 的环境约需 6 GB" -f $dir.Substring(0, 1), $free) Yellow
      if (-not (Ask '仍然继续？')) { return $false }
    }
    $ver = if ($p.List.Python) { $p.List.Python } else { $PYENVS[$n].Py }
    & uv venv $dir --python $ver
    if ($LASTEXITCODE) { Say "新建 $n 失败（上面是 uv 的报错）" Red; return $false }
  }
  $req = Write-SyncList $p
  $idx = @(Get-IndexArgs $p.List.Lines)
  & uv pip sync $req --python $py @idx
  $ok = $LASTEXITCODE -eq 0
  if ($restartServer) { Start-PageServer | Out-Null }
  if (-not $ok) {
    Say "$n 同步失败（上面是 uv 的报错）" Red
    if ($old) { Say "旧环境还在：$old，可以把它改回 $(Split-Path $dir -Leaf)" Yellow }
    return $false
  }
  $v = Get-EnvPlan $n $src
  if ($v.Action -eq 'none') { Say "$n 已与 $($src.Kind) $($src.Name) 一致" Green } else { Say "$n 仍有差异：$($v.Why)" Yellow }
  if ($old -and $v.Action -eq 'none' -and (Test-Path $old)) {
    if (Ask ("删除改名保留的旧环境 {0}（{1:N1} GB）？" -f $old, (Get-DirGB $old))) { Remove-Item -LiteralPath $old -Recurse -Force }
  }
  $v.Action -eq 'none'
}
function Save-Snapshot([string]$label = '', [switch]$Auto) {
  if (-not (Has uv)) { Say '需要 uv 才能读出各环境的包列表' Red; return $null }
  $name = Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'
  $dir = "$SNAP\$name"
  New-Item -ItemType Directory -Force $dir | Out-Null
  $saved = @()
  foreach ($n in $PYENVS.Keys) {
    $st = Get-EnvState $n
    if (-not $st.Works -or -not $st.Freeze) { continue }
    Write-Utf8 "$dir\requirements-$n.txt" (@("# $n  Python $($st.Version)") + $st.Freeze)
    $saved += "$n $($st.Freeze.Count) 个包"
  }
  $tp = [ordered]@{}
  foreach ($n in $REPOS.Keys) {
    $d = $REPOS[$n].Dir
    if (-not (Test-Path "$d\.git")) { continue }
    $tp[$n] = [ordered]@{ url = "$(& git -C $d remote get-url origin 2>$null)".Trim(); commit = Get-Head $d
      local_changes = @(& git -C $d status --short 2>$null | ForEach-Object { "$_".Trim() }) }
  }
  [IO.File]::WriteAllText("$dir\third_party.json", ($tp | ConvertTo-Json -Depth 4), $UTF8)
  $meta = [ordered]@{ date = (Get-Date -Format s); label = $label; auto = [bool]$Auto; uv = "$(& uv --version)"; vl_cache = $env:VL_CACHE }
  [IO.File]::WriteAllText("$dir\snapshot.json", ($meta | ConvertTo-Json), $UTF8)
  Say "快照已保存：env-snapshots\$name（$($saved -join '，')；第三方仓库 $($tp.Count) 个）" Green
  $name
}
function Select-Source([string]$from) {
  $srcs = @(Get-Sources)
  if (-not $srcs) { Say '没有可用的恢复来源：环境正常时先保存一份快照' Red; return $null }
  if ($from) {
    $s = $srcs | Where-Object { $_.Name -eq $from -or $_.Dir -eq $from } | Select-Object -First 1
    if (-not $s) { Say "找不到恢复来源 $from" Red }
    return $s
  }
  if ($Yes -or $srcs.Count -eq 1) { return $srcs[0] }
  Say '可用的恢复来源：'
  for ($i = 0; $i -lt $srcs.Count; $i++) {
    $note = @($srcs[$i].Note, $(if ($srcs[$i].Auto) { '自动保存' })) | Where-Object { $_ }
    Say ("  {0,2}  {1}  {2}{3}" -f ($i + 1), $srcs[$i].Kind, $srcs[$i].Name, $(if ($note) { "（$($note -join '，')）" } else { '' }))
  }
  $c = "$(Read-Host '选哪一个（回车 = 1）')".Trim()
  if (-not $c) { return $srcs[0] }
  if ($c -match '^\d+$' -and [int]$c -ge 1 -and [int]$c -le $srcs.Count) { return $srcs[[int]$c - 1] }
  $null
}
function Get-RepoDiffs($src) {
  if (-not (Test-Path "$($src.Dir)\third_party.json")) { return }
  $tp = Get-ThirdParty $src
  foreach ($n in $REPOS.Keys) {
    $dir = $REPOS[$n].Dir
    if (-not (Test-Path "$dir\.git")) { continue }
    $head = Get-Head $dir
    if ($head -and $tp[$n].commit -and $head -ne $tp[$n].commit) {
      [pscustomobject]@{ Name = $n; Why = "commit $(Short $head) -> $(Short $tp[$n].commit)" }
    }
  }
}
function Invoke-Restore([string]$from, [string[]]$names) {
  $src = Select-Source $from
  if (-not $src) { return }
  Head "恢复到 $($src.Kind) $($src.Name)"
  $bad = @($names | Where-Object { -not $PYENVS.Contains($_) })
  if ($bad) { Say "没有这些环境：$($bad -join ', ')（可选：$(@($PYENVS.Keys) -join ', ')）" Red; return }
  $targets = if ($names) { $names } else { @($PYENVS.Keys | Where-Object { $PYENVS[$_].Need -or $All -or (Test-Path "$($PYENVS[$_].Dir)\Scripts\python.exe") }) }
  $words = @{ none = '一致'; skip = '跳过'; sync = '同步'; create = '新建'; rebuild = '重建' }
  $plans = @(foreach ($n in $targets) { Get-EnvPlan $n $src })
  foreach ($p in $plans) {
    Say ("  {0,-9} {1}  {2}" -f $p.Name, $words[$p.Action], $p.Why) $(if ($p.Action -eq 'none') { 'Green' } elseif ($p.Action -eq 'skip') { 'DarkGray' } else { 'Yellow' })
  }
  $repoWork = @(if (-not $names) { Get-RepoDiffs $src })
  foreach ($r in $repoWork) { Say ("  {0,-9} 切换  {1}" -f $r.Name, $r.Why) Yellow }
  $work = @($plans | Where-Object { $_.Action -in 'sync', 'create', 'rebuild' })
  if (-not $work -and -not $repoWork) { Say '已经与这个来源一致，不用恢复。' Green; return }
  if ($DryRun) {
    foreach ($p in $work) { Invoke-EnvPlan $p $src | Out-Null }
    Say "`n只是预览，没有改动。去掉 -DryRun 再运行就会按上面的计划恢复。" Cyan
    return
  }
  if (-not (Ask '按上面的计划恢复？')) { return }
  if (@($work | Where-Object { $_.State.Works }).Count) {
    Say '先把当前状态存一份快照（恢复错了可以再恢复回来）' DarkGray
    Save-Snapshot '恢复前自动保存' -Auto | Out-Null
  }
  $ok = 0
  foreach ($p in $work) { if (Invoke-EnvPlan $p $src) { $ok++ } }
  foreach ($r in $repoWork) { Sync-Repo $r.Name $src }
  Head '恢复结果'
  if ($work) { Say ("{0}/{1} 个环境已与来源一致" -f $ok, $work.Count) $(if ($ok -eq $work.Count) { 'Green' } else { 'Yellow' }) }
}

# ---------------------------------------------------------------- repairs
function Sync-Repo([string]$n, $src) {
  if (-not (Has git)) { Say '需要 git：先安装 Git for Windows（https://git-scm.com/download/win）' Red; return }
  $dir = $REPOS[$n].Dir; $tp = (Get-ThirdParty $src)[$n]
  $fresh = $false
  if (-not (Test-Path "$dir\.git")) {
    if ((Test-Path $dir) -and (Get-ChildItem -LiteralPath $dir -Force | Select-Object -First 1)) { Say "$dir 已存在但不是 git 仓库，没有动它" Yellow; return }
    Say "下载 $n：$($tp.url)"
    & git clone $tp.url $dir
    if ($LASTEXITCODE) { Say "下载 $n 失败" Red; return }
    $fresh = $true
  }
  $head = Get-Head $dir
  if ($tp.commit -and $head -ne $tp.commit) {
    if (-not $fresh -and -not (Ask "$n 切换到记录的 commit $(Short $tp.commit)（现在是 $(Short $head)）？")) { return }
    & git -C $dir cat-file -e "$($tp.commit)^{commit}" 2>$null
    if ($LASTEXITCODE) { & git -C $dir fetch origin }
    & git -C $dir -c advice.detachedHead=false checkout --quiet $tp.commit
    if ($LASTEXITCODE) { Say "$n 切换失败（多半是本地改动冲突，见上面 git 的提示）" Red; return }
    Say "$n 已在 commit $(Short $tp.commit)" Green
  }
  if ($n -eq 'Applio') { Set-ApplioPatch }
}
function Test-ApplioPatch {
  # $true applied, $false not yet, $null unknown (no patch / no checkout / the file was changed some other way)
  $dir = $REPOS['Applio'].Dir
  if (-not $PATCH -or -not (Test-Path "$dir\.git") -or -not (Has git)) { return $null }
  & git -C $dir apply --check --reverse $PATCH 2>$null
  if ($LASTEXITCODE -eq 0) { return $true }
  & git -C $dir apply --check $PATCH 2>$null
  if ($LASTEXITCODE -eq 0) { return $false }
  $null
}
function Set-ApplioPatch {
  $s = Test-ApplioPatch
  if ($s -eq $false) {
    & git -C $REPOS['Applio'].Dir apply $PATCH
    if (-not $LASTEXITCODE) { Say 'Applio 省内存补丁已打上' Green }
  } elseif ($null -eq $s -and $PATCH -and (Test-Path "$($REPOS['Applio'].Dir)\.git")) {
    Say 'Applio 的 train.py 与补丁对不上（被别的方式改过），没有动它' Yellow
  }
}
function Install-Uv {
  # through winget (Windows' package manager); this script never runs installer scripts fetched from the web
  if (-not (Has winget)) {
    Say '没有 winget：按 uv 官方说明安装（https://docs.astral.sh/uv/getting-started/installation/），装好后重新打开启动器' Yellow
    return
  }
  if (-not (Ask '用 winget 安装 uv（astral-sh.uv）？')) { return }
  & winget install --id astral-sh.uv --exact --source winget
  $env:PATH = "$env:LOCALAPPDATA\Microsoft\WinGet\Links;$env:USERPROFILE\.local\bin;$env:PATH"
  if (Has uv) { Say "uv 已安装：$(& uv --version)" Green } else { Say 'uv 还不能用：重新打开启动器再试' Yellow }
}
function Get-Zip([string]$url, [string]$what) {
  $zip = "$env:TEMP\vl-$what.zip"; $dir = "$env:TEMP\vl-$what"
  Say "下载 $url"
  & curl.exe -L --fail -o $zip $url
  if ($LASTEXITCODE) { Say "下载 $what 失败" Red; return $null }
  Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
  Expand-Archive -LiteralPath $zip -DestinationPath $dir -Force
  Remove-Item -LiteralPath $zip -Force
  $dir
}
function Get-Ffmpeg {
  $url = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip'
  if (-not (Ask 'ffmpeg：从 FFmpeg 官网列出的 Windows 构建（BtbN，GitHub）下载到 tools\ffmpeg？约 200 MB')) { return }
  $d = Get-Zip $url 'ffmpeg'
  if (-not $d) { return }
  $bin = Get-ChildItem $d -Recurse -Filter ffmpeg.exe | Select-Object -First 1
  New-Item -ItemType Directory -Force "$VL\tools\ffmpeg\bin" | Out-Null
  Copy-Item "$($bin.DirectoryName)\*" "$VL\tools\ffmpeg\bin" -Force
  Remove-Item -LiteralPath $d -Recurse -Force
  Say 'ffmpeg 已放到 tools\ffmpeg\bin' Green
}
function Get-Vgmstream {
  $url = 'https://github.com/vgmstream/vgmstream/releases/latest/download/vgmstream-win64.zip'
  if (-not (Ask 'vgmstream：从官方 GitHub 发布页下载到 tools\vgmstream？约 10 MB')) { return }
  $d = Get-Zip $url 'vgmstream'
  if (-not $d) { return }
  New-Item -ItemType Directory -Force "$VL\tools\vgmstream" | Out-Null
  Copy-Item "$d\*" "$VL\tools\vgmstream" -Recurse -Force
  Remove-Item -LiteralPath $d -Recurse -Force
  Say 'vgmstream 已放到 tools\vgmstream' Green
}
function Get-Models([string]$which) {
  switch ($which) {
    'tools' {
      if (-not (Test-Path $T)) { Say 'tools 环境还没装好，先装环境' Red; return }
      Say '下载转写、分离、去混响、降噪、声纹模型（约 5 GB，会用到显卡）'
      & $T "$PSScriptRoot\prefetch_models.py"
    }
    'indextts' {
      $py = "$($PYENVS['indextts'].Dir)\Scripts\python.exe"
      if (-not (Test-Path $py) -or -not (Test-Path "$VL\repos\index-tts")) { Say '先装好 indextts 环境和 repos\index-tts' Red; return }
      Say '从 Hugging Face 下载 IndexTTS-2.5 到 repos\index-tts\checkpoints（约 4 GB）'
      Push-Location "$VL\repos\index-tts"
      & $py -c "from huggingface_hub import snapshot_download; snapshot_download('IndexTeam/IndexTTS-2.5', local_dir='checkpoints')"
      Pop-Location
    }
    'applio' {
      $py = "$($PYENVS['applio'].Dir)\Scripts\python.exe"
      if (-not (Test-Path $py) -or -not (Test-Path "$($REPOS['Applio'].Dir)\core.py")) { Say '先装好 applio 环境和 Applio 仓库' Red; return }
      Say '用 Applio 自带的下载补齐预训练模型（只下载缺的）'
      Push-Location $REPOS['Applio'].Dir
      & $py core.py prerequisites
      Pop-Location
    }
  }
}
function Get-ReproModel([string]$model) {
  # a trained voice model kept in a reproducible package (release/repro-*/models/<model>)
  @(Get-ChildItem "$VL\release" -Directory -Filter 'repro-*' -ErrorAction SilentlyContinue | Sort-Object Name -Descending |
    ForEach-Object { "$($_.FullName)\models\$model" } | Where-Object { Test-Path "$_\*.pth" }) | Select-Object -First 1
}
function Copy-VoiceModel([string]$model) {
  $src = Get-ReproModel $model
  if (-not $src) { Say "复现包里没有 $model：用 run_voice_model.ps1 重新训练" Yellow; return }
  if (@(Get-ChildItem "$src\*.pth" | Where-Object { $_.Length -lt 4KB }).Count) { Say "$src 里是 Git LFS 指针：先在复现包仓库里运行 git lfs pull" Yellow; return }
  $dst = "$env:VL_CACHE\Applio\logs\$model"
  New-Item -ItemType Directory -Force $dst | Out-Null
  Copy-Item "$src\*" $dst -Force
  Say "模型 $model 已从 $src 恢复" Green
}
function Invoke-Fix([string]$key) {
  switch -Regex ($key) {
    '^uv$' { Install-Uv }
    '^cache$' { New-Item -ItemType Directory -Force $env:VL_CACHE | Out-Null; Say "已创建 $env:VL_CACHE" Green }
    '^repo:(.+)$' { Sync-Repo $Matches[1] (Get-SourceFor 'tools') }
    '^patch$' { Set-ApplioPatch }
    '^env:(.+)$' {
      $n = $Matches[1]
      $src = Get-SourceFor $n
      if (-not $src) { Say "没有 $n 的依赖清单，无法自动安装" Red; break }
      $p = Get-EnvPlan $n $src
      if ($p.Action -eq 'sync' -and -not $script:SavedBefore) {
        Say '先把当前状态存一份快照（恢复错了可以再恢复回来）' DarkGray
        Save-Snapshot '修复前自动保存' -Auto | Out-Null
        $script:SavedBefore = $true
      }
      Invoke-EnvPlan $p $src | Out-Null
    }
    '^model:(.+)$' { Get-Models $Matches[1] }
    '^voice:(.+)$' { Copy-VoiceModel $Matches[1] }
    '^ffmpeg$' { Get-Ffmpeg }
    '^vgmstream$' { Get-Vgmstream }
    '^stop$' { Remove-Item -LiteralPath $STOP -Force -ErrorAction SilentlyContinue; Say '已清除残留的 STOP 文件' Green }
    '^route$' { Resume-Route }
    '^gamereset$' { Set-GameDefault }
  }
}

# ---------------------------------------------------------------- check
function Get-Routing {
  if (-not (Test-Path $T)) { return $null }
  $out = @(& $T "$PSScriptRoot\live_devices.py" 2>&1 | ForEach-Object { "$_" })
  [pscustomobject]@{
    Lines = $out; Failed = ($LASTEXITCODE -ne 0)
    Capture = "$($out | Where-Object { $_ -like 'capture:*' } | Select-Object -First 1)".Trim()
    Play = "$($out | Where-Object { $_ -like 'play:*' } | Select-Object -First 1)".Trim()
    NoVirtual = [bool]($out | Where-Object { $_ -like '*no virtual output device*' })
    NoPlay = [bool]($out | Where-Object { $_ -like '*no real output device*' })
    Misrouted = [bool]($out | Where-Object { $_ -like '! *' })
  }
}
function Format-Dev([string]$line) {
  # "capture: <device> (<why>)" -> "<device>" (device names have brackets of their own)
  $s = $line -replace '^\w+:\s*', ''
  $i = $s.LastIndexOf(' (')
  if ($i -gt 0) { $s.Substring(0, $i) } else { $s }
}
function Test-EnvImports([string]$n) {
  $py = "$($PYENVS[$n].Dir)\Scripts\python.exe"
  $code = "import importlib; [importlib.import_module(m) for m in '$($PYENVS[$n].Deep)'.split(',')]; import torch; " +
    "print('torch', torch.__version__, 'GPU' if torch.cuda.is_available() else 'NO-GPU')"
  $out = @(& $py -c $code 2>&1 | ForEach-Object { "$_" } | Where-Object { $_.Trim() })
  [pscustomobject]@{ Ok = ($LASTEXITCODE -eq 0); Text = $(if ($out) { $out[-1].Trim() } else { '' }) }
}
function Invoke-Check([switch]$Deep, [switch]$Quiet) {
  $res = New-Object System.Collections.Generic.List[object]
  function Add([string]$area, [string]$item, [string]$st, [string]$detail, [string]$hint = '', [string]$fix = '') {
    $res.Add([pscustomobject]@{ Area = $area; Item = $item; Status = $st; Detail = $detail; Hint = $hint; Fix = $fix })
  }
  if (-not $Quiet) { Say '检查中…' DarkGray }

  # system
  $os = Get-CimInstance Win32_OperatingSystem
  $ramT = $os.TotalVisibleMemorySize / 1MB; $ramF = $os.FreePhysicalMemory / 1MB
  $hint = if ($ramT -lt 15) { 'IndexTTS 约占 13 GB、AudioSR 约 15 GB：页面文件要够大，重任务一次只跑一个' } else { '可用内存少时，开着游戏用实时替换容易断音：关掉不用的程序' }
  Add '系统' '内存' $(if ($ramT -lt 15 -or $ramF -lt 3) { 'WARN' } else { 'OK' }) ('共 {0:N1} GB，现在可用 {1:N1} GB' -f $ramT, $ramF) $hint
  if (Has nvidia-smi) {
    $g = "$(& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null | Select-Object -First 1)".Trim()
    Add '系统' '显卡' $(if ($g) { 'OK' } else { 'FAIL' }) $(if ($g) { $g } else { 'nvidia-smi 没有返回显卡' }) '生成台词和训练需要 NVIDIA 显卡；实时替换只用 CPU'
  } else { Add '系统' '显卡' 'FAIL' '找不到 NVIDIA 驱动（nvidia-smi）' '生成台词和训练需要 NVIDIA 显卡；实时替换只用 CPU，不受影响' }
  foreach ($d in @('C', $VL.Substring(0, 1), $env:VL_CACHE.Substring(0, 1)) | Select-Object -Unique) {
    $free = Get-FreeGB "${d}:"
    if ($d -eq 'C') { $st = $(if ($free -lt 2) { 'FAIL' } elseif ($free -lt 5) { 'WARN' } else { 'OK' }); $hint = 'C 盘太满会拖垮系统和页面文件：清理 C 盘' }
    else { $st = $(if ($free -lt 5) { 'FAIL' } elseif ($free -lt 15) { 'WARN' } else { 'OK' }); $hint = '新建或重建一个 Python 环境约需 6 GB；分离、训练还要更多' }
    Add '系统' "$d 盘" $st ('剩余 {0:N1} GB' -f $free) $hint
  }
  $pf = @(Get-CimInstance Win32_PageFileUsage -ErrorAction SilentlyContinue)
  $pfText = ($pf | ForEach-Object { '{0} {1:N0} GB' -f $_.Name.Substring(0, 2), ($_.AllocatedBaseSize / 1024) }) -join '，'
  if (@($pf | Where-Object { $_.Name -notlike 'C:*' }).Count) { Add '系统' '页面文件' 'OK' $pfText }
  else { Add '系统' '页面文件' 'WARN' $(if ($pfText) { "只在 C 盘：$pfText" } else { '读不到' }) '在 系统属性 > 高级 > 性能设置 > 虚拟内存 里给 D/E 盘加页面文件（由你来改）' }
  $ascii = $env:VL_CACHE -match '^[\x20-\x7E]+$'
  if (-not $ascii) { Add '系统' '缓存路径 VL_CACHE' 'FAIL' $env:VL_CACHE '必须是纯英文路径（IndexTTS 等打不开含中文的路径）：设置环境变量 VL_CACHE' }
  elseif (-not (Test-Path $env:VL_CACHE)) { Add '系统' '缓存路径 VL_CACHE' 'FAIL' "$env:VL_CACHE 不存在" '' 'cache' }
  else { Add '系统' '缓存路径 VL_CACHE' 'OK' $env:VL_CACHE }

  # tools
  if (Has uv) { Add '工具' 'uv' 'OK' "$(& uv --version)" } else { Add '工具' 'uv' 'FAIL' '没有安装' '用来建立和恢复 Python 环境' 'uv' }
  if (Has git) { Add '工具' 'git' 'OK' "$(& git --version)" } else { Add '工具' 'git' 'FAIL' '没有安装' '安装 Git for Windows（https://git-scm.com/download/win）：第三方仓库要用' }
  $ff = "$VL\tools\ffmpeg\bin\ffmpeg.exe"
  if (Test-Path $ff) { Add '工具' 'ffmpeg' 'OK' (("$(& $ff -version 2>$null | Select-Object -First 1)" -split ' ')[0..2] -join ' ') }
  elseif (Has ffmpeg) { Add '工具' 'ffmpeg' 'OK' '用系统 PATH 里的 ffmpeg' }
  else { Add '工具' 'ffmpeg' 'FAIL' '没有找到' '提取音轨、转码都要用' 'ffmpeg' }
  if (Test-Path "$VL\tools\vgmstream\vgmstream-cli.exe") { Add '工具' 'vgmstream' 'OK' 'tools\vgmstream' }
  else { Add '工具' 'vgmstream' 'FAIL' '没有找到' '读取 .wem 要用' 'vgmstream' }

  # Python environments, compared with the default restore source
  $base = @(Get-Sources) | Select-Object -First 1
  foreach ($n in $PYENVS.Keys) {
    $e = $PYENVS[$n]
    $st = Get-EnvState $n
    if (-not $st.Exists) {
      Add 'Python 环境' $n $(if ($e.Need) { 'FAIL' } else { 'OPT' }) "没有安装（$($e.Use)）" '' $(if ($e.Need -or $All) { "env:$n" })
      continue
    }
    if (-not $st.Works) { Add 'Python 环境' $n 'FAIL' 'Python 无法运行（它依赖的基础 Python 可能被删了）' '' "env:$n"; continue }
    $detail = "Python $($st.Version)，$($st.Freeze.Count) 个包"
    $list = Get-SourceList $base $n
    if (-not (Has uv)) { Add 'Python 环境' $n 'WARN' "$detail（没有 uv，无法比对包）" }
    elseif ($list) {
      $mm = (("$($list.Python)" -split '\.')[0..1]) -join '.'
      $d = Compare-Pkgs (ConvertTo-PkgMap $list.Lines) (ConvertTo-PkgMap $st.Freeze)
      if ($list.Python -and -not $st.Version.StartsWith("$mm.")) { Add 'Python 环境' $n 'WARN' "$detail；$($base.Kind) $($base.Name) 里是 Python $($list.Python)" '恢复时会重建这个环境' "env:$n" }
      elseif ($d.Count) { Add 'Python 环境' $n 'WARN' "$detail；与$($base.Kind) $($base.Name) 不同：$(Format-Diff $d)" '恢复：选“恢复环境到快照”，或在“环境配置与修复”里一起处理' "env:$n" }
      else { Add 'Python 环境' $n 'OK' "$detail，与$($base.Kind) $($base.Name) 一致" }
    } else { Add 'Python 环境' $n 'OK' $detail }
  }
  if (-not @(Get-ChildItem $SNAP -Directory -ErrorAction SilentlyContinue).Count) {
    Add 'Python 环境' '快照' 'INFO' '还没有保存过环境快照' '环境都正常时选“保存环境快照”，以后可以一键恢复到这个状态'
  }
  if ($Deep) { foreach ($r in @(Get-DeepResults)) { $res.Add($r) } }

  # third-party code
  $tp = Get-ThirdParty $base
  foreach ($n in $REPOS.Keys) {
    $r = $REPOS[$n]
    if (-not (Test-Path "$($r.Dir)\.git")) {
      Add '第三方仓库' $n $(if ($r.Need) { 'FAIL' } else { 'OPT' }) "没有检出（$($r.Dir)）" '' $(if ($r.Need -or $All) { "repo:$n" })
      continue
    }
    $head = Get-Head $r.Dir
    $want = $tp[$n].commit
    if ($want -and $head -ne $want) { Add '第三方仓库' $n 'WARN' "commit $(Short $head)，记录的是 $(Short $want)" '' "repo:$n" }
    else { Add '第三方仓库' $n 'OK' "commit $(Short $head)" }
    if ($n -eq 'Applio') {
      $s = Test-ApplioPatch
      if ($s -eq $true) { Add '第三方仓库' 'Applio 补丁' 'OK' '省内存补丁已打上' }
      elseif ($s -eq $false) { Add '第三方仓库' 'Applio 补丁' 'WARN' '省内存补丁没有打上' '训练时内存占用会高很多' 'patch' }
      elseif ($PATCH) { Add '第三方仓库' 'Applio 补丁' 'WARN' 'train.py 与补丁对不上（被别的方式改过）' '对照 patches\applio-train-workers.patch 手动检查' }
    }
  }

  # models
  foreach ($m in $MODELS) {
    if (Test-Path $m.Path) { Add '模型' $m.What 'OK' '已下载' }
    elseif ($m.Fix -eq 'auto') { Add '模型' $m.What 'INFO' '还没下载（第一次用到时自动下载）' }
    else { Add '模型' $m.What 'FAIL' '没有下载' '' "model:$($m.Fix)" }
  }

  # voices: trained model and finished lines
  foreach ($v in @(Get-Voices)) {
    $logs = "$env:VL_CACHE\Applio\logs\$($v.Model)"
    $model = @(Get-ChildItem $logs -Filter "$($v.Model)_*e_*s.pth" -ErrorAction SilentlyContinue).Count -and
      @(Get-ChildItem $logs -Filter '*.index' -ErrorAction SilentlyContinue).Count
    $what = "$($v.Label)（$($v.Name)）"
    if ($model) {
      if ($v.Lines) { Add '声线' $what 'OK' "模型 $($v.Model)，成品 $($v.Lines) 句" }
      else { Add '声线' $what 'WARN' "模型 $($v.Model)，还没有成品台词" "生成：run_voice_lines.ps1 -Voice $($v.Name)" }
      continue
    }
    $repro = Get-ReproModel $v.Model
    $hint = if ($repro) { "可以从复现包恢复：$repro" } else { "训练：run_voice_model.ps1 -Voice $($v.Name)" }
    $fix = if ($repro) { "voice:$($v.Model)" } else { '' }
    # finished lines are all the live tool needs; the model only matters when lines are made again
    if ($v.Lines) { Add '声线' $what 'INFO' "成品 $($v.Lines) 句，实时替换可用；voice.json 指定的模型 $($v.Model) 还没有（重新生成台词时才需要）" $hint $fix }
    else { Add '声线' $what 'WARN' "没有模型 $($v.Model)，也没有成品台词" $hint $fix }
  }
  if (Test-Path "$VL\exp\all\voice_path_fir.npy") { Add '声线' '实时替换的通道滤波器' 'OK' 'exp\all\voice_path_fir.npy' }
  else { Add '声线' '实时替换的通道滤波器' 'WARN' '没有 exp\all\voice_path_fir.npy' '用一段录屏重新估计（README“录屏替换”）；没有它识别会变差' }

  # what is running
  $srv = Get-PageServer
  if ($srv) { Add '运行状态' '页面服务' 'OK' "运行中（端口 $PORT，PID $($srv.OwningProcess)）" }
  else { Add '运行状态' '页面服务' 'INFO' '没开（打开试听页面时会自动开）' }
  if (Test-Path $STOP) { Add '运行状态' 'STOP 文件' 'WARN' '有上次留下的 logs\live\STOP' '不清掉的话实时工具一启动就会退出' 'stop' }
  $live = @(Get-LiveProcs)
  if ($live) { Add '运行状态' '实时工具' 'INFO' "正在运行（PID $(($live | ForEach-Object ProcessId) -join ', ')）" }
  $r = Get-Routing
  if ($r -and -not $r.Failed) {
    if ($r.NoVirtual) { Add '运行状态' '声音路由' 'WARN' '没有虚拟输出设备' '装 VB-CABLE，或确认 Steam Streaming Speakers 没被禁用（由你来装）' }
    elseif ($r.Misrouted -and $live) { Add '运行状态' '声音路由' 'WARN' '实时工具开着，游戏却直接输出到你听的设备' '自动切换几秒内会处理；一直这样就看 logs\live\route.log' }
    elseif ($r.Misrouted) { Add '运行状态' '声音路由' 'OK' "游戏现在直接输出到 $(Format-Dev $r.Capture)（开实时替换时自动切到虚拟设备）" }
    else {
      $wait = if ($r.Capture -like '*not running*') { '（游戏没开，先在这里等）' } else { '' }
      Add '运行状态' '声音路由' 'OK' "截获 $(Format-Dev $r.Capture)$wait；播放 $(Format-Dev $r.Play)"
    }
  } elseif ($r) { Add '运行状态' '声音路由' 'WARN' "设备检查出错：$(@($r.Lines)[-1])" }
  $s = Invoke-Route 'status'
  if ($s) {
    $item = '游戏声音自动切换'
    if ($s.pending) {
      $w = if ($s.watcher) { '（后台小程序在等）' } else { '，但后台小程序没在运行' }
      Add '运行状态' $item $(if ($s.watcher) { 'INFO' } else { 'WARN' }) "游戏关着，下次启动时切回 $(Format-Back $s.back)$w" '' $(if (-not $s.watcher) { 'route' })
    } elseif ($s.active -and $s.live) { Add '运行状态' $item 'OK' "实时工具运行中：游戏 -> $($s.to)" }
    elseif ($s.active) { Add '运行状态' $item 'WARN' "实时工具已关，游戏还没切回 $(Format-Back $s.back)" '' 'route' }
    elseif ($s.game_running -and -not $live -and $s.game_device -match $VIRTUAL) {
      Add '运行状态' $item 'WARN' "游戏输出到 $($s.game_device)，但实时工具没开：你听不到游戏" '' 'gamereset'
    } elseif ($s.game_running) { Add '运行状态' $item 'OK' "游戏输出到 $($s.game_device)；开实时替换时自动切换" }
    else { Add '运行状态' $item 'OK' '没有待切回的设置；开实时替换时自动切换' }
  }

  if (-not $Quiet) { Show-Results $res }
  $res
}
function Get-DeepResults {
  foreach ($n in $PYENVS.Keys) {
    if (-not (Test-Path "$($PYENVS[$n].Dir)\Scripts\python.exe")) { continue }
    Write-Host "  导入 $n 的主要模块…" -ForegroundColor DarkGray
    $t = Test-EnvImports $n
    $st = if (-not $t.Ok) { 'FAIL' } elseif ($t.Text -like '*NO-GPU*') { 'WARN' } else { 'OK' }
    [pscustomobject]@{ Area = '深度检查'; Item = $n; Status = $st; Detail = $t.Text
      Hint = $(if ($st -eq 'WARN') { 'torch 看不到显卡：驱动或 torch 的 CUDA 版本不对' } else { '恢复这个环境：选“恢复环境到快照”' }); Fix = $(if ($st -ne 'OK') { "env:$n" } else { '' }) }
  }
}
function Show-Results($res) {
  $words = @{ OK = '正常'; WARN = '注意'; FAIL = '问题'; INFO = '提示'; OPT = '可选' }
  $colors = @{ OK = 'Green'; WARN = 'Yellow'; FAIL = 'Red'; INFO = 'Gray'; OPT = 'DarkGray' }
  $area = ''
  foreach ($r in $res) {
    if ($r.Area -ne $area) { Head $r.Area; $area = $r.Area }
    Write-Host ("  [{0}] " -f $words[$r.Status]) -ForegroundColor $colors[$r.Status] -NoNewline
    Write-Host ("{0}：{1}" -f $r.Item, $r.Detail)
    if ($r.Hint -and $r.Status -in 'WARN', 'FAIL', 'OPT', 'INFO') { Write-Host "         $($r.Hint)" -ForegroundColor DarkGray }
  }
  $f = @($res | Where-Object { $_.Status -eq 'FAIL' }).Count
  $w = @($res | Where-Object { $_.Status -eq 'WARN' }).Count
  $fixable = @($res | Where-Object { $_.Fix -and $_.Status -in 'FAIL', 'WARN' }).Count
  Write-Host ''
  if (-not $f -and -not $w) { Say '全部正常。' Green }
  else {
    $tail = if ($fixable) { '其中 {0} 项可以自动处理：选“环境配置与修复”（或 -Setup）。' -f $fixable } else { '' }
    Say ("{0} 项问题，{1} 项注意。{2}" -f $f, $w, $tail) $(if ($f) { 'Red' } else { 'Yellow' })
  }
}

# ---------------------------------------------------------------- setup
function Invoke-Setup {
  Head '环境配置与修复：先检查一遍'
  $res = @(Invoke-Check -Quiet)
  $todo = @($res | Where-Object { $_.Fix -and (($_.Status -in 'FAIL', 'WARN') -or ($All -and $_.Status -eq 'OPT')) })
  $manual = @($res | Where-Object { -not $_.Fix -and $_.Status -in 'FAIL', 'WARN' })
  if ($manual) {
    Say '需要你自己处理的：' Yellow
    $manual | ForEach-Object { Say "  - $($_.Item)：$($_.Detail)$(if ($_.Hint) { "。$($_.Hint)" })" DarkGray }
  }
  if (-not $todo) { Say '没有需要自动处理的项目。' Green; return }
  Say '可以自动处理的：'
  $todo | ForEach-Object { Say "  - $($_.Area) / $($_.Item)：$($_.Detail)" }
  if (-not (Ask '开始处理？（下载和安装可能要较长时间）')) { return }
  $script:SavedBefore = $false
  $keys = @($todo | ForEach-Object { $_.Fix } | Select-Object -Unique)
  foreach ($prefix in 'uv', 'cache', 'repo:', 'patch', 'env:', 'model:', 'voice:', 'ffmpeg', 'vgmstream', 'stop', 'route', 'gamereset') {
    foreach ($k in @($keys | Where-Object { $_ -eq $prefix -or ($prefix.EndsWith(':') -and $_.StartsWith($prefix)) })) { Invoke-Fix $k }
  }
  Head '处理完，再检查一遍'
  $after = @(Invoke-Check)
  if (-not @($after | Where-Object { $_.Status -eq 'FAIL' }).Count -and -not @(Get-Sources | Where-Object { $_.Kind -eq '快照' -and -not $_.Auto }).Count) {
    if (Ask '环境已经可用，保存一份快照作为以后恢复的基准？') { Save-Snapshot '配置完成' | Out-Null }
  }
}

# ---------------------------------------------------------------- live tool, pages, shortcut
function Stop-Live {
  New-Item -ItemType File -Force $STOP | Out-Null                  # the tool's own clean exit: recordings are finalized
  for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Milliseconds 500
    if (-not @(Get-LiveProcs).Count) { Say '上次的实时工具已正常退出' Green; return $true }
  }
  Remove-Item -LiteralPath $STOP -Force -ErrorAction SilentlyContinue
  if (-not (Ask '10 秒内没有退出，强制结束？（那次的录音可能不完整）')) { return $false }
  @(Get-LiveProcs) | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  $true
}
function Start-Live([string]$v) {
  $voice = @(Get-Voices) | Where-Object { $_.Name -eq $v } | Select-Object -First 1
  Head "实时替换：$(if ($voice) { "$($voice.Label)（$v）" } else { $v })"
  if (-not (Test-Path $T)) { Say 'tools 环境没有装好：先选“环境配置与修复”' Red; return }
  if (-not $voice -or -not $voice.Lines) { Say "声线 $v 还没有成品台词（$(Get-ExpDir 'all' $v)\final_48k）" Red; return }
  if (Test-Path $STOP) { Remove-Item -LiteralPath $STOP -Force; Say '清除了上次留下的 STOP 文件（否则工具一启动就会退出）' DarkGray }
  $running = @(Get-LiveProcs)
  if ($running) {
    Say "实时工具已经在运行（PID $(($running | ForEach-Object ProcessId) -join ', ')）" Yellow
    if (-not (Ask '让它正常退出（录音会保存好），再重新启动？')) { return }
    if (-not (Stop-Live)) { return }
  }
  Resume-Route
  Say '检查声音设备…' DarkGray
  $r = Get-Routing
  if ($r) {
    $r.Lines | ForEach-Object { Say "  $_" DarkGray }
    if ($r.NoVirtual) { Say '没有虚拟输出设备：装 VB-CABLE，或确认 Steam Streaming Speakers 没被禁用（由你来装）。' Yellow }
    if ($r.NoPlay) { Say '没有可以播放的真实设备：插上耳机或音箱（工具运行中插上也会自动切过去）。' Yellow }
    if (-not $NoRoute) { Say '游戏的输出会自动切到虚拟设备（先开游戏还是先开工具都行），退出工具时切回。' Cyan }
    elseif ($r.Misrouted) {
      Say '游戏的声音直接出到了你听的设备上，替换后会听到两遍。' Yellow
      Say '在“应用音量和设备首选项”里，把游戏的输出改成 Steam Streaming Speakers（或 CABLE Input）。Windows 会记住，只需改一次；改好后工具会自动跟上。' Yellow
      if (Ask '现在打开这个设置页？') { Start-Process 'ms-settings:apps-volume' }
    }
  }
  $free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
  if ($free -lt 3) { Say ('可用内存只有 {0:N1} GB：游戏开着时可能断音，先关掉不用的程序' -f $free) Yellow }
  Say '热键：~ 标记“刚才听着不对”，Ctrl+Alt+B 旁路（听原声），Ctrl+Alt+Q 退出' Cyan
  $la = @{ Voice = $v }
  if ($NoRoute) { $la.NoRoute = $true }
  if ($Play) { $la.Play = $Play }
  & "$PSScriptRoot\live.ps1" @la
  if ($NoRoute) { return }
  $st = Read-Json $ROUTE
  if ($st -and $st.pending) { Say "游戏先关了：之后不开实时工具直接玩时，游戏一启动就切回 $(Format-Back $st.back_name)。" Cyan }
  elseif ($st -and $st.active) { Say '游戏声音没能自动切回：选菜单 9，或在“应用音量和设备首选项”里改回' Yellow }
}
function Get-PageList {
  $files = @(Get-ChildItem "$VL\eval" -Recurse -Filter *.html -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 14)
  $files += @(Get-ChildItem "$VL\logs\live" -Directory -ErrorAction SilentlyContinue | Where-Object { Test-Path "$($_.FullName)\review\review.html" } |
      Sort-Object Name -Descending | Select-Object -First 3 | ForEach-Object { Get-Item "$($_.FullName)\review\review.html" })
  foreach ($f in $files) {
    $head = (Get-Content -LiteralPath $f.FullName -Encoding UTF8 -TotalCount 40) -join ' '
    [pscustomobject]@{ Rel = $f.FullName.Substring($VL.Length + 1).Replace('\', '/'); Time = $f.LastWriteTime
      Title = $(if ($head -match '<title>([^<]+)</title>') { $Matches[1].Trim() } else { '' }) }
  }
}
function Open-Pages {
  if (-not (Start-PageServer)) { return }
  $pages = @(Get-PageList)
  if (-not $pages) { Say '还没有生成任何页面' Yellow; return }
  Head '试听页面（新的在前；最后几项是实时工具的复盘）'
  for ($i = 0; $i -lt $pages.Count; $i++) {
    $p = $pages[$i]
    Say ("  {0,2}  {1:MM-dd HH:mm}  {2}" -f ($i + 1), $p.Time, $(if ($p.Title) { "$($p.Title)  ($($p.Rel))" } else { $p.Rel }))
  }
  Say '   a  在浏览器里浏览 eval 文件夹'
  $c = "$(Read-Host '打开哪一个（编号，回车返回）')".Trim()
  if ($c -eq 'a') { Start-Process "http://localhost:$PORT/eval/"; return }
  if ($c -match '^\d+$' -and [int]$c -ge 1 -and [int]$c -le $pages.Count) { Start-Process "http://localhost:$PORT/$($pages[[int]$c - 1].Rel)" }
}
function New-DesktopShortcut {
  $lnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'voice-lab 启动器.lnk'
  $s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
  # through cmd.exe, not the .cmd itself: Windows runs its "打开文件 - 安全警告" check on the shortcut's target,
  # and a .cmd in a folder with a low-integrity label (a sandbox may set one on the project root) counts as
  # downloaded. cmd.exe is a system file, so the check passes, and cmd.exe does not check what it runs.
  $s.TargetPath = "$env:SystemRoot\System32\cmd.exe"
  $s.Arguments = "/c `"`"$VL\一键启动.cmd`"`""
  $s.WorkingDirectory = $VL
  $s.IconLocation = "$env:SystemRoot\System32\SndVol.exe,0"
  $s.WindowStyle = 7                                               # the .cmd only starts the window: keep it minimized
  $s.Description = 'voice-lab 启动器'
  $s.Save()
  Say "已创建：$lnk" Green
}

# ---------------------------------------------------------------- menu
function Show-Menu([string]$v) {
  Resume-Route
  while ($true) {
    $voice = @(Get-Voices) | Where-Object { $_.Name -eq $v } | Select-Object -First 1
    $missing = @($PYENVS.Keys | Where-Object { $PYENVS[$_].Need -and -not (Test-Path "$($PYENVS[$_].Dir)\Scripts\python.exe") })
    $rt = Read-Json $ROUTE
    $game = if ($rt -and $rt.pending) { "等游戏下次启动时切回 $(Format-Back $rt.back_name)" }
    elseif ($rt -and $rt.active) { "已切到 $($rt.to_name)" } else { '自动切换' }
    Write-Host ''
    Write-Host '  voice-lab 启动器' -ForegroundColor Cyan
    Write-Host ('  声线：{0}    页面服务：{1}    Python 环境：{2}' -f $(if ($voice) { "$($voice.Label)（$v）" } else { "$v（没有成品）" }),
      $(if (Get-PageServer) { '运行中' } else { '未开' }), $(if ($missing) { "缺 $($missing -join '、')（选 5）" } else { '齐全' }))
    Write-Host "  游戏声音：$game"
    Write-Host '  ----------------------------------------------'
    Write-Host '  1  实时替换（游戏声音自动切换）'
    Write-Host '  2  切换声线'
    Write-Host '  3  打开试听页面'
    Write-Host '  4  环境检查'
    Write-Host '  5  环境配置与修复'
    Write-Host '  6  保存环境快照'
    Write-Host '  7  恢复环境到快照'
    Write-Host '  8  在桌面创建快捷方式'
    Write-Host '  9  游戏声音切回默认设备（补救用）'
    Write-Host '  0  退出'
    $c = "$(Read-Host '  选择')".Trim()
    switch ($c) {
      '1' { Start-Live $v }
      '2' { $v = Select-Voice $v }
      '3' { Open-Pages }
      '4' {
        $res = @(Invoke-Check)
        if (Ask '再做深度检查（每个环境导入主要模块、确认显卡可用，约 1 分钟）？') { Show-Results @(Get-DeepResults) }
      }
      '5' { Invoke-Setup }
      '6' { Save-Snapshot "$(Read-Host '给这份快照写个说明（可留空）')" | Out-Null }
      '7' { Invoke-Restore '' @() }
      '8' { New-DesktopShortcut }
      '9' { Set-GameDefault }
      { $_ -in '0', 'q', 'exit' } { return }
    }
  }
}

$Envs = @($Envs | ForEach-Object { "$_" -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$v0 = if ($Voice) { $Voice } else { Get-DefaultVoice }
if ($Live) { Start-Live $v0; exit }
if ($Pages) { Open-Pages; exit }
if ($GameDefault) { Set-GameDefault; exit }
if ($Shortcut) { New-DesktopShortcut; exit }
if ($Check) {
  $r = @(Invoke-Check -Deep:$Deep -Quiet:([bool]$JsonOut))
  if ($JsonOut) {
    $json = ConvertTo-Json -InputObject @($r | Select-Object Area, Item, Status, Detail, Hint, Fix) -Depth 3
    [IO.File]::WriteAllText($JsonOut, $json, $UTF8)
    $f = @($r | Where-Object { $_.Status -eq 'FAIL' }).Count; $w = @($r | Where-Object { $_.Status -eq 'WARN' }).Count
    Say ("检查完成：{0} 项问题，{1} 项注意，{2} 项正常" -f $f, $w, @($r | Where-Object { $_.Status -eq 'OK' }).Count)
  }
  exit $(if (@($r | Where-Object { $_.Status -eq 'FAIL' }).Count) { 1 } else { 0 })
}
if ($Setup) { Invoke-Setup; exit }
if ($Snapshot) { Save-Snapshot $Label | Out-Null; exit }
if ($Restore) { Invoke-Restore $From $Envs; exit }
Show-Menu $v0
