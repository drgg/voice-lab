# Dot-source before any heavy command:  . '<repo>\scripts\env.ps1'
# Keeps every cache / temp / managed Python on E: (C: has very little free space).
$global:VL = Split-Path -Parent $PSScriptRoot
$env:VL_ROOT = $VL
$env:HF_HOME = "$VL\cache\hf"
$env:HUGGINGFACE_HUB_CACHE = "$VL\cache\hf\hub"
$env:MODELSCOPE_CACHE = "$VL\cache\modelscope"
$env:UV_CACHE_DIR = "$VL\cache\uv"
$env:UV_PYTHON_INSTALL_DIR = "$VL\cache\uv-python"
$env:UV_LINK_MODE = 'hardlink'
$env:TORCH_HOME = "$VL\cache\torch"
$env:XDG_CACHE_HOME = "$VL\cache\xdg"   # openai-whisper (used by Vevo2) downloads to ~/.cache otherwise
$env:PIP_CACHE_DIR = "$VL\cache\pip"
$env:TEMP = "$VL\cache\tmp"
$env:TMP = "$VL\cache\tmp"
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:PATH = "$VL\tools\ffmpeg\bin;$VL\tools\vgmstream;$env:PATH"
# IndexTTS's text normaliser (kaldifst, C++) cannot open files under a non-ASCII path such as
# a non-ASCII folder name, so that one venv lives on an ASCII path. uv sync/run in repos/index-tts picks this up.
# Applio, the IndexTTS venv live under VL_CACHE (an ASCII path, set it to your own)
if (-not $env:VL_CACHE) { $env:VL_CACHE = 'C:\voice-lab-cache' }
$env:UV_PROJECT_ENVIRONMENT = "$env:VL_CACHE\envs\indextts"
$global:INDEXTTS_PY = "$env:VL_CACHE\envs\indextts\Scripts\python.exe"

# The target voice's settings (data/<voice>/voice.json, as vl_paths.voice_cfg; defaults = the first voice)
function global:Get-VoiceCfg([string]$Voice) {
  $cfg = [ordered]@{ label = 'target'; nonverbal_pitch = 2 }
  $p = "$VL\data\$Voice\voice.json"
  if ($Voice -and (Test-Path $p)) {
    (Get-Content $p -Raw -Encoding UTF8 | ConvertFrom-Json).PSObject.Properties | ForEach-Object { $cfg[$_.Name] = $_.Value }
  }
  [pscustomobject]$cfg
}
# "升 2 半音" / "降 1 半音" / "不变调" for page labels
function global:Format-Shift([double]$St) {
  if ($St -gt 0) { "升 $St 半音" } elseif ($St -lt 0) { "降 $(-$St) 半音" } else { '不变调' }
}
# exp/<set> for the first voice, exp/<set>_<voice> for the others; another game character than the first
# (source) puts its name in front: exp/<char>_<set>[_<voice>] (vl_paths.EXP)
function global:Get-ExpDir([string]$Set, [string]$Voice, [string]$Char = $env:VL_CHAR) {
  $name = if ($Set) { $Set } else { 'test20' }
  if ($Char -and $Char -ne 'source') {
    $name = "${Char}_$name"
    if (-not $Voice -or $Voice -eq 'default') { return "$VL\exp\$name" } else { return "$VL\exp\${name}_$Voice" }
  }
  if (-not $Voice -or $Voice -eq 'default') { if ($Set) { "$VL\exp\$Set" } else { "$VL\exp" } } else { "$VL\exp\${name}_$Voice" }
}
# the game character's lines: data/<char>/sets/<set>, or data/<char> for the 20 test lines (vl_paths.SET_DIR)
function global:Get-SetDir([string]$Set, [string]$Char = $env:VL_CHAR) {
  $c = if ($Char) { $Char } else { 'source' }
  if ($Set) { "$VL\data\$c\sets\$Set" } else { "$VL\data\$c" }
}
# the name a set's results carry: <set> for the first character, <char>_<set> for another (vl_paths._BASE)
function global:Get-SetKey([string]$Set, [string]$Char = $env:VL_CHAR) {
  $name = if ($Set) { $Set } else { 'test20' }
  if ($Char -and $Char -ne 'source') { "${Char}_$name" } else { $name }
}
