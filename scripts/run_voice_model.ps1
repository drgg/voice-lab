# Train the RVC model of a target voice on data/<voice>/train (run_voice_data.ps1 first), Applio env.
# Training's own export of the inference weights sometimes fails; the export step rebuilds them from the
# last checkpoint. ~1 h for 300 epochs of 7 min on a 4060 Ti; scale -Epochs down for bigger sets.
# Usage:  .\scripts\run_voice_model.ps1 -Voice <voice> [-Epochs 200]
param([Parameter(Mandatory)][string]$Voice, [int]$Epochs = 300, [string[]]$Skip = @())
. "$PSScriptRoot\env.ps1"
$env:VL_VOICE = $Voice
$R = "$env:VL_CACHE\envs\applio\Scripts\python.exe"

function Step($name, $exe, $argv) {
  if ($Skip -contains $name) { Write-Output "$name - skipped"; return }
  $t0 = Get-Date
  Write-Output "=== $name start $($t0.ToString('HH:mm:ss'))"
  Push-Location $PSScriptRoot
  & $exe @argv 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -notmatch 'it/s\]|s/it\]|Failed to load|Please follow|Xet Storage' }
  $script:LastCode = $LASTEXITCODE
  Pop-Location
  Write-Output ("=== {0} exit {1} after {2:N1} min" -f $name, $script:LastCode, ((Get-Date) - $t0).TotalMinutes)
}

$m = (Get-VoiceCfg $Voice).rvc_model; if (-not $m) { $m = "${Voice}_v1" }   # voice.json rvc_model (run_rvc.py)
Write-Output "model: $m"
Step 'RVC train' $R @('run_rvc.py', 'train', '--epochs', "$Epochs")
$w = Get-ChildItem "$env:VL_CACHE\Applio\logs\$m" -Filter "${m}_${Epochs}e_*s.pth" -ErrorAction SilentlyContinue
if (-not $w) { Step 'RVC export' $R @('run_rvc.py', 'export') }
Get-ChildItem "$env:VL_CACHE\Applio\logs\$m" -Filter "${m}_*e_*s.pth" | Sort-Object LastWriteTime |
  Select-Object -Last 1 | ForEach-Object { Write-Output "weights: $($_.FullName)" }
Get-ChildItem "$env:VL_CACHE\Applio\logs\$m" -Filter '*.index' | ForEach-Object { Write-Output "index: $($_.FullName)" }
