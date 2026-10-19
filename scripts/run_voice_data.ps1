# Material for a target voice, from episode files to a training set and references (incremental: episodes
# already extracted / separated / transcribed are skipped, so new episodes can be added later):
#   extract     the Japanese audio track of every episode in -Src (read in place)
#   separate    vocals / instrumental (Mel-Band RoFormer)
#   slice       speech clips (Silero VAD) + level, background and radio-voice measures
#   transcribe  Whisper large-v3 text per clip
#   rank        speaker similarity to her (catchphrase anchors, pitch floor)  - data/<voice>/voice.json
#   classify    speaker classifier from speech-habit cues (own vs other characters' markers)
#   clean       de-reverb + de-noise the clips that pass
#   trainset    data/<voice>/train/ for RVC
#   refs        reference clips (ref_12s, ref_25s, abx_*) for TTS prompts, metrics and the register offset
# Usage:  .\scripts\run_voice_data.ps1 -Voice <voice> -Src '<show folder>\<season folder>' [-Skip extract,...]
param([Parameter(Mandatory)][string]$Voice, [string[]]$Src = @(), [string[]]$Skip = @())   # -Src: one or more season folders
. "$PSScriptRoot\env.ps1"
$env:VL_VOICE = $Voice
$T = "$VL\envs\tools\Scripts\python.exe"
if (-not (Test-Path "$VL\data\$Voice\voice.json")) { throw "no settings at data\$Voice\voice.json (label, markers, keyword, pitch floor)" }

function Step($name, $exe, $argv) {
  if ($Skip -contains $name) { Write-Output "$name - skipped"; return }
  $t0 = Get-Date
  Write-Output "=== $name start $($t0.ToString('HH:mm:ss'))"
  Push-Location $PSScriptRoot
  & $exe @argv 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -notmatch 'it/s\]|s/it\]|Failed to load|Please follow|Xet Storage|INFO' }
  $script:LastCode = $LASTEXITCODE
  Pop-Location
  Write-Output ("=== {0} exit {1} after {2:N1} min" -f $name, $script:LastCode, ((Get-Date) - $t0).TotalMinutes)
  Write-Output ("    E free GB: {0:N1}  C free GB: {1:N2}" -f ((Get-PSDrive E).Free / 1GB), ((Get-PSDrive C).Free / 1GB))
  if ($script:LastCode -ne 0) { throw "step '$name' failed (exit $script:LastCode)" }
}

foreach ($s in $Src) { Step 'extract' $T @('prep_voice.py', '--stage', 'extract', '--src', $s) }
Step 'separate' $T @('prep_voice.py', '--stage', 'separate')
Step 'slice' $T @('prep_voice.py', '--stage', 'slice')
Step 'transcribe' $T @('prep_voice.py', '--stage', 'transcribe')
Step 'rank' $T @('prep_voice.py', '--stage', 'rank')
Step 'classify' $T @('prep_voice.py', '--stage', 'classify')
Step 'clean' $T @('clean_refs.py')
Step 'trainset' $T @('build_trainset.py')
Step 'refs' $T @('build_refs.py')
