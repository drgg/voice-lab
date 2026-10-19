# Every line of a set in a target voice, after run_voice_data.ps1 and run_voice_model.ps1:
#   run_d4.ps1         re-dub (per phrase, best of -Cands) + RVC polish, non-verbal RVC, 48 kHz
#   run_nonverbal.ps1  non-verbal options + check + choice page (auto pick per line)
#   run_refine.ps1     hand-refined lines of the first voice, redone in this voice; the auto pick (best take
#                      per phrase) is used until you pick by ear on eval/<set>_<voice>/refine_<line>.html
#   finalize           final_48k with -Choices (default: auto), exported to out/<set>_<voice>/
# The shared analysis of the set (scan, segment) is reused.
# Usage:  .\scripts\run_voice_lines.ps1 -Voice <voice> [-Set all] [-Refine 'L159=<phrase 1>|<phrase 2>|<phrase 3>']
param([Parameter(Mandatory)][string]$Voice, [string]$Set = 'all', [string]$Char = 'source', [string]$Refine = '', [string]$Choices = 'default=auto',
      [int]$Cands = 4, [switch]$NoWem, [switch]$Legacy)   # -Legacy: as before 2026-10-12
. "$PSScriptRoot\env.ps1"
$env:VL_SET = $Set
$env:VL_VOICE = $Voice
$env:VL_CHAR = $Char  # the game character (vl_paths)
$T = "$VL\envs\tools\Scripts\python.exe"

& "$PSScriptRoot\run_d4.ps1" -Set $Set -Voice $Voice -Char $Char -Cands $Cands -Skip scan, segment, finalize, 'review page' -Legacy:$Legacy
& "$PSScriptRoot\run_nonverbal.ps1" -Set $Set -Voice $Voice -Char $Char -Cands $Cands -Legacy:$Legacy
foreach ($item in ($Refine -split ';' | Where-Object { $_ })) {
  $line, $texts = $item -split '=', 2
  & "$PSScriptRoot\run_refine.ps1" -Set $Set -Voice $Voice -Char $Char -Line $line -Texts $texts
  $n = ($texts -split '\|').Count
  $auto = (1..$n | ForEach-Object { "$_=$_-01" }) -join ','   # the best-scored take of every phrase
  & "$PSScriptRoot\run_refine.ps1" -Set $Set -Voice $Voice -Char $Char -Line $line -Pick $auto
}
$t0 = Get-Date
Write-Output "=== finalize start $($t0.ToString('HH:mm:ss'))"
Push-Location $PSScriptRoot
& $T finalize.py --tts-group D4_indextts_phrase_rvc_48k --choices $Choices --export 2>&1 | ForEach-Object { "$_" }
& $T make_listening_page.py --labeled --methods 'final_48k,D4_indextts_phrase_48k,D4_indextts_phrase_rvc_48k,V_rvc_48k,refined_48k' --out review.html 2>&1 | ForEach-Object { "$_" }
Pop-Location
Write-Output ("=== finalize exit {0} after {1:N1} min" -f $LASTEXITCODE, ((Get-Date) - $t0).TotalMinutes)
Write-Output "=== voice done exit 0"
