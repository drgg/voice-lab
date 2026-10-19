# Production pipeline, D4 direction (chosen after experiment rounds 1-4, see README):
#   scan      route each line: speech / refine (re-dub, flagged) / nonverbal (grunts, breaths)
#   re-dub    split at the original's pauses, per-phrase reference of the target voice + emotion prompt, best of
#             -Cands IndexTTS takes by intonation fidelity, stitched on the original timing,
#             then polished by RVC (the voice's timbre)
#   nonverbal RVC on the original performance, shifted to the voice's register (voice.json nonverbal_pitch)
#   48 kHz    AudioSR + loudness matched to the original line
#   finalize  hand-refined > RVC-polished re-dub (or -Final tts) / non-verbal RVC -> final_48k,
#             -Export copies it to out/<set>/ under the original file names
# Usage:  .\scripts\run_d4.ps1 -Set all -Export        (sets from prep_set.py; '' = the 20 test lines)
#         .\scripts\run_d4.ps1 -Set all -Voice <voice> -Export   (another target voice: data/<voice>/, results in
#         exp/<set>_<voice>/; prepare it with run_voice_data.ps1 and train its RVC model first)
param([string]$Set = '', [string]$Voice = 'default', [string]$Char = 'source', [int]$Cands = 4, [ValidateSet('rvc', 'tts')][string]$Final = 'rvc',
      [switch]$Export, [switch]$Eval, [switch]$AllVariants, [switch]$Legacy, [string]$Choices = '', [string[]]$Skip = @())
# -Choices: code from a choice page (run_nonverbal.ps1), e.g. 'default=rvc;L102=orig'; it is stored, so
# later runs keep applying it. Re-runs are incremental: unchanged takes, lines and files are skipped.
# -AllVariants: also super-resolve the re-dub variant the final mix does not use (for side-by-side
# listening); by default only what goes into final_48k is taken to 48 kHz (AudioSR is the slow step)
. "$PSScriptRoot\env.ps1"
$env:VL_SET = $Set  # read by scripts/vl_paths.py in every env
$env:VL_VOICE = $Voice
$env:VL_CHAR = $Char  # the game character (vl_paths)
$vc = Get-VoiceCfg $Voice
$T = "$VL\envs\tools\Scripts\python.exe"
$S = "$VL\envs\seedvc\Scripts\python.exe"
$R = "$env:VL_CACHE\envs\applio\Scripts\python.exe"
$A = "$VL\envs\audiosr\Scripts\python.exe"
$setDir = Get-SetDir $Set $Char
$refs = if ($Set) { "$VL\data\$Voice\refs\sets\$(Get-SetKey $Set $Char)\emo_phrase" } else { "$VL\data\$Voice\refs\emo_phrase" }
$phr = "$setDir\phrases\phrases.csv"
if (-not (Test-Path "$setDir\lines\lines.csv")) { throw "no line set at $setDir - run scripts/prep_set.py --name $Set first" }

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

Step 'scan' $T @('scan_lines.py')
Step 'segment' $T @('segment_lines.py')
Step 'emotion phrases' $S @('emotion_match.py', '--lines-csv', $phr, '--out', $refs)
Step 'D4 IndexTTS candidates' $INDEXTTS_PY @('run_indextts.py', '--phrases', $phr, '--cands', "$Cands",
  '--spk-dir', $refs, '--emo-alpha', '1.0', '--name', 'D4_cands')
# -Legacy: selection and RVC handling as before 2026-10-12 (no line fit, no pitch guard) - reproduces older sets
$legacyArgs = if ($Legacy) { @('--legacy') } else { @() }   # not $legacy: PowerShell names ignore case
Step 'select phrases' $T (@('select_phrases.py', '--cands', 'D4_cands', '--name', 'D4_indextts_phrase') + $legacyArgs)
Step 'RVC polish D4' $R @('run_rvc.py', 'infer', '--src', 'D4_indextts_phrase')
# pitch guard: a phrase RVC moved by more than 3 st (its tracker lost a breathy or short stretch) is redone
# with the FCPE tracker, and the version that kept the melody better is used
if ($Skip -notcontains 'RVC guard' -and -not $Legacy) {
  Push-Location $PSScriptRoot
  $off = & $T rvc_guard.py check 2>$null | Select-Object -Last 1
  Pop-Location
  if ($off) {
    Write-Output "RVC pitch guard: $off"
    Step 'RVC guard redo' $R @('run_rvc.py', 'infer', '--src', 'D4_indextts_phrase', '--name', 'D4_indextts_phrase_rvc_alt',
      '--f0-method', 'fcpe', '--lines', $off, '--force')
    Write-Output "=== RVC guard pick start $((Get-Date).ToString('HH:mm:ss'))"
    Push-Location $PSScriptRoot
    $pick = & $T rvc_guard.py pick --lines $off 2>$null
    Pop-Location
    $pick | Where-Object { $_ -notmatch '^FALLBACK' } | ForEach-Object { Write-Output $_ }
    Write-Output "=== RVC guard pick exit 0 after 0.0 min"
    # still off with both trackers: the take is the problem (a murmur the re-dub cannot say), so the original
    # performance of that phrase goes through RVC (the voice's register shift) and is pasted in its place
    $fb = (($pick | Where-Object { $_ -match '^FALLBACK' } | Select-Object -Last 1) -replace '^FALLBACK\s*', '').Trim()
    if ($fb) {
      Write-Output "RVC pitch guard fallback (original performance through RVC): $fb"
      Step 'RVC guard original' $R @('run_rvc.py', 'infer', '--src', "$setDir\phrases", '--name', 'D4_phrase_vc',
        '--pitch', "$($vc.nonverbal_pitch)", '--lines', $fb, '--force', '--protect', '0.15', '--index-rate', '0.25')
      Step 'RVC guard splice' $T @('rvc_guard.py', 'splice', '--phrases', $fb)
    }
  }
}
Step 'RVC nonverbal' $R @('run_rvc.py', 'infer', '--src', 'source', '--route', 'nonverbal', '--pitch', "$($vc.nonverbal_pitch)", '--name', 'V_rvc')
$tts = if ($Final -eq 'rvc') { 'D4_indextts_phrase_rvc' } else { 'D4_indextts_phrase' }
$sr = if ($AllVariants) { @('D4_indextts_phrase', 'D4_indextts_phrase_rvc', 'V_rvc') } else { @($tts, 'V_rvc') }
Step 'postprocess 22k' $T @('postprocess.py', '--groups', 'D4_indextts_phrase,D4_indextts_phrase_rvc,V_rvc')
Step 'AudioSR 48k' $A @('superres.py', '--groups', ($sr -join ','))
Step 'postprocess 48k' $T @('postprocess.py', '--groups', (($sr | ForEach-Object { "${_}_48k" }) -join ','))
$tts = "${tts}_48k"
$fin = @('finalize.py', '--tts-group', $tts)
if ($Export) { $fin += '--export' }
if ($Choices) { $fin += @('--choices', $Choices) }
Step 'finalize' $T $fin
if ($Eval) {
  Step 'metrics' $T @('metrics.py')
  Step 'noise check' $T @('noise_check.py')
}
Step 'review page' $T @('make_listening_page.py', '--labeled', '--methods',
  'final_48k,D4_indextts_phrase_48k,D4_indextts_phrase_rvc_48k,V_rvc_48k,refined_48k', '--out', 'review.html')
