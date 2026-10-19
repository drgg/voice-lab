# Refine one problem line (scan_lines.py route "refine", or anything that fails by ear).
#  1) takes + pick page:  .\scripts\run_refine.ps1 -Line L14 -Texts '<phrase 1>|<phrase 2>|<phrase 3>' [-Set all]
#     (routes.csv column `suggest` has a per-burst transcript to start the texts from)
#  2) after listening:    .\scripts\run_refine.ps1 -Line L14 -Pick '1=1-07,2=2-03,3=3-01' [-Set all]
#     -> exp/<set>/refined(_48k)/<line>.wav, which finalize.py prefers over the automatic version
# Takes per phrase: IndexTTS (emotion 1.0 and 0.7, -Cands each), the same polished by RVC (stronger
# breath/consonant protection, lighter index for clearer words), and voice conversion of the original
# phrase (RVC shifted to the voice's register, Seed-VC), which keeps non-verbal sounds such as grunts and
# breathy speech. -Voice picks the target voice (default default; see run_d4.ps1).
param([Parameter(Mandatory)][string]$Line, [string]$Texts = '', [string]$Pick = '', [string]$Set = '',
      [string]$Voice = 'default', [string]$Char = 'source',
      [string]$Emotions = 'sad,fearful', [int]$Cands = 6, [string[]]$Skip = @())
. "$PSScriptRoot\env.ps1"
$env:VL_SET = $Set
$env:VL_VOICE = $Voice
$env:VL_CHAR = $Char  # the game character (vl_paths)
$vc = Get-VoiceCfg $Voice
$T = "$VL\envs\tools\Scripts\python.exe"
$S = "$VL\envs\seedvc\Scripts\python.exe"
$R = "$env:VL_CACHE\envs\applio\Scripts\python.exe"
$A = "$VL\envs\audiosr\Scripts\python.exe"
$setDir = Get-SetDir $Set $Char
$expDir = Get-ExpDir $Set $Voice
$dir = "$setDir\refine\$Line"
$refs = if ($Set) { "$VL\data\$Voice\refs\sets\$(Get-SetKey $Set $Char)\refine_$Line" } else { "$VL\data\$Voice\refs\refine_$Line" }
$rvcClear = @('--protect', '0.15', '--index-rate', '0.25')

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

if ($Pick) {
  Step 'refine pick' $T @('make_refine_page.py', '--line', $Line, '--pick', $Pick)
  Step 'refine postprocess 22k' $T @('postprocess.py', '--groups', 'refined')
  Step 'refine AudioSR 48k' $A @('superres.py', '--groups', 'refined')
  Step 'refine postprocess 48k' $T @('postprocess.py', '--groups', 'refined_48k')
  if (Test-Path "$expDir\final_48k") { Step 'refine finalize' $T @('finalize.py') }
  return
}
if (-not $Texts) { throw '-Texts is required (phrase texts separated by |), or -Pick after listening' }

Step 'refine split' $T @('refine_split.py', '--line', $Line, '--texts', $Texts)
Step 'refine emotion refs' $S @('emotion_match.py', '--lines-csv', "$dir\phrases.csv", '--out', $refs, '--emotions', $Emotions)
foreach ($alpha in @('1.0', '0.7')) {
  $tag = 'a' + $alpha.Replace('.', '')
  Step "refine IndexTTS $tag" $INDEXTTS_PY @('run_indextts.py', '--phrases', "$dir\phrases.csv", '--cands', "$Cands",
    '--spk-dir', $refs, '--emo-alpha', $alpha, '--name', "${Line}_tts_$tag")
  Step "refine RVC polish $tag" $R (@('run_rvc.py', 'infer', '--src', "$expDir\${Line}_tts_$tag", '--name', "${Line}_tts_${tag}_rvc") + $rvcClear)
}
Step 'refine RVC on original' $R (@('run_rvc.py', 'infer', '--src', $dir, '--pitch', "$($vc.nonverbal_pitch)", '--name', "${Line}_vc_rvc") + $rvcClear)
Step 'refine Seed-VC on original' $S @('run_seedvc.py', '--groups', 'A', '--src-dir', $dir, '--ref-dir', $refs, '--name', "${Line}_vc_seedvc")
$sets = @(
  "${Line}_tts_a10=IndexTTS · 情感 1.0", "${Line}_tts_a07=IndexTTS · 情感 0.7",
  "${Line}_tts_a10_rvc=IndexTTS · 情感 1.0 + RVC", "${Line}_tts_a07_rvc=IndexTTS · 情感 0.7 + RVC",
  "${Line}_vc_rvc=原声表演 → RVC（$(Format-Shift $vc.nonverbal_pitch)）", "${Line}_vc_seedvc=原声表演 → Seed-VC"
) -join ','
Step 'refine page' $T @('make_refine_page.py', '--line', $Line, '--sets', $sets)
