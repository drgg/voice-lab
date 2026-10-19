# Non-verbal lines (grunts, breaths, shouts): build every option and one page to pick from.
#   N_orig  the original performance (same actress; no conversion artifacts at all)
#   N_rvc   RVC with 0.5 s of silence around each clip (so the pitch tracker locks on), stronger
#           breath protection, lighter index; 40 kHz output resampled to 48 kHz instead of AudioSR,
#           which as a speech model invented detail on 82 of 93 grunts
#   N_rvc2  the same, shifted to the voice's register (voice.json nonverbal_pitch; target: +2 semitones)
#   N_tts   IndexTTS speaking the interjection, original as emotion prompt, best of -Cands
#   V_rvc   the version in the current final mix, for comparison
# Then: listen on eval/<set>/nonverbal.html, copy the code, and run
#   .\scripts\run_d4.ps1 -Set all -Export -Choices '<code>' -Skip ...   (or finalize.py --choices directly)
param([string]$Set = 'all', [string]$Voice = 'default', [string]$Char = 'source', [int]$Cands = 4, [string[]]$Skip = @(), [switch]$Legacy)
. "$PSScriptRoot\env.ps1"
$env:VL_SET = $Set
$env:VL_VOICE = $Voice
$env:VL_CHAR = $Char  # the game character (vl_paths)
$vc = Get-VoiceCfg $Voice
$shift = Format-Shift $vc.nonverbal_pitch
$T = "$VL\envs\tools\Scripts\python.exe"
$S = "$VL\envs\seedvc\Scripts\python.exe"
$R = "$env:VL_CACHE\envs\applio\Scripts\python.exe"
$A = "$VL\envs\audiosr\Scripts\python.exe"
$setDir = Get-SetDir $Set $Char
$refs = if ($Set) { "$VL\data\$Voice\refs\sets\$(Get-SetKey $Set $Char)\emo_nonverbal" } else { "$VL\data\$Voice\refs\emo_nonverbal" }
$phr = "$setDir\nonverbal_phrases"
$rvc = @('run_rvc.py', 'infer', '--src', 'source', '--route', 'nonverbal', '--pad', '0.5', '--protect', '0.15', '--index-rate', '0.25')

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

Step 'nonverbal prep' $T @('prep_nonverbal.py')
Step 'nonverbal RVC' $R ($rvc + @('--pitch', '0', '--name', 'N_rvc'))
Step 'nonverbal RVC shifted' $R ($rvc + @('--pitch', "$($vc.nonverbal_pitch)", '--name', 'N_rvc2'))
Step 'nonverbal emotion refs' $S @('emotion_match.py', '--lines-csv', "$phr\phrases.csv", '--out', $refs)
Step 'nonverbal IndexTTS' $INDEXTTS_PY @('run_indextts.py', '--phrases', "$phr\phrases.csv", '--cands', "$Cands",
  '--spk-dir', $refs, '--emo-alpha', '1.0', '--name', 'N_cands')
Step 'nonverbal select' $T (@('select_phrases.py', '--phrases-dir', $phr, '--cands', 'N_cands', '--name', 'N_tts') +
  $(if ($Legacy) { @('--legacy') } else { @() }))
Step 'nonverbal postprocess' $T @('postprocess.py', '--groups', 'N_rvc,N_rvc2,N_tts')
Step 'nonverbal resample 48k' $A @('superres.py', '--groups', 'N_rvc,N_rvc2', '--plain')
Step 'nonverbal AudioSR 48k' $A @('superres.py', '--groups', 'N_tts')
Step 'nonverbal postprocess 48k' $T @('postprocess.py', '--groups', 'N_rvc_48k,N_rvc2_48k,N_tts_48k')
Step 'nonverbal check' $T @('check_nonverbal.py', '--groups', 'N_rvc_48k,N_rvc2_48k,N_tts_48k,N_orig_48k,V_rvc_48k')
Step 'nonverbal page' $T @('make_choice_page.py', '--route', 'nonverbal', '--default', 'auto', '--auto', 'rvc,rvc2,tts,orig',
  '--out', 'nonverbal.html',
  '--groups', "rvc=N_rvc_48k=RVC 新版（不变调）,rvc2=N_rvc2_48k=RVC 新版（$($shift)）,tts=N_tts_48k=IndexTTS 重配,orig=N_orig_48k=原声保留,old=V_rvc_48k=当前版本")
