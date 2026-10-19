# Live replacement: the game's sound in -> the character's lines replaced -> your headphones.
# The game's output is switched to "Steam Streaming Speakers" (or "CABLE Input" with VB-CABLE) while this runs and
# given back when it stops (scripts/app_route.py); -NoRoute leaves that to you (Settings > System > Sound > Volume
# mixer, <game>.exe).
#   .\scripts\live.ps1              devices found automatically and followed while running: the game's
#                                   output from its audio session; you hear the Windows default output if
#                                   it is a real device (else headphones > speakers > display), switching
#                                   when you plug or unplug something or change the default
#   .\scripts\live.ps1 -List        what is connected and what would be used
#   .\scripts\live.ps1 -Play 'Headphones'   pin the output (any part of the device name)
#   .\scripts\live.ps1 -Voice <voice>    replace with another voice's finished lines (exp/<set>_<voice>/final_48k);
#   .\scripts\live.ps1 -Also @()         only the source character (default); other characters replaced in the same session:
#                                        -Also <char>:<set>:<voice>:<group>
#   .\scripts\live.ps1 -Final <group>    the source character's version (exp/<set>_<voice>/<group>; default final_48k)
# Without -Also / -Final, the launcher page's picks are used: logs/launcher.json "lineup" (per character a voice and
# version, "" = not replaced; added 2026-10-17), else the defaults above.
# Hotkeys while running: ~ marks a wrong moment, Space records the voice that just went by (the game still gets
# both keys), Ctrl+Alt+1..n a character in or out of the party, Ctrl+Alt+B bypass, Ctrl+Alt+Q quit. Afterwards the
# quick review of the session opens; the deep check is printed to run.
param(
    [string]$Capture = 'auto',
    [string]$Play = 'auto',
    [int]$Delay = 200,
    [int]$Buffer = 80,
    [string]$Voice = 'default',
    [string[]]$Also = @(),
    [string]$Final = 'final_48k',
    [switch]$List,
    [switch]$NoRecord,
    [switch]$NoRoute,
    [switch]$PrintArgs               # show what live_replace.py would get (lineup checks) and stop
)
. "$PSScriptRoot\env.ps1"
$env:VL_SET = 'all'
$env:VL_VOICE = $Voice
$py = "$VL\envs\tools\Scripts\python.exe"
$lineupFile = "$VL\logs\launcher.json"
if ((-not $PSBoundParameters.ContainsKey('Also') -or -not $PSBoundParameters.ContainsKey('Final')) -and (Test-Path $lineupFile)) {
    try {
        $lineup = (Get-Content $lineupFile -Raw -Encoding UTF8 | ConvertFrom-Json).lineup
    } catch { $lineup = $null }
    if ($lineup) {
        $fromPage = @()
        foreach ($p in $lineup.PSObject.Properties) {
            if ($p.Name -eq 'source') {
                if ($p.Value.group -and -not $PSBoundParameters.ContainsKey('Final')) { $Final = $p.Value.group }
            } elseif ($p.Value.voice) {
                $fromPage += "$($p.Name):all:$($p.Value.voice):$($p.Value.group)"
            }
        }
        if (-not $PSBoundParameters.ContainsKey('Also')) { $Also = $fromPage }
    }
}
$a = @('--capture', $Capture, '--play', $Play, '--delay', $Delay, '--buffer', $Buffer, '--final', $Final)
if ($List) { & $py "$PSScriptRoot\live_replace.py" --list @a; return }
if ($NoRecord) { $a += '--no-record' }
if ($NoRoute) { $a += @('--route', 'off') }
foreach ($x in $Also) { if ($x) { $a += @('--also', $x) } }
if ($PrintArgs) { Write-Host "VL_VOICE=$Voice live_replace.py $($a -join ' ')"; return }
& $py "$PSScriptRoot\live_replace.py" @a

$last = Get-ChildItem "$VL\logs\live" -Directory | Where-Object { $_.Name -match '^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$' } |
    Sort-Object LastWriteTime | Select-Object -Last 1
if ($last -and (Test-Path "$($last.FullName)\events.jsonl")) {
    & $py "$PSScriptRoot\live_review.py" $last.FullName --no-deep
    $page = "$($last.FullName)\review\review.html"
    $restart = Select-String -Path "$($last.FullName)\events.jsonl" -Pattern '"type": "stop_file"' -Quiet
    if ((Test-Path $page) -and -not $restart) { Start-Process $page }   # a restart from outside: no page popping up
    elseif (Test-Path $page) { Write-Host "Review: $page" }
    Write-Host "Deep check (offline comparison, slower than real time):"
    Write-Host "  & '$py' '$PSScriptRoot\live_review.py' '$($last.FullName)'"
}
