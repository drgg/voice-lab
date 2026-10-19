# Watch a running pipeline and EXIT with a reason as soon as something needs the user's attention:
#   SLOW     a stage has run longer than 2x its time budget
#   STALLED  a stage is open but python processes used < 10 CPU-seconds over the last 10 minutes
#   DISK     C: free space below -MinCFreeGB (default 3 GB); the pipeline is killed first, because a
#            growing pagefile took C: from 12.7 GB to 0.3 GB within one 60 s poll
#   MEMORY   commit charge above 95 %
#   VRAM SPILL  a python process above 20 GB private memory (GPU memory overflowed into RAM; real spills hit 25-41 GB,
#               while IndexTTS normally commits ~13 GB)
#   DONE     the watched log reports its final stage finished
# Usage:  .\scripts\watchdog.ps1 -Logs round3_data.log,round3.log,round3_rvc.log -FinalStage 'compare page'
param([string[]]$Logs, [string]$FinalStage = 'compare page', [int]$MaxHours = 10, [double]$MinCFreeGB = 3.0,
      [double]$BudgetScale = 1.0)  # budgets are sized for ~20 lines; e.g. -BudgetScale 4 for all 180
$logDir = Join-Path (Split-Path -Parent $PSScriptRoot) 'logs'
$budget = @{   # minutes per stage name (prefix match); 2x this = SLOW
  'extract' = 5; 'separate' = 90; 'slice' = 10; 'transcribe' = 60; 'rank' = 30; 'clean' = 30
  'trainset' = 5; 'emotion' = 15; 'fine-tune' = 45; 'A3' = 15; 'B3' = 20; 'D3' = 20
  'postprocess' = 5; 'AudioSR' = 40; 'metrics' = 10; 'noise' = 3; 'compare' = 3
  'RVC train' = 150; 'RVC infer' = 15; 'RVC polish' = 15
  'segment' = 5; 'D4' = 40; 'select' = 15; 'refine' = 15
  'scan' = 10; 'RVC nonverbal' = 15; 'finalize' = 3; 'review' = 3; 'nonverbal' = 10
}
function CpuSeconds { (Get-Process python -ErrorAction SilentlyContinue | Measure-Object CPU -Sum).Sum }
function KillPipeline {  # every voice-lab python tree except the static page server
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'voice-lab' -and $_.CommandLine -notmatch 'http\.server' } |
    ForEach-Object { taskkill /PID $_.ProcessId /T /F 2>$null | Out-Null }
}
function DiskLow {  # returns the alert text (or nothing); the caller prints it - output inside `if (...)` is swallowed
  $c = (Get-PSDrive C).Free / 1GB
  if ($c -lt $MinCFreeGB) {
    $stage = $lastStage
    KillPipeline
    "DISK: C: free {0:N2} GB during '{1}' - pipeline processes killed" -f $c, $stage
  }
}
$cpuHist = New-Object System.Collections.Generic.Queue[double]
$lastStage = ''
$end = (Get-Date).AddHours($MaxHours)
while ((Get-Date) -lt $end) {
  $alert = DiskLow; if ($alert) { $alert; exit 0 }
  $os = Get-CimInstance Win32_OperatingSystem
  $commit = ($os.TotalVirtualMemorySize - $os.FreeVirtualMemory) / $os.TotalVirtualMemorySize
  if ($commit -gt 0.95) { "MEMORY: commit charge {0:P0} (RAM free {1:N1} GB)" -f $commit, ($os.FreePhysicalMemory / 1MB); exit 0 }
  # a python process this large means VRAM overflowed into system RAM (Windows sysmem fallback)
  $big = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.PrivateMemorySize64 -gt 20GB } | Select-Object -First 1
  if ($big) { "VRAM SPILL: python pid {0} holds {1:N1} GB private memory" -f $big.Id, ($big.PrivateMemorySize64 / 1GB); exit 0 }

  $active = $Logs | ForEach-Object { Join-Path $logDir $_ } | Where-Object { Test-Path $_ } |
    Sort-Object { (Get-Item $_).LastWriteTime } | Select-Object -Last 1
  if ($active) {
    $lines = Get-Content $active | Where-Object { $_ -match '^=== ' }
    if (($lines | Where-Object { $_ -match "^=== $([regex]::Escape($FinalStage)) exit" }) -and $active -like "*$($Logs[-1])") {
      "DONE: $FinalStage finished ($([IO.Path]::GetFileName($active)))"; exit 0
    }
    $last = $lines | Select-Object -Last 1
    if ($last -match '^=== (.+) start (\d\d:\d\d:\d\d)$') {
      $stage = $Matches[1]
      $started = [datetime]::ParseExact($Matches[2], 'HH:mm:ss', $null)
      if ($started -gt (Get-Date)) { $started = $started.AddDays(-1) }
      $mins = ((Get-Date) - $started).TotalMinutes
      $key = $budget.Keys | Where-Object { $stage -like "$_*" } | Select-Object -First 1
      if ($key -and $mins -gt 2 * $budget[$key] * $BudgetScale) {
        "SLOW: stage '$stage' running {0:N0} min (budget {1} min) in {2}" -f $mins, ($budget[$key] * $BudgetScale), [IO.Path]::GetFileName($active); exit 0
      }
      # a new stage means new processes: the old ones' CPU time leaves the sum, so start over
      if ($stage -ne $lastStage) { $cpuHist.Clear(); $lastStage = $stage }
      $cpuHist.Enqueue((CpuSeconds))
      while ($cpuHist.Count -gt 11) { [void]$cpuHist.Dequeue() }
      if ($cpuHist.Count -eq 11) {
        $arr = $cpuHist.ToArray()
        $used = 0.0  # only count increases: a helper process exiting must not look like a stall
        for ($i = 1; $i -lt $arr.Length; $i++) { $used += [math]::Max(0, $arr[$i] - $arr[$i - 1]) }
        if ($used -lt 10) { "STALLED: stage '$stage' - python used <10 CPU-s in 10 min"; exit 0 }
      }
    } else { $cpuHist.Clear() }
  }
  for ($i = 0; $i -lt 4; $i++) { Start-Sleep -Seconds 15; $alert = DiskLow; if ($alert) { $alert; exit 0 } }
}
"TIMEOUT: watchdog ran $MaxHours h"
