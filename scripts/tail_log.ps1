# Follow a pipeline log that another PowerShell is still writing (its *> redirect keeps the file locked for
# tail and other readers; Get-Content shares it). Prints the new lines matching -Pattern every -Every s.
# Usage:  powershell -File scripts\tail_log.ps1 -Log logs\<voice>_data.log [-Pattern '^=== |Error'] [-Until 'refs exit']
param([Parameter(Mandatory)][string]$Log, [string]$Pattern = '^=== |Traceback|Error|failed', [string]$Until = '',
      [int]$Every = 10, [switch]$New)   # -New: only lines written from now on
$seen = if ($New -and (Test-Path $Log)) { @(Get-Content $Log -ErrorAction SilentlyContinue).Count } else { 0 }
while ($true) {
  if (Test-Path $Log) {
    $lines = @(Get-Content $Log -ErrorAction SilentlyContinue)
    if ($lines.Count -lt $seen) { $seen = 0 }               # the log was rewritten
    for ($i = $seen; $i -lt $lines.Count; $i++) {
      if ($lines[$i] -match $Pattern) { [Console]::Out.WriteLine($lines[$i].Substring(0, [Math]::Min(220, $lines[$i].Length))) }
      if ($Until -and $lines[$i] -match $Until) { [Console]::Out.Flush(); exit 0 }
    }
    $seen = $lines.Count
    [Console]::Out.Flush()
  }
  Start-Sleep -Seconds $Every
}
