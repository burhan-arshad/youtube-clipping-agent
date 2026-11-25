# Registers the "featureclip" task: starts hidden at every Windows login, also on battery, never times out.
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument ('"' + (Join-Path $dir "start_background.vbs") + '"') -WorkingDirectory $dir
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$trigger.Delay = "PT30S"
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "featureclip" -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Write-Host "Done. featureclip will start by itself (hidden) every time you log in to Windows."
