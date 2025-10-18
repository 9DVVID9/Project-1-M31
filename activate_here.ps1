# activate_here.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
. .\.venv\Scripts\Activate.ps1
Write-Host "✅ Virtual environment activated in this window." -ForegroundColor Green
