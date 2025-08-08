# Discord Messages Visualizer - PowerShell Runner
# Double-click this file to run the Streamlit app

Write-Host "Starting Discord Messages Visualizer..." -ForegroundColor Green
Write-Host "This will open in your web browser..." -ForegroundColor Yellow

try {
    # Change to script directory
    Set-Location $PSScriptRoot
    
    # Run streamlit
    python -m streamlit run main.py
}
catch {
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Make sure Python and the required packages are installed." -ForegroundColor Yellow
    Write-Host "Run: pip install -r requirements.txt" -ForegroundColor Cyan
    Read-Host "Press Enter to exit"
}
