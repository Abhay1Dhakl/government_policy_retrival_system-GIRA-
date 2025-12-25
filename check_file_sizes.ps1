# PowerShell Script to Check Python File Sizes in GIRA Project
# Usage: .\check_file_sizes.ps1

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "GIRA Project - Python File Size Analysis" -ForegroundColor Cyan
Write-Host "Target: All files should be 200-300 lines maximum" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host ""

$projectRoot = $PSScriptRoot
$targetDirs = @("gira-ai\gira-agent", "gira-ai\gira-mcp-server")

$results = @{
    "Critical" = @()   # > 500 lines
    "Warning" = @()    # 300-500 lines
    "Good" = @()       # 200-300 lines
    "Excellent" = @()  # < 200 lines
}

foreach ($dir in $targetDirs) {
    $fullPath = Join-Path $projectRoot $dir
    
    if (Test-Path $fullPath) {
        Write-Host "Analyzing: $dir" -ForegroundColor Yellow
        
        $files = Get-ChildItem -Path $fullPath -Recurse -Filter "*.py" | Where-Object {
            $_.Name -ne "__pycache__" -and $_.DirectoryName -notlike "*__pycache__*"
        }
        
        foreach ($file in $files) {
            $lineCount = (Get-Content $file.FullName -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
            $relativePath = $file.FullName.Replace($projectRoot + "\", "")
            
            $fileInfo = @{
                Path = $relativePath
                Lines = $lineCount
            }
            
            if ($lineCount -gt 500) {
                $results["Critical"] += $fileInfo
            }
            elseif ($lineCount -gt 300) {
                $results["Warning"] += $fileInfo
            }
            elseif ($lineCount -ge 200) {
                $results["Good"] += $fileInfo
            }
            else {
                $results["Excellent"] += $fileInfo
            }
        }
    }
}

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "RESULTS SUMMARY" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host ""

# Critical Files (> 500 lines)
if ($results["Critical"].Count -gt 0) {
    Write-Host "🔴 CRITICAL - Files > 500 lines (Needs immediate refactoring):" -ForegroundColor Red
    $results["Critical"] | Sort-Object -Property Lines -Descending | ForEach-Object {
        Write-Host ("  {0,5} lines  {1}" -f $_.Lines, $_.Path) -ForegroundColor Red
    }
    Write-Host ""
}

# Warning Files (300-500 lines)
if ($results["Warning"].Count -gt 0) {
    Write-Host "⚠️  WARNING - Files 300-500 lines (Should be refactored):" -ForegroundColor Yellow
    $results["Warning"] | Sort-Object -Property Lines -Descending | ForEach-Object {
        Write-Host ("  {0,5} lines  {1}" -f $_.Lines, $_.Path) -ForegroundColor Yellow
    }
    Write-Host ""
}

# Good Files (200-300 lines)
if ($results["Good"].Count -gt 0) {
    Write-Host " GOOD - Files 200-300 lines (Target range):" -ForegroundColor Green
    $results["Good"] | Sort-Object -Property Lines -Descending | ForEach-Object {
        Write-Host ("  {0,5} lines  {1}" -f $_.Lines, $_.Path) -ForegroundColor Green
    }
    Write-Host ""
}

# Excellent Files (< 200 lines)
Write-Host "⭐ EXCELLENT - Files < 200 lines:" -ForegroundColor Cyan
Write-Host "  Total: $($results['Excellent'].Count) files" -ForegroundColor Cyan
Write-Host ""

# Statistics
$totalFiles = $results["Critical"].Count + $results["Warning"].Count + $results["Good"].Count + $results["Excellent"].Count
$problematicFiles = $results["Critical"].Count + $results["Warning"].Count

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "STATISTICS" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "Total Python files analyzed: $totalFiles"
Write-Host "Files needing refactoring: $problematicFiles" -ForegroundColor $(if ($problematicFiles -gt 0) { "Red" } else { "Green" })
Write-Host "Files in target range (200-300): $($results['Good'].Count)" -ForegroundColor Green
Write-Host "Files under 200 lines: $($results['Excellent'].Count)" -ForegroundColor Cyan
Write-Host ""

if ($problematicFiles -gt 0) {
    Write-Host "⚠️  Action Required: $problematicFiles files need refactoring" -ForegroundColor Yellow
} else {
    Write-Host " All files meet the size requirements!" -ForegroundColor Green
}

Write-Host "==============================================================" -ForegroundColor Cyan
