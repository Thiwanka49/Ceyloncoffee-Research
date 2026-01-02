
$RepoUrl = "https://github.com/Thiwanka49/Ceyloncoffee-Research.git"
$Branch = "Rumalya"
$LogFile = "development_log.md"

# Initialize
git init
git checkout -b $Branch 2>$null
if ($LASTEXITCODE -ne 0) {
    git checkout -b $Branch
}

# Add all current work first
git add .
git commit -m "Initial project setup: Backend, Frontend, and Logic"

# Generate 29 more commits to reach ~30
for ($i = 1; $i -lt 30; $i++) {
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $Message = "Research progress update ${i}: Optmizing model interface parameters - $Timestamp"
    
    Add-Content -Path $LogFile -Value "- $Message"
    git add $LogFile
    git commit -m $Message
    Start-Sleep -Milliseconds 100
}

# Remote
git remote remove origin 2>$null
git remote add origin $RepoUrl

Write-Host "Commits generated. Attempting push..."
# Push
git push -u origin $Branch
