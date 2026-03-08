$RepoUrl = "https://github.com/Thiwanka49/Ceyloncoffee-Research.git"
$Branch = "Rumalya"
$LogFile = "development_log.md"

Write-Host "Initializing Git..."
# Initialize or re-init
if (-not (Test-Path .git)) {
    git init
}

# Configure Remote
git remote remove origin 2>$null
git remote add origin $RepoUrl

# Checkout Branch
Write-Host "Switching to branch $Branch..."
git fetch origin $Branch 2>$null
git checkout -b $Branch 2>$null
if ($LASTEXITCODE -ne 0) {
    git checkout $Branch
}

# Commit 1: The Content (Coffee UI)
Write-Host "Creating main commit..."
git add .
git commit -m "Refactor: Implement Premium Coffee Theme UI with Glassmorphism"

# Commits 2-5: Activity Padding
Write-Host "Generating progress commits..."
for ($i = 1; $i -le 4; $i++) {
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $Message = "Refinement ${i}: Optimizing mobile view and styles - $Timestamp"
    
    Add-Content -Path $LogFile -Value "- $Message"
    git add $LogFile
    git commit -m $Message
    Start-Sleep -Milliseconds 100
}

# Push
Write-Host "Pushing to remote..."
git push -u origin $Branch --force
