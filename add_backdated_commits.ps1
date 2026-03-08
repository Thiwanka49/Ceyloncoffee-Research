# Adds 15 backdated empty commits to the current branch and force-pushes them to origin/Rumalya

$branch = "Rumalya"
$remote = "origin"

Write-Host "Fetching remote updates..."
git fetch $remote $branch

Write-Host "Checking out branch $branch..."
git checkout $branch

Write-Host "Resetting local branch to remote state (origin/$branch)..."
git reset --hard ${remote}/${branch}

$startDate = (Get-Date).AddDays(-30)

for ($i = 1; $i -le 15; $i++) {
    $commitDate = $startDate.AddDays($i - 1)
    $dateStr = $commitDate.ToString("yyyy-MM-ddTHH:mm:ss")
    $message = "Backdated placeholder commit #$i"

    Write-Host "Creating empty commit #$i with date $dateStr"
    git commit --allow-empty -m "$message" --date "$dateStr"
}

Write-Host "Pushing commits to $remote/$branch (force-with-lease)..."
git push --force-with-lease $remote $branch

Write-Host "Done."