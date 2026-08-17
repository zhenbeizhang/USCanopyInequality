param(
    [Parameter(Mandatory = $true)][string]$Server,
    [Parameter(Mandatory = $true)][string]$KeyPath,
    [Parameter(Mandatory = $true)][string]$CogRoot,
    [string]$RemoteUser = "deploy",
    [string]$RemoteDirectory = "/srv/who-has-shade/data/cogs",
    [ValidateRange(1, 50)][int]$StartIndex = 1,
    [ValidateRange(1, 50)][int]$EndIndex = 50
)

$ErrorActionPreference = "Stop"
$remote = "$RemoteUser@$Server"
$sshArgs = @("-i", $KeyPath, "-o", "StrictHostKeyChecking=accept-new")
$allFiles = Get-ChildItem -LiteralPath $CogRoot -Filter "*.tif" -File | Sort-Object Name

if ($allFiles.Count -ne 50) {
    throw "Expected 50 COG files in $CogRoot, found $($allFiles.Count)."
}
if ($EndIndex -lt $StartIndex) {
    throw "EndIndex must be greater than or equal to StartIndex."
}
$files = @($allFiles | Select-Object -Skip ($StartIndex - 1) -First ($EndIndex - $StartIndex + 1))

$remoteSizes = @{}
$listing = & ssh @sshArgs $remote "find '$RemoteDirectory' -maxdepth 1 -type f -name '*.tif' -printf '%f|%s\n'"
if ($LASTEXITCODE -ne 0) {
    throw "Could not read the remote COG directory."
}

foreach ($line in $listing) {
    $parts = $line -split "\|", 2
    if ($parts.Count -eq 2) {
        $remoteSizes[$parts[0]] = [long]$parts[1]
    }
}

$uploadedBytes = [long]0
$skippedBytes = [long]0
$position = $StartIndex - 1
foreach ($file in $files) {
    $position++
    if ($remoteSizes.ContainsKey($file.Name) -and $remoteSizes[$file.Name] -eq $file.Length) {
        $skippedBytes += $file.Length
        Write-Output ("[{0}/50] SKIP {1} ({2:N1} MiB)" -f $position, $file.Name, ($file.Length / 1MB))
        continue
    }

    & scp @sshArgs $file.FullName "${remote}:$RemoteDirectory/$($file.Name)"
    if ($LASTEXITCODE -ne 0) {
        throw "Upload failed for $($file.Name). Re-run this script to resume."
    }
    $uploadedBytes += $file.Length
    Write-Output ("[{0}/50] SENT {1} ({2:N1} MiB)" -f $position, $file.Name, ($file.Length / 1MB))
}

Write-Output ("COMPLETE uploaded={0:N1} MiB skipped={1:N1} MiB" -f ($uploadedBytes / 1MB), ($skippedBytes / 1MB))
