[CmdletBinding()]
param(
    [string]$CogRoot = $env:ATLAS_COG_ROOT,
    [int]$ApiPort = 8000,
    [int]$WebPort = 5174,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$atlasRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $atlasRoot ".venv\Scripts\python.exe"
$catalog = Join-Path $atlasRoot "public\catalog\cities.json"
$runtime = Join-Path $atlasRoot "runtime"
$logRoot = Join-Path $runtime "logs"
$apiUrl = "http://127.0.0.1:$ApiPort"
$webUrl = "http://127.0.0.1:$WebPort"

function Test-AtlasEndpoint {
    param([Parameter(Mandatory = $true)][string]$Url)
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

function Wait-AtlasEndpoint {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$Attempts = 30
    )
    for ($attempt = 0; $attempt -lt $Attempts; $attempt++) {
        if (Test-AtlasEndpoint -Url $Url) {
            return $true
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

if (-not (Test-AtlasEndpoint -Url "$apiUrl/api/health")) {
    if (-not (Test-Path -LiteralPath $python)) {
        throw "Atlas Python environment is missing: $python"
    }
    if (-not (Test-Path -LiteralPath $catalog)) {
        throw "Atlas city catalogue is missing: $catalog"
    }
    if ([string]::IsNullOrWhiteSpace($CogRoot) -or -not (Test-Path -LiteralPath $CogRoot)) {
        throw "Canopy COG directory is missing. Pass -CogRoot or set ATLAS_COG_ROOT."
    }

    $previousCogRoot = $env:ATLAS_COG_ROOT
    $previousCatalog = $env:ATLAS_CATALOG
    try {
        $env:ATLAS_COG_ROOT = (Resolve-Path -LiteralPath $CogRoot).Path
        $env:ATLAS_CATALOG = (Resolve-Path -LiteralPath $catalog).Path
        Start-Process `
            -FilePath $python `
            -ArgumentList @("-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", $ApiPort) `
            -WorkingDirectory $atlasRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $logRoot "api.stdout.log") `
            -RedirectStandardError (Join-Path $logRoot "api.stderr.log") | Out-Null
    }
    finally {
        $env:ATLAS_COG_ROOT = $previousCogRoot
        $env:ATLAS_CATALOG = $previousCatalog
    }

    if (-not (Wait-AtlasEndpoint -Url "$apiUrl/api/health")) {
        throw "Canopy API did not become ready. Inspect $logRoot\api.stderr.log"
    }
    Write-Host "Canopy API ready: $apiUrl" -ForegroundColor Green
}
else {
    Write-Host "Canopy API already running: $apiUrl" -ForegroundColor DarkGreen
}

if (-not (Test-AtlasEndpoint -Url $webUrl)) {
    $pnpm = (Get-Command pnpm.cmd -ErrorAction Stop).Source
    Start-Process `
        -FilePath $pnpm `
        -ArgumentList @("--ignore-workspace", "run", "dev", "--", "--host", "127.0.0.1", "--port", $WebPort, "--strictPort") `
        -WorkingDirectory $atlasRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logRoot "web.stdout.log") `
        -RedirectStandardError (Join-Path $logRoot "web.stderr.log") | Out-Null

    if (-not (Wait-AtlasEndpoint -Url $webUrl)) {
        throw "Atlas web client did not become ready. Inspect $logRoot\web.stderr.log"
    }
    Write-Host "Atlas web client ready: $webUrl" -ForegroundColor Green
}
else {
    Write-Host "Atlas web client already running: $webUrl" -ForegroundColor DarkGreen
}

if (-not $NoBrowser) {
    Start-Process $webUrl
}

Write-Host "Atlas is ready at $webUrl" -ForegroundColor Cyan
