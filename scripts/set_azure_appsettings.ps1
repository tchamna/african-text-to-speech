<#
PowerShell helper: read Azure Web App name and Resource Group from repo files
and run `az webapp config appsettings set` with recommended app settings.

Usage:
  .\scripts\set_azure_appsettings.ps1

This script tries to read:
 - `.github/workflows/azure-deploy-cli.yml` for `AZURE_WEBAPP_NAME` and `--resource-group`
 - `upload_to_azure_cloudshell.sh` for `RESOURCE_GROUP` as a fallback

It will prompt before executing the `az` command.
#>

param(
    [switch]$WhatIf
)

# --- Configuration (change defaults here if you want) ---
$defaultSettings = @{
    APP_MODE = 'production'
    WHISPER_MODEL_SIZE = 'small'
    WHISPER_LOAD_EXTRA_MODELS = '0'
    WHISPER_FALLBACK_MODEL = 'small'
}

# --- Discover repo root and candidate files ---
$repoRoot = (Get-Location).ProviderPath
$workflowFile = Join-Path $repoRoot '.github\workflows\azure-deploy-cli.yml'
$cloudShellScript = Join-Path $repoRoot 'upload_to_azure_cloudshell.sh'

$foundAppName = $null
$foundResourceGroup = $null

if (Test-Path $workflowFile) {
    Write-Host "Reading workflow file: $workflowFile"
    $wf = Get-Content $workflowFile -Raw
    if ($wf -match 'AZURE_WEBAPP_NAME:\s*(\S+)') { $foundAppName = $Matches[1] }
    if (-not $foundResourceGroup -and $wf -match '--resource-group\s+(\S+)') { $foundResourceGroup = $Matches[1] }
}

if (-not $foundResourceGroup -and Test-Path $cloudShellScript) {
    Write-Host "Reading Cloud Shell helper: $cloudShellScript"
    $sh = Get-Content $cloudShellScript -Raw
    if ($sh -match 'RESOURCE_GROUP\s*=\s*"?([^\"\s]+)"?') { $foundResourceGroup = $Matches[1] }
}

# Prompt user for any missing values
if (-not $foundAppName) {
    $foundAppName = Read-Host "AZURE_WEBAPP_NAME not found in repo. Enter Web App name (AZURE_WEBAPP_NAME)"
}
if (-not $foundResourceGroup) {
    $foundResourceGroup = Read-Host "Resource Group not found in repo. Enter RESOURCE_GROUP"
}

Write-Host "\nDetected values:" -ForegroundColor Cyan
Write-Host "  Web App Name : $foundAppName"
Write-Host "  Resource Group: $foundResourceGroup\n"

# Show the settings we will apply
Write-Host "The following app settings will be set on the Web App:" -ForegroundColor Cyan
foreach ($k in $defaultSettings.Keys) { Write-Host "  $k = $($defaultSettings[$k])" }

 $ok = Read-Host "Proceed to apply these settings to the Web App? (type 'yes' to continue)"
 if ($ok -ne 'yes') { Write-Host 'Aborted by user.'; exit 1 }

 # Build --settings argument list
 $settingsArg = $defaultSettings.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" } -join ' '

 # Show the exact command for review
 $fullAzCmd = "az webapp config appsettings set --name $foundAppName --resource-group $foundResourceGroup --settings $settingsArg"
 Write-Host "\nPrepared command:" -ForegroundColor Cyan
 Write-Host $fullAzCmd -ForegroundColor Yellow

 if ($WhatIf) {
    Write-Host "WhatIf specified: skipping execution." -ForegroundColor Green
    Write-Host "Tip: Remove -WhatIf to actually apply the settings." -ForegroundColor Gray
    exit 0
 }

 # Ensure az CLI is available
 if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Azure CLI 'az' not found in PATH. Install CLI or run this in Azure Cloud Shell." -ForegroundColor Red
    exit 2
 }

 # Execute the command
 $cmd = @( 'webapp', 'config', 'appsettings', 'set', '--name', $foundAppName, '--resource-group', $foundResourceGroup, '--settings' ) + $settingsArg
 try {
    az @cmd
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ App settings updated successfully." -ForegroundColor Green
        Write-Host "Tip: Verify in Azure Portal or with: az webapp config appsettings list --name $foundAppName --resource-group $foundResourceGroup"
    } else {
        Write-Host "az exited with code $LASTEXITCODE" -ForegroundColor Red
    }
 } catch {
    Write-Host "Failed to run az CLI: $_" -ForegroundColor Red
    exit 3
 }
