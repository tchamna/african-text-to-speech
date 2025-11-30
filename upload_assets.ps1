# Upload Assets to Azure Blob Storage
# Run this script in PowerShell

$STORAGE_ACCOUNT = "africanobjectaudio"
$CONTAINER_NAME = "african-speaks-assets"

Write-Host "============================================================"
Write-Host "Uploading Assets to Azure Blob Storage"
Write-Host "============================================================"
Write-Host ""

# Create container
Write-Host "Creating container: $CONTAINER_NAME"
az storage container create `
    --name $CONTAINER_NAME `
    --account-name $STORAGE_ACCOUNT `
    --public-access off `
    --auth-mode key

Write-Host ""
Write-Host "Uploading assets files..."
Write-Host ""

# Upload all files from assets folder
az storage blob upload-batch `
    --destination $CONTAINER_NAME `
    --source assets `
    --account-name $STORAGE_ACCOUNT `
    --auth-mode key `
    --overwrite

Write-Host ""
Write-Host "============================================================"
Write-Host "Upload complete!"
Write-Host "============================================================"
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Add AZURE_STORAGE_CONNECTION_STRING to your Web App settings"
Write-Host "2. Trigger a new deployment"
