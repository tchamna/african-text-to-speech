# Upload Audio Files to Azure Blob Storage with Public Access
# This allows the app to reference audio files via direct URLs

$STORAGE_ACCOUNT = "africanobjectaudio"

Write-Host "============================================================"
Write-Host "Uploading Audio Files to Azure Blob Storage"
Write-Host "============================================================"
Write-Host ""

# Create containers with public blob access
Write-Host "Creating containers with public access..."
az storage container create `
    --name nufi-phrasebook-audio `
    --account-name $STORAGE_ACCOUNT `
    --public-access blob `
    --auth-mode key

az storage container create `
    --name word-dictionary-audio `
    --account-name $STORAGE_ACCOUNT `
    --public-access blob `
    --auth-mode key

Write-Host ""
Write-Host "Uploading phrasebook audio files (2,030 files)..."
az storage blob upload-batch `
    --destination nufi-phrasebook-audio `
    --source "audio/nufi_phrasebook_audio" `
    --account-name $STORAGE_ACCOUNT `
    --auth-mode key `
    --content-type "audio/mpeg" `
    --overwrite

Write-Host ""
Write-Host "Uploading word dictionary audio files (9,149 files)..."
az storage blob upload-batch `
    --destination word-dictionary-audio `
    --source "audio/word_dictionary" `
    --account-name $STORAGE_ACCOUNT `
    --auth-mode key `
    --content-type "audio/mpeg" `
    --overwrite

Write-Host ""
Write-Host "============================================================"
Write-Host "Upload complete!"
Write-Host "============================================================"
Write-Host ""
Write-Host "Audio files are now publicly accessible via:"
Write-Host "  Phrasebook: https://africanobjectaudio.blob.core.windows.net/nufi-phrasebook-audio/"
Write-Host "  Dictionary: https://africanobjectaudio.blob.core.windows.net/word-dictionary-audio/"
