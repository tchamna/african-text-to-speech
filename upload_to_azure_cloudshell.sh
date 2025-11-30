#!/bin/bash
# Upload audio files to Azure Blob Storage from Cloud Shell
# This script should be run in Azure Cloud Shell

STORAGE_ACCOUNT="africanobjectaudio"
RESOURCE_GROUP="rag-ai-foundations-demo-rg"

echo "================================================"
echo "Azure Audio Upload Script"
echo "================================================"
echo "Storage Account: $STORAGE_ACCOUNT"
echo "Resource Group: $RESOURCE_GROUP"
echo ""

# Create containers
echo "Creating containers..."
az storage container create \
    --name nufi-phrasebook-audio \
    --account-name $STORAGE_ACCOUNT \
    --public-access blob \
    --auth-mode login

az storage container create \
    --name word-dictionary-audio \
    --account-name $STORAGE_ACCOUNT \
    --public-access blob \
    --auth-mode login

echo ""
echo "================================================"
echo "Containers created successfully!"
echo "================================================"
echo ""
echo "Now upload your local audio files:"
echo ""
echo "1. Upload phrasebook audio (2,030 files):"
echo "   az storage blob upload-batch \\"
echo "     --destination nufi-phrasebook-audio \\"
echo "     --source audio/nufi_phrasebook_audio \\"
echo "     --account-name $STORAGE_ACCOUNT \\"
echo "     --auth-mode login \\"
echo "     --content-type audio/mpeg"
echo ""
echo "2. Upload word dictionary audio (9,149 files):"
echo "   az storage blob upload-batch \\"
echo "     --destination word-dictionary-audio \\"
echo "     --source audio/word_dictionary \\"
echo "     --account-name $STORAGE_ACCOUNT \\"
echo "     --auth-mode login \\"
echo "     --content-type audio/mpeg"
echo ""
echo "================================================"
echo "NOTE: You need to upload the 'audio' folder to Cloud Shell first"
echo "Use the Cloud Shell upload button or drag & drop"
echo "================================================"
