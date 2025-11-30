"""
Upload audio files to Azure Blob Storage

To get your connection string, run in PowerShell:
az storage account show-connection-string --name africanobjectaudio --resource-group rag-ai-foundations-demo-rg --query connectionString --output tsv

Then set it: $env:AZURE_STORAGE_CONNECTION_STRING = "your-connection-string-here"
Or paste it below in the code.
"""
import os
from azure.storage.blob import BlobServiceClient, ContentSettings
from pathlib import Path

# Get connection string from environment variable or paste it here
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

if not connection_string:
    print("ERROR: AZURE_STORAGE_CONNECTION_STRING environment variable not set!")
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("="*60)
    print("1. Get your connection string by running in a NEW PowerShell window:")
    print("   az storage account show-connection-string --name africanobjectaudio --resource-group rag-ai-foundations-demo-rg --query connectionString --output tsv")
    print("\n2. Then run:")
    print("   $env:AZURE_STORAGE_CONNECTION_STRING = \"<paste-connection-string-here>\"")
    print("\n3. Then run this script again:")
    print("   python upload_audio_to_azure.py")
    print("="*60)
    exit(1)

try:
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
except Exception as e:
    print(f"ERROR: Failed to create BlobServiceClient: {e}")
    exit(1)

# Create containers
containers = [
    ('nufi-phrasebook-audio', 'audio/nufi_phrasebook_audio'),
    ('word-dictionary-audio', 'audio/word_dictionary')
]

for container_name, local_path in containers:
    print(f"\n{'='*60}")
    print(f"Processing: {container_name}")
    print(f"{'='*60}")
    
    # Create container if it doesn't exist
    try:
        container_client = blob_service_client.create_container(
            container_name,
            public_access='blob'
        )
        print(f"✓ Container '{container_name}' created")
    except Exception as e:
        if "ContainerAlreadyExists" in str(e):
            print(f"✓ Container '{container_name}' already exists")
            container_client = blob_service_client.get_container_client(container_name)
        else:
            print(f"✗ Error creating container: {e}")
            continue
    
    # Upload files
    audio_files = list(Path(local_path).rglob('*.mp3'))
    total_files = len(audio_files)
    print(f"Found {total_files} audio files to upload")
    
    uploaded = 0
    failed = 0
    
    for idx, file_path in enumerate(audio_files, 1):
        # Get relative path to preserve folder structure
        relative_path = file_path.relative_to(local_path)
        blob_name = str(relative_path).replace('\\', '/')
        
        try:
            blob_client = container_client.get_blob_client(blob_name)
            
            # Check if blob already exists
            if blob_client.exists():
                if idx % 100 == 0:
                    print(f"  [{idx}/{total_files}] Skipping (exists): {blob_name}")
                uploaded += 1
                continue
            
            # Upload with content type
            with open(file_path, 'rb') as data:
                blob_client.upload_blob(
                    data,
                    content_settings=ContentSettings(content_type='audio/mpeg')
                )
            
            uploaded += 1
            if idx % 100 == 0:
                print(f"  [{idx}/{total_files}] Uploaded: {blob_name}")
                
        except Exception as e:
            failed += 1
            print(f"  ✗ Failed to upload {blob_name}: {e}")
    
    print(f"\n✓ {container_name}: {uploaded} uploaded, {failed} failed")

print(f"\n{'='*60}")
print("Upload complete!")
print(f"{'='*60}")
