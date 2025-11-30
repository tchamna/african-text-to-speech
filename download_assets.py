"""
Download private assets from Azure Blob Storage at container startup.
This keeps sensitive data out of git while making it available in production.
"""
import os
import sys

def download_assets():
    """
    Download assets from Azure Blob Storage if they don't exist locally.
    Set AZURE_STORAGE_CONNECTION_STRING as an environment variable.
    """
    assets_dir = 'assets'
    required_files = [
        'faiss_index.bin',
        'index_mapping.pkl',
        'Nufi_Francais_phrasebook.csv',
        'nufi_dictionary_dictionnaire_audio_maping.csv'
    ]
    
    # Check if assets already exist (for local development)
    all_exist = all(os.path.exists(os.path.join(assets_dir, f)) for f in required_files)
    if all_exist:
        print("✓ All assets found locally")
        return True
    
    # Try to download from Azure Storage
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    if not connection_string:
        print("⚠ WARNING: Assets not found and AZURE_STORAGE_CONNECTION_STRING not set")
        print("⚠ App will run but features requiring assets will fail")
        return False
    
    try:
        from azure.storage.blob import BlobServiceClient
        
        print("Downloading assets from Azure Blob Storage...")
        blob_service = BlobServiceClient.from_connection_string(connection_string)
        container_name = 'african-speaks-assets'  # Change this to your container name
        container_client = blob_service.get_container_client(container_name)
        
        os.makedirs(assets_dir, exist_ok=True)
        
        for filename in required_files:
            blob_client = container_client.get_blob_client(filename)
            download_path = os.path.join(assets_dir, filename)
            
            print(f"  Downloading {filename}...")
            with open(download_path, 'wb') as f:
                f.write(blob_client.download_blob().readall())
            print(f"  ✓ {filename} downloaded")
        
        print("✓ All assets downloaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading assets: {e}")
        return False

if __name__ == '__main__':
    success = download_assets()
    sys.exit(0 if success else 1)
