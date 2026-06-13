"""
Download model files from Google Drive
Run: python download_models.py
"""

import os
import gdown
from pathlib import Path

# Create model directory if it doesn't exist
model_dir = Path(__file__).parent / 'model'
model_dir.mkdir(exist_ok=True)

# REPLACE THESE WITH YOUR ACTUAL FILE IDs FROM GOOGLE DRIVE
MODEL_FILES = {
    'best_model.pt': 'YOUR_BEST_MODEL_FILE_ID_HERE',    # ← Replace this
    'final_model.pt': 'YOUR_FINAL_MODEL_FILE_ID_HERE',  # ← Replace this
    'bert_model.pt': 'YOUR_BERT_MODEL_FILE_ID_HERE',    # ← Replace this
}

print("="*60)
print("FAKE NEWS DETECTION - MODEL DOWNLOADER")
print("="*60)

for filename, file_id in MODEL_FILES.items():
    file_path = model_dir / filename
    
    # Check if file already exists
    if file_path.exists():
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ {filename} already exists ({file_size_mb:.1f} MB)")
        continue
    
    # Check if file ID is still placeholder
    if 'YOUR_' in file_id:
        print(f"\n⚠️ Please update the file ID for {filename} in download_models.py")
        print(f"   Get the shareable link from Google Drive and extract the file ID")
        continue
    
    print(f"\n📥 Downloading {filename}...")
    url = f'https://drive.google.com/uc?id={file_id}'
    
    try:
        # Download the file
        gdown.download(url, str(file_path), quiet=False)
        
        # Verify download
        if file_path.exists():
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✓ Successfully downloaded {filename} ({file_size_mb:.1f} MB)")
        else:
            print(f"❌ Failed to download {filename}")
            
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")

print("\n" + "="*60)
print("Download complete! Models saved to 'model/' directory")
print("="*60)

# Verify all files
print("\nVerifying model files:")
for filename in MODEL_FILES.keys():
    file_path = model_dir / filename
    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ {filename} ({size_mb:.1f} MB)")
    else:
        print(f"  ❌ {filename} - MISSING")