import os
import sys
import zipfile
import subprocess
import shutil

# Ensure config is loaded
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def setup_kaggle_credentials():
    """Sets up kaggle.json in the user's home directory or configures env vars."""
    username = "viswa00723"  # Inferred from github username
    key = config.KAGGLE_KEY
    
    # Also set env variables for the current process
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    
    # Write to ~/.kaggle/kaggle.json
    home = os.path.expanduser("~")
    kaggle_dir = os.path.join(home, ".kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    
    with open(kaggle_json_path, "w") as f:
        f.write(f'{{"username":"{username}","key":"{key}"}}')
    
    # On Unix-like systems, set permissions (Windows will ignore, but good practice)
    try:
        os.chmod(kaggle_json_path, 0o600)
    except Exception:
        pass
    print(f"Kaggle credentials configured in {kaggle_json_path}")

def download_kaggle_dataset(dataset_name, output_dir):
    """Downloads a dataset from Kaggle and extracts it using the Python API."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {dataset_name} to {output_dir}...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        # Download files and unzip them automatically
        api.dataset_download_files(dataset_name, path=output_dir, unzip=True)
        print(f"Successfully downloaded and extracted {dataset_name}.\n")
        return True
    except Exception as e:
        print(f"Error downloading {dataset_name}: {e}")
        print("Please check your Kaggle API key or download the dataset manually from:")
        print(f"https://www.kaggle.com/datasets/{dataset_name}")
        print(f"And extract it to: {output_dir}\n")
        return False

def main():
    print("Road Hazard Intelligence System - Dataset Downloader")
    setup_kaggle_credentials()
    
    # We will try to download RDD2022
    rdd_dir = os.path.join(config.DATASET_DIR, "rdd2022")
    rdd_success = download_kaggle_dataset("aliabdelmenam/rdd-2022", rdd_dir)
    
    # We will try to download Pothole-600
    pothole_dir = os.path.join(config.DATASET_DIR, "pothole600")
    pothole_success = download_kaggle_dataset("rangerfan/pothole-600", pothole_dir)
    
    # Let's print summary
    print("--- Download Summary ---")
    print(f"RDD2022: {'SUCCESS' if rdd_success else 'FAILED (Manual action required)'}")
    print(f"Pothole-600: {'SUCCESS' if pothole_success else 'FAILED (Manual action required)'}")

if __name__ == "__main__":
    main()
