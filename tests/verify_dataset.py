import os
import sys
import numpy as np
import torch

# Add parent directory and src directory to python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'src'))

import config
from segmentation.dataset import collect_dataset_paths, RoadHazardDataset

def main():
    print("--- Running Dataset Scanning and Loading Verification ---")
    print(f"Dataset Root Directory: {config.DATASET_DIR}")
    
    # 1. Collect paths
    dataset_files = collect_dataset_paths(config.DATASET_DIR)
    print(f"Total dataset pairs found: {len(dataset_files)}")
    
    if len(dataset_files) == 0:
        print("ERROR: Scanned 0 dataset pairs!")
        sys.exit(1)
        
    # Categorize dataset types
    rdd_count = sum(1 for d in dataset_files if d['dataset_type'] == 'rdd2022')
    p600_count = sum(1 for d in dataset_files if d['dataset_type'] == 'pothole600')
    cf_count = sum(1 for d in dataset_files if d['dataset_type'] == 'crackforest')
    
    print(f"  - RDD2022: {rdd_count} pairs")
    print(f"  - Pothole-600: {p600_count} pairs")
    print(f"  - CrackForest: {cf_count} pairs")
    
    # 2. Instantiate Dataset
    try:
        dataset = RoadHazardDataset(dataset_files, img_size=(512, 512))
        print("RoadHazardDataset successfully instantiated.")
    except Exception as e:
        print(f"ERROR instantiating RoadHazardDataset: {e}")
        sys.exit(1)
        
    # 3. Load a few samples and check shapes & classes
    # We will pick a few indices representing different datasets
    rdd_indices = [i for i, d in enumerate(dataset_files) if d['dataset_type'] == 'rdd2022'][:3]
    p600_indices = [i for i, d in enumerate(dataset_files) if d['dataset_type'] == 'pothole600'][:3]
    
    test_indices = rdd_indices + p600_indices
    
    print(f"\nTesting loading of {len(test_indices)} sample items...")
    success = True
    for idx in test_indices:
        item_info = dataset_files[idx]
        print(f"\nIndex {idx}: type={item_info['dataset_type']}, img={os.path.basename(item_info['image_path'])}")
        try:
            img_tensor, mask_tensor = dataset[idx]
            unique_classes = torch.unique(mask_tensor).tolist()
            print(f"  - Image tensor shape: {img_tensor.shape}")
            print(f"  - Mask tensor shape: {mask_tensor.shape}")
            print(f"  - Mask unique classes: {unique_classes}")
            
            # Basic sanity checks
            if img_tensor.shape != (3, 512, 512) or mask_tensor.shape != (512, 512):
                print("  [FAILED] Unexpected tensor shapes!")
                success = False
            for cls in unique_classes:
                if cls not in config.CLASS_MAP:
                    print(f"  [FAILED] Found invalid class index {cls} in mask!")
                    success = False
        except Exception as e:
            print(f"  [FAILED] Error loading item: {e}")
            success = False
            
    if success:
        print("\n--- ALL SANITY CHECKS PASSED SUCCESSFULLY ---")
        sys.exit(0)
    else:
        print("\n--- SOME SANITY CHECKS FAILED ---")
        sys.exit(1)

if __name__ == "__main__":
    main()
