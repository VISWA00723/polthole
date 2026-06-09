import os
import sys
import torch
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import config
from segmentation.dataset import collect_dataset_paths, RoadHazardDataset

def main():
    dataset_files = collect_dataset_paths(config.DATASET_DIR)
    dataset = RoadHazardDataset(dataset_files, img_size=(512, 512))
    
    print("Scanning first 500 masks in dataset...")
    class_counts = Counter()
    for i in range(500):
        _, mask = dataset[i]
        unique = torch.unique(mask).tolist()
        class_counts.update(unique)
        
    print("Class frequency in first 500 masks (number of masks containing this class):")
    for cls, count in class_counts.items():
        print(f"  Class {cls} ({config.CLASS_MAP[cls]}): {count}")

if __name__ == '__main__':
    main()
