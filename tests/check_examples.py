import os
import glob
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import config
from segmentation.dataset import collect_dataset_paths

def main():
    dataset_files = collect_dataset_paths(config.DATASET_DIR)
    
    print("Class 3 files examples:")
    count3 = 0
    for d in dataset_files:
        if d['dataset_type'] == 'rdd2022' and d['annotation_path'].endswith('.txt'):
            with open(d['annotation_path'], "r") as f:
                for line in f:
                    if line.startswith("3 "):
                        print(f"  {os.path.basename(d['image_path'])}: {line.strip()}")
                        count3 += 1
                        break
            if count3 >= 5:
                break
                
    print("\nClass 4 files examples:")
    count4 = 0
    for d in dataset_files:
        if d['dataset_type'] == 'rdd2022' and d['annotation_path'].endswith('.txt'):
            with open(d['annotation_path'], "r") as f:
                for line in f:
                    if line.startswith("4 "):
                        print(f"  {os.path.basename(d['image_path'])}: {line.strip()}")
                        count4 += 1
                        break
            if count4 >= 5:
                break

if __name__ == '__main__':
    main()
