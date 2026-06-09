import os
import glob
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import config
from segmentation.dataset import collect_dataset_paths

def main():
    dataset_files = collect_dataset_paths(config.DATASET_DIR)
    
    class_4_files = []
    for d in dataset_files:
        if d['dataset_type'] == 'rdd2022' and d['annotation_path'].endswith('.txt'):
            with open(d['annotation_path'], "r") as f:
                for line in f:
                    if line.startswith("4 "):
                        class_4_files.append(d['image_path'])
                        break
                        
    print(f"Total Class 4 files: {len(class_4_files)}")
    print("Example Class 4 file paths:")
    for f in class_4_files[:20]:
        print("  ", f)

if __name__ == '__main__':
    main()
