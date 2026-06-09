import os
import glob
import sys
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import config
from segmentation.dataset import collect_dataset_paths

def main():
    dataset_files = collect_dataset_paths(config.DATASET_DIR)
    txt_paths = [d['annotation_path'] for d in dataset_files if d['dataset_type'] == 'rdd2022' and d['annotation_path'].endswith('.txt')]
    
    print(f"Checking {len(txt_paths)} YOLO annotation files...")
    
    class_counter = Counter()
    for txt_path in txt_paths:
        try:
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counter[class_id] += 1
        except Exception as e:
            pass
            
    print("Unique classes found in YOLO txt files:")
    for cls, count in class_counter.most_common():
        print(f"  Class {cls}: {count}")

if __name__ == '__main__':
    main()
