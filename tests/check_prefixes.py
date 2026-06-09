import os
import glob
import sys
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import config
from segmentation.dataset import collect_dataset_paths

def main():
    dataset_files = collect_dataset_paths(config.DATASET_DIR)
    prefixes = Counter()
    for d in dataset_files:
        if d['dataset_type'] == 'rdd2022':
            base = os.path.basename(d['image_path'])
            parts = base.split('_')
            if len(parts) > 1:
                prefixes[parts[0]] += 1
            else:
                prefixes['None'] += 1
                
    print("RDD2022 image prefixes:")
    for pref, count in prefixes.most_common():
        print(f"  {pref}: {count}")

if __name__ == '__main__':
    main()
