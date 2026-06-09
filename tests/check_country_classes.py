import os
import glob
import sys
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import config
from segmentation.dataset import collect_dataset_paths

def main():
    dataset_files = collect_dataset_paths(config.DATASET_DIR)
    
    country_classes = {}
    for d in dataset_files:
        if d['dataset_type'] == 'rdd2022' and d['annotation_path'].endswith('.txt'):
            base = os.path.basename(d['image_path'])
            country = base.split('_')[0]
            if country not in country_classes:
                country_classes[country] = Counter()
                
            with open(d['annotation_path'], "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls = int(parts[0])
                        country_classes[country][cls] += 1
                        
    print("Class distributions by country in RDD2022 YOLO annotations:")
    for country, counts in country_classes.items():
        print(f"  Country: {country}")
        for cls, count in counts.most_common():
            print(f"    Class {cls}: {count}")

if __name__ == '__main__':
    main()
