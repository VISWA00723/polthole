import os
import glob
import xml.etree.ElementTree as ET
import sys
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import config
from segmentation.dataset import collect_dataset_paths

def main():
    dataset_files = collect_dataset_paths(config.DATASET_DIR)
    xml_paths = [d['annotation_path'] for d in dataset_files if d['dataset_type'] == 'rdd2022' and d['annotation_path'].endswith('.xml')]
    
    print(f"Checking {len(xml_paths)} XML files...")
    
    name_counter = Counter()
    for xml_path in xml_paths[:5000]:  # Check first 5000
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text.strip()
                name_counter[name] += 1
        except Exception as e:
            pass
            
    print("Unique names found in first 5000 XMLs:")
    for name, count in name_counter.most_common():
        print(f"  {name}: {count}")

if __name__ == '__main__':
    main()
