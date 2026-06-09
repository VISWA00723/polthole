import os
import glob

def main():
    txt_files = glob.glob("d:/road_hazard_datasets/**/*.txt", recursive=True)
    print("Found txt files:", len(txt_files))
    # Filter for non-annotation files (e.g., classes.txt, README.txt, etc.)
    non_annos = [f for f in txt_files if not os.path.basename(f).replace('.txt', '').isdigit() and "labels" not in f]
    print("Non-annotation text files:")
    for f in non_annos[:10]:
        print("  ", f)
        
    yaml_files = glob.glob("d:/road_hazard_datasets/**/*.yaml", recursive=True)
    print("Found yaml files:", len(yaml_files))
    for f in yaml_files:
        print("  ", f)

if __name__ == '__main__':
    main()
