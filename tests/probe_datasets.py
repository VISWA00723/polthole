import os
import glob
import sys

def main():
    root = "d:/road_hazard_datasets"
    print("Listing top directories:")
    for d in os.listdir(root):
        print(" ", d)
    
    print("\nSearching for any mask files that might contain class 3 or 4...")
    # Find all unique pixel values in all masks of pothole600
    p600_masks = glob.glob(os.path.join(root, "pothole600", "**", "*.png"), recursive=True)
    print(f"Found {len(p600_masks)} PNG files in pothole600.")
    
    unique_vals = set()
    for p in p600_masks[:100]:
        import cv2
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            unique_vals.update(np.unique(img))
    print("Unique pixel values in first 100 pothole600 masks:", unique_vals)

if __name__ == '__main__':
    import numpy as np
    main()
