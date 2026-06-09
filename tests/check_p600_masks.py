import os
import glob
import cv2
import numpy as np

def main():
    root = "d:/road_hazard_datasets/pothole600"
    masks = glob.glob(os.path.join(root, "**", "*.png"), recursive=True)
    print(f"Checking {len(masks)} masks in pothole600...")
    
    unique_vals = set()
    for m in masks:
        img = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            unique_vals.update(np.unique(img))
            
    print("All unique values in pothole600 masks:", unique_vals)

if __name__ == '__main__':
    main()
