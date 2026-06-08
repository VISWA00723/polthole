import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class RoadHazardDataset(Dataset):
    """
    Unified PyTorch Dataset for Road Hazard Segmentation.
    Supports on-the-fly mask generation for RDD2022 (from XML bboxes),
    CrackForest (pixel masks), Pothole-600 (binary masks), and custom PNG masks.
    """
    def __init__(self, data_list, transform=None, img_size=(512, 512)):
        """
        Args:
            data_list (list): List of dicts, each containing:
                - 'image_path': path to RGB image
                - 'annotation_path': path to XML (for RDD2022) or mask image (for Pothole-600/CrackForest)
                - 'dataset_type': 'rdd2022', 'crackforest', 'pothole600', or 'custom'
            transform: torchvision transforms or albumentations
            img_size (tuple): Target size (width, height)
        """
        self.data_list = data_list
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.data_list)

    def _generate_mask_from_xml(self, xml_path, img_w, img_h):
        """Converts RDD2022 XML bounding boxes to a segmentation mask."""
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if not os.path.exists(xml_path):
            return mask
            
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            name = obj.find('name').text.strip()
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            
            # Map RDD2022 classes:
            # D00 (Longitudinal Crack) -> 2 (Crack)
            # D10 (Transverse Crack)   -> 2 (Crack)
            # D20 (Alligator Crack)    -> 2 (Crack)
            # D40 (Pothole)            -> 1 (Pothole)
            # D43 (Water Pothole)      -> 3 (Water Pothole) - if present
            # Repair / Manhole etc     -> 4
            class_id = 0
            if name in ["D00", "D10", "D20", "Crack"]:
                class_id = 2  # Crack
            elif name in ["D40", "Pothole"]:
                class_id = 1  # Pothole
            elif name in ["D43", "Water Pothole", "water_pothole"]:
                class_id = 3  # Water Pothole
            elif name in ["Manhole", "manhole", "D50"]:
                class_id = 4  # Manhole
                
            if class_id > 0:
                cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), class_id, -1)
                
        return mask

    def _generate_mask_from_yolo(self, txt_path, img_w, img_h):
        """Converts RDD2022 YOLO bounding boxes to a segmentation mask."""
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if not os.path.exists(txt_path):
            return mask
            
        with open(txt_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id_yolo = int(parts[0])
                x_center = float(parts[1]) * img_w
                y_center = float(parts[2]) * img_h
                w = float(parts[3]) * img_w
                h = float(parts[4]) * img_h
                
                xmin = int(x_center - w / 2)
                ymin = int(y_center - h / 2)
                xmax = int(x_center + w / 2)
                ymax = int(y_center + h / 2)
                
                # Clip to image boundaries
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(img_w - 1, xmax)
                ymax = min(img_h - 1, ymax)
                
                # Map YOLO classes to target:
                # 0 (longitudinal crack), 1 (transverse crack), 2 (alligator crack) -> 2 (Crack)
                # 4 (pothole) -> 1 (Pothole)
                # Others (e.g. 3, other corruption) -> Ignore (0)
                class_id = 0
                if class_id_yolo in [0, 1, 2]:
                    class_id = 2  # Crack
                elif class_id_yolo == 4:
                    class_id = 1  # Pothole
                    
                if class_id > 0:
                    cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), class_id, -1)
            except Exception:
                continue
                
        return mask

    def _load_pixel_mask(self, mask_path, dataset_type, img_w, img_h):
        """Loads and processes pixel-level masks for CrackForest / Pothole-600."""
        if not os.path.exists(mask_path):
            return np.zeros((img_h, img_w), dtype=np.uint8)
            
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.resize(mask_img, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if dataset_type == 'pothole600':
            # In Pothole-600, potholes are typically white (255) or > 0
            mask[mask_img > 127] = 1  # Pothole
        elif dataset_type == 'crackforest':
            # CrackForest labels
            mask[mask_img > 0] = 2    # Crack
        elif dataset_type == 'custom':
            # Custom is already labeled 0-4
            mask = mask_img
            
        return mask

    def __getitem__(self, idx):
        item = self.data_list[idx]
        img_path = item['image_path']
        anno_path = item['annotation_path']
        dtype = item['dataset_type']
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        # Load or generate mask
        if dtype == 'rdd2022':
            if anno_path.endswith('.xml'):
                mask = self._generate_mask_from_xml(anno_path, w, h)
            elif anno_path.endswith('.txt'):
                mask = self._generate_mask_from_yolo(anno_path, w, h)
            else:
                mask = np.zeros((h, w), dtype=np.uint8)
        else:
            mask = self._load_pixel_mask(anno_path, dtype, w, h)
            
        # Resize image and mask to target size
        img = img.resize(self.img_size, Image.BILINEAR)
        mask = Image.fromarray(mask).resize(self.img_size, Image.NEAREST)
        
        img_np = np.array(img)
        mask_np = np.array(mask)
        
        # Apply transformation if available
        if self.transform:
            augmented = self.transform(image=img_np, mask=mask_np)
            img_np = augmented['image']
            mask_np = augmented['mask']
            
        # Standard Hugging Face normalization for SegFormer
        # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
        img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        mask_tensor = torch.tensor(mask_np, dtype=torch.long)
        
        return img_tensor, mask_tensor


def collect_dataset_paths(dataset_dir):
    """
    Scans the dataset folder and registers image/mask pairs.
    Looks for:
    - RDD2022: under rdd2022/*/images and rdd2022/*/annotations/xmls, or RDD_SPLIT format
    - Pothole-600: under pothole600/images and pothole600/masks, or nested split format
    - CrackForest: under crackforest/images and crackforest/masks
    """
    data_list = []
    
    # 1. Scan RDD2022
    rdd_path = os.path.join(dataset_dir, "rdd2022")
    if os.path.exists(rdd_path):
        # 1a. Country subfolders format (e.g. India, Japan with XML annotations)
        countries = [d for d in os.listdir(rdd_path) if os.path.isdir(os.path.join(rdd_path, d))]
        for country in countries:
            if country == "RDD_SPLIT":
                continue
            img_dir = os.path.join(rdd_path, country, "train", "images")
            xml_dir = os.path.join(rdd_path, country, "train", "annotations", "xmls")
            
            if os.path.exists(img_dir) and os.path.exists(xml_dir):
                images = glob.glob(os.path.join(img_dir, "*.jpg"))
                for img_path in images:
                    base = os.path.splitext(os.path.basename(img_path))[0]
                    xml_path = os.path.join(xml_dir, f"{base}.xml")
                    if os.path.exists(xml_path):
                        data_list.append({
                            'image_path': img_path,
                            'annotation_path': xml_path,
                            'dataset_type': 'rdd2022'
                        })
        
        # 1b. RDD_SPLIT YOLO format (.txt annotations)
        rdd_split_path = os.path.join(rdd_path, "RDD_SPLIT")
        if os.path.exists(rdd_split_path):
            for split in ["train", "val", "test"]:
                img_dir = os.path.join(rdd_split_path, split, "images")
                lbl_dir = os.path.join(rdd_split_path, split, "labels")
                if os.path.exists(img_dir) and os.path.exists(lbl_dir):
                    images = glob.glob(os.path.join(img_dir, "*.jpg"))
                    for img_path in images:
                        base = os.path.splitext(os.path.basename(img_path))[0]
                        txt_path = os.path.join(lbl_dir, f"{base}.txt")
                        if os.path.exists(txt_path):
                            data_list.append({
                                'image_path': img_path,
                                'annotation_path': txt_path,
                                'dataset_type': 'rdd2022'
                            })
                        
    # 2. Scan Pothole-600
    p600_path = os.path.join(dataset_dir, "pothole600")
    if os.path.exists(p600_path):
        # 2a. Flat format (images and masks)
        img_dir = os.path.join(p600_path, "images")
        mask_dir = os.path.join(p600_path, "masks")
        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            images = glob.glob(os.path.join(img_dir, "*.*"))
            for img_path in images:
                base = os.path.splitext(os.path.basename(img_path))[0]
                for ext in [".png", ".jpg", ".bmp"]:
                    mask_path = os.path.join(mask_dir, f"{base}{ext}")
                    if os.path.exists(mask_path):
                        data_list.append({
                            'image_path': img_path,
                            'annotation_path': mask_path,
                            'dataset_type': 'pothole600'
                        })
                        break
        
        # 2b. Nested split format (training/validation/testing)
        p600_nested_path = os.path.join(p600_path, "pothole600")
        if os.path.exists(p600_nested_path):
            for split in ["training", "validation", "testing"]:
                img_dir = os.path.join(p600_nested_path, split, "rgb")
                mask_dir = os.path.join(p600_nested_path, split, "label")
                if os.path.exists(img_dir) and os.path.exists(mask_dir):
                    images = glob.glob(os.path.join(img_dir, "*.png"))
                    for img_path in images:
                        base = os.path.splitext(os.path.basename(img_path))[0]
                        mask_path = os.path.join(mask_dir, f"{base}.png")
                        if os.path.exists(mask_path):
                            data_list.append({
                                'image_path': img_path,
                                'annotation_path': mask_path,
                                'dataset_type': 'pothole600'
                            })
                        
    # 3. Scan CrackForest
    cf_path = os.path.join(dataset_dir, "crackforest")
    if os.path.exists(cf_path):
        img_dir = os.path.join(cf_path, "images")
        mask_dir = os.path.join(cf_path, "masks")
        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            images = glob.glob(os.path.join(img_dir, "*.jpg"))
            for img_path in images:
                base = os.path.splitext(os.path.basename(img_path))[0]
                mask_path = os.path.join(mask_dir, f"{base}.png")
                if os.path.exists(mask_path):
                    data_list.append({
                        'image_path': img_path,
                        'annotation_path': mask_path,
                        'dataset_type': 'crackforest'
                    })
                    
    return data_list
