import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from segmentation.model import get_segformer_model, MultiTaskRoadHazardModel

class RoadDamageSegmenter:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = get_segformer_model(pretrained=True)
        self.model = MultiTaskRoadHazardModel(base_model, hidden_dim=512)
        
        # Load weights if available
        if os.path.exists(config.SEGMENTATION_WEIGHTS):
            print(f"Loading custom segmentation weights from: {config.SEGMENTATION_WEIGHTS}")
            self.model.load_state_dict(torch.load(config.SEGMENTATION_WEIGHTS, map_location=self.device))
        else:
            print("WARNING: Custom weights not found. Using pretrained backbone for inference.")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Mean and standard deviation for Segformer
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

    def preprocess(self, img_bgr):
        """Converts BGR image to normalized PyTorch tensor resized to 512x512."""
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Resize to SegFormer expected size
        img_resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        # Normalize
        img_tensor = torch.tensor(img_resized, dtype=torch.float32, device=self.device).permute(2, 0, 1) / 255.0
        img_tensor = (img_tensor - self.mean) / self.std
        return img_tensor.unsqueeze(0)  # Add batch dimension (1, 3, 512, 512)

    @torch.no_grad()
    def segment(self, img_bgr):
        """
        Segments road damage classes.
        Returns:
            mask (np.ndarray): H x W array with class IDs (0-4) resized back to input resolution.
            hazards (list): List of detected hazards, each a dict:
                {
                    'class_id': int,
                    'label': str,
                    'mask': np.ndarray (binary mask of this hazard at original size),
                    'area_ratio': float (fraction of image size),
                    'bbox': (xmin, ymin, xmax, ymax)
                }
        """
        orig_h, orig_w = img_bgr.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess(img_bgr)
        
        # Inference
        logits, severity_logits, road_score, risk_score = self.model(input_tensor)
        
        # Store latest multi-task predictions for access by downstream pipeline
        self.latest_severity = int(torch.argmax(severity_logits, dim=1).item())
        self.latest_road_score = float(road_score.item())
        self.latest_risk_score = float(risk_score.item())
        
        # Interpolate logits back to original image size
        logits_upscaled = F.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        )
        
        # Get predictions
        preds = torch.argmax(logits_upscaled, dim=1).squeeze(0).cpu().numpy()
        
        # Extract individual hazards by finding connected components on the mask
        hazards = []
        
        # Classes: 1: pothole, 2: crack, 3: water_pothole, 4: manhole
        for class_id in range(1, config.NUM_CLASSES):
            class_mask = (preds == class_id).astype(np.uint8)
            if class_mask.sum() == 0:
                continue
                
            # Find connected components for this specific hazard type
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask)
            
            # Label 0 is background
            for label_idx in range(1, num_labels):
                area = stats[label_idx, cv2.CC_STAT_AREA]
                
                # Filter out very small noise (less than 50 pixels)
                if area < 50:
                    continue
                    
                xmin = stats[label_idx, cv2.CC_STAT_LEFT]
                ymin = stats[label_idx, cv2.CC_STAT_TOP]
                w = stats[label_idx, cv2.CC_STAT_WIDTH]
                h = stats[label_idx, cv2.CC_STAT_HEIGHT]
                
                hazard_mask = (labels == label_idx).astype(np.uint8)
                
                hazards.append({
                    'class_id': class_id,
                    'label': config.CLASS_MAP[class_id],
                    'mask': hazard_mask,
                    'area_ratio': float(area / (orig_h * orig_w)),
                    'pixel_area': int(area),
                    'bbox': (xmin, ymin, xmin + w, ymin + h),
                    'centroid': (float(centroids[label_idx][0]), float(centroids[label_idx][1]))
                })
                
        return preds, hazards
