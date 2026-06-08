import os
import sys
import torch
import numpy as np
import cv2
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class DepthAnythingV2Wrapper:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Depth Anything V2 model: {config.DEPTH_MODEL_HF}")
        
        # Load the Hugging Face Auto model and image processor
        self.image_processor = AutoImageProcessor.from_pretrained(config.DEPTH_MODEL_HF)
        self.model = AutoModelForDepthEstimation.from_pretrained(config.DEPTH_MODEL_HF)
        
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def estimate_depth(self, img_bgr):
        """
        Predicts relative depth for a BGR image.
        Returns:
            depth_map (np.ndarray): H x W depth map where higher values mean closer (or further, depending on model).
                                    Specifically, HF Depth Anything V2 outputs inverse depth (higher = closer).
        """
        orig_h, orig_w = img_bgr.shape[:2]
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Preprocess for the Hugging Face model
        inputs = self.image_processor(images=img_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        outputs = self.model(**inputs)
        predicted_depth = outputs.predicted_depth
        
        # Interpolate raw depth map back to original image size
        depth_upscaled = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        ).squeeze(1).squeeze(0)
        
        # Convert to numpy
        depth_map = depth_upscaled.cpu().numpy()
        return depth_map
