import os
import sys
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.geometry import get_ground_distance_map, get_ground_coordinates

class RoadHazardEstimator:
    def __init__(self):
        # Precompute the flat road distance map for calibration reference
        self.flat_road_Z = get_ground_distance_map(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)

    def calibrate_depth_map(self, raw_depth, road_mask):
        """
        Calibrates the raw depth map (inverse relative depth) to physical distance Z in meters
        using the flat road distance map as reference.
        
        Fits: raw_depth = alpha / Z_flat + beta
        => Z_calibrated = alpha / (raw_depth - beta)
        
        Args:
            raw_depth (np.ndarray): H x W relative depth map (higher values are closer)
            road_mask (np.ndarray): H x W binary mask where 1 indicates flat road background
        Returns:
            calibrated_Z (np.ndarray): H x W physical distance map in meters
        """
        h, w = raw_depth.shape[:2]
        
        # Resize flat road reference to match raw depth map shape if different
        if self.flat_road_Z.shape != raw_depth.shape:
            flat_Z = cv2.resize(self.flat_road_Z, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            flat_Z = self.flat_road_Z
            
        # Select calibration points in the lower half of the image (guaranteed to be road)
        # where we don't have hazards.
        lower_half_mask = np.zeros_like(road_mask)
        lower_half_mask[int(h*0.5):, :] = 1
        calib_mask = (road_mask == 0) & (lower_half_mask == 1)  # 0 class is background/road
        
        y_indices, x_indices = np.where(calib_mask)
        
        if len(y_indices) < 100:
            # Fallback to general lower half if mask is too empty
            y_indices, x_indices = np.where(lower_half_mask)
            
        # Sample points to make fitting fast and robust
        sample_indices = np.random.choice(len(y_indices), min(1000, len(y_indices)), replace=False)
        y_samples = y_indices[sample_indices]
        x_samples = x_indices[sample_indices]
        
        d_val = raw_depth[y_samples, x_samples]
        inv_z_val = 1.0 / (flat_Z[y_samples, x_samples] + 1e-5)
        
        # Fit linear model: d = alpha * inv_z + beta
        # We use RANSAC to exclude outliers (other vehicles, curbs, etc.)
        try:
            # Fit line using OpenCV RANSAC or basic least squares
            A = np.vstack([inv_z_val, np.ones_like(inv_z_val)]).T
            # Solve using robust L1/L2 or simple solver
            alpha, beta = np.linalg.lstsq(A, d_val, rcond=None)[0]
            
            # Sanity check: alpha should be positive (as inv_z increases [closer], raw_depth should increase [closer])
            if alpha <= 0:
                # Fallback to simple scaling if fit is inverted
                alpha = np.median(d_val) * np.median(flat_Z[y_samples, x_samples])
                beta = 0.0
        except Exception:
            # Absolute fallback coefficients
            alpha = 1000.0
            beta = 0.0
            
        # Compute calibrated distance: Z = alpha / (raw_depth - beta + epsilon)
        # Prevent division by zero
        calibrated_Z = alpha / (np.maximum(raw_depth - beta, 1e-4) + 1e-5)
        
        # Clip max range to 50 meters for safety
        calibrated_Z = np.clip(calibrated_Z, 0.5, 50.0)
        
        return calibrated_Z

    def estimate_hazard_metrics(self, hazard, calibrated_Z):
        """
        Estimates the distance, depth, and physical area of a detected hazard.
        
        Args:
            hazard (dict): Hazard dict from segmentation inference
            calibrated_Z (np.ndarray): H x W calibrated distance map in meters
        Returns:
            metrics (dict): Dict of physical properties:
                {
                    'distance_m': float,
                    'depth_cm': float,
                    'area_m2': float
                }
        """
        mask = hazard['mask']
        y_indices, x_indices = np.where(mask)
        
        if len(y_indices) == 0:
            return {'distance_m': 99.0, 'depth_cm': 0.0, 'area_m2': 0.0}
            
        # 1. Distance from bike: The closest point of the hazard (minimum distance value)
        hazard_distances = calibrated_Z[mask == 1]
        distance_m = float(np.min(hazard_distances))
        
        # 2. Physical Area: Sum of physical area of each pixel
        # Area of pixel(x, y) = Z^2 / (f^2 * cos(tilt))
        f = config.CAMERA_FOCAL_LENGTH_PX
        cos_tilt = np.cos(np.radians(config.CAMERA_TILT_DEG))
        
        pixel_zs = calibrated_Z[y_indices, x_indices]
        pixel_areas = (pixel_zs ** 2) / (f ** 2 * cos_tilt)
        area_m2 = float(np.sum(pixel_areas))
        
        # 3. Physical Depth (Only applicable to Potholes/Water Potholes)
        depth_cm = 0.0
        if hazard['class_id'] in [1, 3]:  # Pothole or Water Pothole
            # Find the boundary of the hazard mask
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(mask, kernel, iterations=1)
            boundary = dilated - mask
            
            boundary_y, boundary_x = np.where(boundary)
            
            if len(boundary_y) > 0:
                # Calibrated distance at boundary (representing road plane)
                boundary_Z = calibrated_Z[boundary_y, boundary_x]
                
                # Get the flat road reference for the boundary pixels
                # Pothole actual depth is depth_measured - depth_road_flat
                # For each pixel inside the pothole, compare it to the expected road surface distance
                h, w = calibrated_Z.shape[:2]
                if self.flat_road_Z.shape != (h, w):
                    flat_road_Z_resized = cv2.resize(self.flat_road_Z, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    flat_road_Z_resized = self.flat_road_Z
                    
                # The depth is the difference between the calibrated Z inside the pothole
                # and the expected flat road Z at that same pixel coordinate.
                road_plane_Z = flat_road_Z_resized[y_indices, x_indices]
                pothole_measured_Z = calibrated_Z[y_indices, x_indices]
                
                # Depth = (Z_measured - Z_road) * cos(tilt)
                depths_m = (pothole_measured_Z - road_plane_Z) * cos_tilt
                depth_cm = float(np.max(depths_m) * 100.0)  # Convert to cm
                depth_cm = max(0.0, depth_cm)  # Ensure positive depth
            else:
                depth_cm = 0.0
                
        return {
            'distance_m': round(distance_m, 2),
            'depth_cm': round(depth_cm, 1),
            'area_m2': round(area_m2, 3)
        }
