import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def get_ground_coordinates(x, y):
    """
    Computes the 3D ground coordinates (X_w, Z_w) in meters from image pixel (x, y)
    assuming a flat road surface.
    
    Camera coordinate system:
        Z points forward along optical axis
        X points right
        Y points down
    
    World coordinate system:
        Y_w points up (ground is Y_w = 0)
        X_w points right
        Z_w points forward along road
    
    Args:
        x (float or np.ndarray): Pixel X coordinate(s)
        y (float or np.ndarray): Pixel Y coordinate(s)
    Returns:
        X_w, Z_w (float or np.ndarray): Ground coordinates in meters
    """
    # Pitch angle magnitude in radians (tilted down is positive pitch)
    pitch_rad = np.radians(-config.CAMERA_TILT_DEG)
    H = config.CAMERA_HEIGHT_M
    f = config.CAMERA_FOCAL_LENGTH_PX
    cx = config.PRINCIPAL_POINT_X
    cy = config.PRINCIPAL_POINT_Y
    
    # Ray direction in camera space
    rx_c = (x - cx) / f
    ry_c = (y - cy) / f
    
    # Ray direction in world space (rotated by pitch angle theta around X-axis)
    # Since camera is pitched down, camera Y points downwards and forwards.
    # We rotate the camera ray by pitch_rad around the X-axis:
    # rw_x = rx_c
    # rw_y = -cos(pitch) * ry_c - sin(pitch)
    # rw_z = sin(pitch) * ry_c + cos(pitch)
    rw_x = rx_c
    rw_y = -np.cos(pitch_rad) * ry_c - np.sin(pitch_rad)
    rw_z = np.sin(pitch_rad) * ry_c + np.cos(pitch_rad)
    
    # Ray: P_w(t) = [0, H, 0] + t * r_w
    # Ground plane is Y_w = 0 => H + t * rw_y = 0 => t = -H / rw_y
    # We only care about intersections in front of the camera (rw_y < 0)
    if isinstance(y, np.ndarray):
        t = np.zeros_like(rw_y)
        valid = rw_y < -1e-5
        t[valid] = -H / rw_y[valid]
        t[~valid] = 100.0  # Pointing to/above horizon: set to a large value (100m)
        
        X_w = t * rw_x
        Z_w = t * rw_z
    else:
        if rw_y < -1e-5:
            t = -H / rw_y
        else:
            t = 100.0  # Above horizon
        X_w = t * rw_x
        Z_w = t * rw_z
        
    return X_w, Z_w

def get_ground_distance_map(width, height):
    """Generates a map of shape (height, width) with the physical ground distance Z_w at each pixel."""
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)
    _, Z_w = get_ground_coordinates(xx, yy)
    return Z_w
