import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def calculate_risk_score(area_m2, depth_cm, distance_m, bike_speed_kmh, hazard_class_id):
    """
    Computes a dynamic risk score between 0 and 100.
    
    Args:
        area_m2 (float): Physical area of the hazard in m^2.
        depth_cm (float): Physical depth of the hazard in cm.
        distance_m (float): Distance to the hazard in meters.
        bike_speed_kmh (float): Current speed of the bicycle in km/h.
        hazard_class_id (int): Category of the hazard (1: pothole, 2: crack, etc.)
    Returns:
        risk_score (float): Calculated risk score in [0, 100]
        risk_level (str): Categorized risk level ('Low', 'Medium', 'High', 'Critical')
    """
    # 1. Convert bicycle speed to m/s
    speed_ms = (bike_speed_kmh * 1000.0) / 3600.0
    
    # 2. Normalize hazard severity parameters
    area_norm = min(area_m2 / config.MAX_HAZARD_AREA_M2, 1.0)
    depth_norm = min(depth_cm / config.MAX_HAZARD_DEPTH_CM, 1.0)
    
    # Class-specific adjustments for severity
    # Cracks have lower depth weight, potholes have higher depth weight
    if hazard_class_id == 2:  # Crack
        w_area = 0.8
        w_depth = 0.2
    else:  # Potholes, Water Potholes, Manholes
        w_area = config.WEIGHT_AREA
        w_depth = config.WEIGHT_DEPTH
        
    severity = w_area * area_norm + w_depth * depth_norm
    
    # 3. Calculate Time-to-Collision (TTC)
    # If bike is stationary, collision risk is zero
    if speed_ms < 0.1:
        return 0.0, 'Low'
        
    ttc = distance_m / speed_ms
    
    # 4. Compute Proximity urgency using exponential decay
    # Tau defines how quickly risk increases as we get closer
    proximity = np.exp(-ttc / config.TTC_TAU)
    
    # 5. Calculate final Risk Score
    risk_score = severity * proximity * 100.0
    risk_score = float(np.clip(risk_score, 0.0, 100.0))
    
    # Categorize Risk Level
    if risk_score < 25.0:
        risk_level = 'Low'
    elif risk_score < 55.0:
        risk_level = 'Medium'
    elif risk_score < 80.0:
        risk_level = 'High'
    else:
        risk_level = 'Critical'
        
    return round(risk_score, 1), risk_level
