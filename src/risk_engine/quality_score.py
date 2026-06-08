import os
import sys
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class RoadQualityRegressor:
    def __init__(self):
        self.model_path = config.ROAD_QUALITY_WEIGHTS
        self.model = None
        
        # Load custom trained model if it exists
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Loaded trained Road Quality Regression model from {self.model_path}")
            except Exception as e:
                print(f"Error loading road quality model: {e}. Falling back to formula.")
                self.model = None

    def extract_features(self, hazards):
        """
        Extracts a feature vector from detected hazards list for regression input.
        
        Feature layout:
            [pothole_area_ratio, crack_area_ratio, water_pothole_area_ratio, max_depth, hazard_count]
        """
        pothole_area = 0.0
        crack_area = 0.0
        water_pothole_area = 0.0
        max_depth = 0.0
        hazard_count = len(hazards)
        
        for h in hazards:
            area = h.get('area_ratio', 0.0)
            depth = h.get('depth_cm', 0.0)
            class_id = h.get('class_id', 0)
            
            if class_id == 1:    # Pothole
                pothole_area += area
                max_depth = max(max_depth, depth)
            elif class_id == 2:  # Crack
                crack_area += area
            elif class_id == 3:  # Water Pothole
                water_pothole_area += area
                max_depth = max(max_depth, depth)
                
        return np.array([
            pothole_area,
            crack_area,
            water_pothole_area,
            max_depth,
            float(hazard_count)
        ])

    def calculate_score(self, hazards):
        """
        Predicts the road quality score (0.0 to 10.0).
        If a custom regression model is trained, it uses it.
        Otherwise, it falls back to a calibrated regression formula.
        """
        features = self.extract_features(hazards)
        
        if self.model is not None:
            # Predict using the trained scikit-learn model
            score = float(self.model.predict(features.reshape(1, -1))[0])
        else:
            # Fallback to physical baseline regression equation:
            # Start at perfect score (10.0) and deduct based on hazard severities
            pothole_area, crack_area, water_pothole_area, max_depth, hazard_count = features
            
            # Normalize inputs
            norm_pothole_area = min(pothole_area / 0.15, 1.0)        # Max pothole area coverage of 15%
            norm_crack_area = min(crack_area / 0.25, 1.0)            # Max crack area coverage of 25%
            norm_water_pothole_area = min(water_pothole_area / 0.10, 1.0)
            norm_depth = min(max_depth / config.MAX_HAZARD_DEPTH_CM, 1.0)
            norm_count = min(hazard_count / 10.0, 1.0)              # Max 10 hazards in single frame
            
            # Regression coefficients (deductions)
            deduction = (
                4.5 * norm_pothole_area +
                2.5 * norm_crack_area +
                5.0 * norm_water_pothole_area +
                3.5 * norm_depth +
                1.5 * norm_count
            )
            
            score = 10.0 - deduction
            
        score = max(0.0, min(10.0, score))
        
        # Categorize label
        if score <= 2.0:
            label = "Dangerous"
        elif score <= 5.0:
            label = "Poor"
        elif score <= 8.0:
            label = "Average"
        else:
            label = "Excellent"
            
        return round(score, 1), label

    def train_model(self, dataset_features, dataset_targets):
        """
        Trains the Linear Regression head using annotated road scenes features.
        
        Args:
            dataset_features (list or np.ndarray): N x 5 array of features
            dataset_targets (list or np.ndarray): N array of road quality scores (0.0 to 10.0)
        """
        X = np.array(dataset_features)
        y = np.array(dataset_targets)
        
        print(f"Training Road Quality regression head on {len(X)} samples...")
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        # Save model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Successfully trained and saved model weights to {self.model_path}")
