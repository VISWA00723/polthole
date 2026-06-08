import os
import sys
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

# Add parent directory and src directory to python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'src'))

import config
from utils.geometry import get_ground_coordinates, get_ground_distance_map
from risk_engine.risk import calculate_risk_score
from risk_engine.quality_score import RoadQualityRegressor

class TestRoadHazardMath(unittest.TestCase):
    
    def test_geometry_projection(self):
        """Tests that pixel coordinate projection produces physically logical distances."""
        # A pixel at the bottom center of the image should be very close to the bike
        x_close = config.PRINCIPAL_POINT_X
        y_close = config.IMAGE_HEIGHT - 10
        _, Z_close = get_ground_coordinates(x_close, y_close)
        
        # A pixel near the horizon (principal point y) should be far away
        x_far = config.PRINCIPAL_POINT_X
        y_far = config.PRINCIPAL_POINT_Y + 10
        _, Z_far = get_ground_coordinates(x_far, y_far)
        
        self.assertGreater(Z_far, Z_close)
        self.assertGreater(Z_close, 0.0)
        
        # Distance map shape test
        dist_map = get_ground_distance_map(160, 90)
        self.assertEqual(dist_map.shape, (90, 160))

    def test_risk_score_bounds(self):
        """Verifies physics-based collision risk engine edge cases and bounds."""
        # 1. Stationary bike should have 0 risk
        risk, level = calculate_risk_score(area_m2=0.5, depth_cm=10.0, distance_m=5.0, bike_speed_kmh=0.0, hazard_class_id=1)
        self.assertEqual(risk, 0.0)
        self.assertEqual(level, 'Low')
        
        # 2. Risk should increase with higher speed
        risk_slow, _ = calculate_risk_score(area_m2=0.5, depth_cm=10.0, distance_m=5.0, bike_speed_kmh=10.0, hazard_class_id=1)
        risk_fast, _ = calculate_risk_score(area_m2=0.5, depth_cm=10.0, distance_m=5.0, bike_speed_kmh=30.0, hazard_class_id=1)
        self.assertGreater(risk_fast, risk_slow)
        
        # 3. Risk should decrease with larger distance
        risk_near, _ = calculate_risk_score(area_m2=0.5, depth_cm=10.0, distance_m=3.0, bike_speed_kmh=20.0, hazard_class_id=1)
        risk_distant, _ = calculate_risk_score(area_m2=0.5, depth_cm=10.0, distance_m=15.0, bike_speed_kmh=20.0, hazard_class_id=1)
        self.assertGreater(risk_near, risk_distant)
        
        # 4. Range clamping
        risk_extreme, level_extreme = calculate_risk_score(area_m2=5.0, depth_cm=50.0, distance_m=0.5, bike_speed_kmh=45.0, hazard_class_id=1)
        self.assertGreaterEqual(risk_extreme, 0.0)
        self.assertLessEqual(risk_extreme, 100.0)
        self.assertEqual(level_extreme, 'Critical')

    def test_road_quality_scoring(self):
        """Tests that road quality index correctly rates hazard profiles."""
        regressor = RoadQualityRegressor()
        
        # 1. No hazards should yield a perfect 10.0 score
        score_perfect, label_perfect = regressor.calculate_score([])
        self.assertEqual(score_perfect, 10.0)
        self.assertEqual(label_perfect, "Excellent")
        
        # 2. Severe pothole should reduce score heavily
        bad_hazards = [{
            'class_id': 1,
            'label': 'pothole',
            'area_ratio': 0.10,
            'depth_cm': 15.0
        }]
        score_bad, label_bad = regressor.calculate_score(bad_hazards)
        self.assertLess(score_bad, 10.0)
        self.assertIn(label_bad, ["Poor", "Dangerous", "Average"])
        
        # 3. Multiple hazards should degrade score further
        very_bad_hazards = [
            {'class_id': 1, 'label': 'pothole', 'area_ratio': 0.12, 'depth_cm': 20.0},
            {'class_id': 2, 'label': 'crack', 'area_ratio': 0.20, 'depth_cm': 0.0},
            {'class_id': 3, 'label': 'water_pothole', 'area_ratio': 0.08, 'depth_cm': 15.0}
        ]
        score_vbad, label_vbad = regressor.calculate_score(very_bad_hazards)
        self.assertLess(score_vbad, score_bad)
        self.assertEqual(score_vbad, 0.0)  # Should saturate at 0.0
        self.assertEqual(label_vbad, "Dangerous")


class TestPipelineMocked(unittest.TestCase):
    
    @patch('segmentation.inference.RoadDamageSegmenter')
    @patch('depth.model.DepthAnythingV2Wrapper')
    def test_e2e_pipeline_flow(self, MockDepthWrapper, MockSegmenter):
        """Validates that the pipeline runner links stages correctly using mocked models."""
        from pipeline import RoadHazardIntelligenceSystem
        
        # Setup mocks
        mock_seg = MockSegmenter.return_value
        mock_depth = MockDepthWrapper.return_value
        
        # Mock segmentation outputs
        h_mask = np.zeros((720, 1280), dtype=np.uint8)
        h_mask[500:550, 600:650] = 1  # Pothole box
        mock_seg.segment.return_value = (
            h_mask, 
            [{
                'class_id': 1,
                'label': 'pothole',
                'mask': h_mask,
                'area_ratio': 0.01,
                'bbox': (600, 500, 650, 550)
            }]
        )
        
        # Mock depth map output
        mock_depth.estimate_depth.return_value = np.ones((720, 1280), dtype=np.float32) * 500.0
        
        # Initialize pipeline and process a dummy frame
        system = RoadHazardIntelligenceSystem()
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        hud_frame, diagnostics = system.process_frame(dummy_frame, bike_speed=20.0)
        
        # Assertions
        self.assertEqual(hud_frame.shape, (720, 1280, 3))
        self.assertIn('road_score', diagnostics)
        self.assertIn('hazards', diagnostics)
        self.assertEqual(len(diagnostics['hazards']), 1)
        self.assertIn('risk_score', diagnostics['hazards'][0])

if __name__ == "__main__":
    unittest.main()
