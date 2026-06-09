import os
import sys
import argparse
import cv2
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from segmentation.inference import RoadDamageSegmenter
from depth.model import DepthAnythingV2Wrapper
from depth.estimator import RoadHazardEstimator
from risk_engine.risk import calculate_risk_score
from risk_engine.quality_score import RoadQualityRegressor
from alert_system.alerts import HazardAlertSystem

class RoadHazardIntelligenceSystem:
    def __init__(self, device=None):
        print("Initializing Road Hazard Intelligence System...")
        self.device = device
        
        # Load all components
        self.segmenter = RoadDamageSegmenter(device=self.device)
        self.depth_model = DepthAnythingV2Wrapper(device=self.device)
        self.estimator = RoadHazardEstimator()
        self.quality_regressor = RoadQualityRegressor()
        self.alert_system = HazardAlertSystem()
        print("System ready!")

    def process_frame(self, frame, bike_speed):
        """
        Runs the full pipeline on a single frame.
        
        Args:
            frame (np.ndarray): Input BGR camera image.
            bike_speed (float): Bicycle speed in km/h.
        Returns:
            hud_frame (np.ndarray): Frame overlayed with HUD.
            outputs (dict): Diagnostic values calculated during pipeline.
        """
        # Step 1: Road Damage Segmentation (SegFormer)
        seg_mask, hazards = self.segmenter.segment(frame)
        
        # Step 2: Depth Map Estimation (Depth Anything V2)
        raw_depth = self.depth_model.estimate_depth(frame)
        
        # Step 3: Physical Geometry Calibration
        # We pass the segmentation mask so we can ignore hazard pixels when calibrating the road plane
        calibrated_depth = self.estimator.calibrate_depth_map(raw_depth, seg_mask)
        
        # Step 4: Estimate physical dimensions and calculate collision risk
        for hazard in hazards:
            # Physical metrics
            metrics = self.estimator.estimate_hazard_metrics(hazard, calibrated_depth)
            hazard.update(metrics)
            
            # Risk estimation
            risk_score, risk_level = calculate_risk_score(
                area_m2=hazard['area_m2'],
                depth_cm=hazard['depth_cm'],
                distance_m=hazard['distance_m'],
                bike_speed_kmh=bike_speed,
                hazard_class_id=hazard['class_id']
            )
            hazard['risk_score'] = risk_score
            hazard['risk_level'] = risk_level
            
        # Step 5: Road Quality Rating
        road_score, road_label = self.quality_regressor.calculate_score(hazards)
        
        # Step 6: Generate HUD Overlay & Voice Alerts
        hud_frame = self.alert_system.draw_hud(
            frame=frame,
            seg_mask=seg_mask,
            depth_map=calibrated_depth,
            hazards=hazards,
            road_score=road_score,
            road_label=road_label,
            bike_speed=bike_speed
        )
        
        # Fetch deep learning multi-task predictions if available
        dl_road_score = getattr(self.segmenter, 'latest_road_score', None)
        dl_severity = getattr(self.segmenter, 'latest_severity', None)
        dl_risk_score = getattr(self.segmenter, 'latest_risk_score', None)
        
        diagnostics = {
            'hazards': [
                {
                    'label': h['label'],
                    'bbox': h['bbox'],
                    'distance_m': h['distance_m'],
                    'depth_cm': h['depth_cm'],
                    'area_m2': h['area_m2'],
                    'width_m': h.get('width_m', 0.0),
                    'length_m': h.get('length_m', 0.0),
                    'volume_m3': h.get('volume_m3', 0.0),
                    'risk_score': h['risk_score'],
                    'risk_level': h['risk_level']
                } for h in hazards
            ],
            'road_score': road_score,
            'road_label': road_label,
            'dl_road_score': dl_road_score,
            'dl_severity': dl_severity,
            'dl_risk_score': dl_risk_score
        }
        
        return hud_frame, diagnostics


def main():
    parser = argparse.ArgumentParser(description="Road Hazard Intelligence System - E2E Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input image, video file, or camera index (e.g. 0)")
    parser.add_argument("--speed", type=float, default=config.DEFAULT_BIKE_SPEED_KMH, help="Speed of the bicycle in km/h")
    parser.add_argument("--output", type=str, default=None, help="Path to save processed video or image")
    parser.add_argument("--no-show", action="store_true", help="Disable real-time cv2.imshow visualization")
    args = parser.parse_args()

    # Initialize E2E system
    system = RoadHazardIntelligenceSystem()
    
    # Check if input is a camera index
    is_camera = False
    if args.input.isdigit():
        input_source = int(args.input)
        is_camera = True
    else:
        input_source = args.input
        
    # Open source
    is_video = True
    if not is_camera and os.path.exists(input_source):
        # Check file extension
        ext = os.path.splitext(input_source)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            is_video = False
            
    if not is_video:
        # Process single image
        print(f"Reading image: {input_source}")
        frame = cv2.imread(input_source)
        if frame is None:
            print("Error: Could not load image file.")
            sys.exit(1)
            
        t0 = time.time()
        hud_frame, diagnostics = system.process_frame(frame, args.speed)
        print(f"Frame processed in {time.time() - t0:.3f}s")
        print(f"Road Quality Score (Physics): {diagnostics['road_score']}/10 ({diagnostics['road_label']})")
        if diagnostics['dl_road_score'] is not None:
            print(f"Road Quality Score (Deep Learning): {diagnostics['dl_road_score']:.2f}/10")
        print(f"Detected Hazards Count: {len(diagnostics['hazards'])}")
        for idx, h in enumerate(diagnostics['hazards']):
            dimensions_str = f"W: {h['width_m']}m, L: {h['length_m']}m, Vol: {h['volume_m3']}m3"
            print(f"  [{idx+1}] {h['label'].upper()} - Dist: {h['distance_m']}m, Depth: {h['depth_cm']}cm, Area: {h['area_m2']}m2 ({dimensions_str}), Risk: {h['risk_score']}% ({h['risk_level']})")
            
        if args.output:
            cv2.imwrite(args.output, hud_frame)
            print(f"Saved processed output to {args.output}")
            
        if not args.no_show:
            cv2.imshow("Road Hazard HUD", hud_frame)
            print("Press any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    else:
        # Process video stream / camera feed
        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            sys.exit(1)
            
        # Get video properties for writer
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if not is_camera else 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Override output resolution to standard if needed
        if width == 0 or height == 0:
            width, height = config.IMAGE_WIDTH, config.IMAGE_HEIGHT
            
        writer = None
        if args.output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
            print(f"Saving processed video to {args.output}")
            
        print("Starting video processing loop. Press 'q' to stop.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # If camera frame is empty or different, resize to standard
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
                
            hud_frame, diagnostics = system.process_frame(frame, args.speed)
            
            if writer:
                writer.write(hud_frame)
                
            if not args.no_show:
                cv2.imshow("Road Hazard HUD", hud_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Video processing finished.")

if __name__ == "__main__":
    main()
