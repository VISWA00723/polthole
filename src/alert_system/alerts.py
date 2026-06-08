import cv2
import numpy as np
import threading
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Attempt to import pyttsx3 for offline Text-to-Speech
try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

class HazardAlertSystem:
    def __init__(self):
        self.last_speech_time = 0
        self.speech_cooldown = 4.0  # seconds between spoken alerts
        self.tts_engine = None
        
        if HAS_TTS:
            try:
                # Initialize TTS on main thread first
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 160)  # Speaking speed
                self.tts_engine.setProperty('volume', 1.0)
            except Exception as e:
                print(f"Failed to initialize pyttsx3 engine: {e}. Voice alerts will be logged to console.")
                self.tts_engine = None

    def _speak_worker(self, text):
        """Worker thread function to speak without blocking video display."""
        try:
            # Re-initialize engine inside the thread to avoid COM model thread clashes on Windows
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            # Fallback console announcement if threading engine fails
            print(f"[VOICE ALERT]: {text}")

    def trigger_voice_alert(self, text):
        """Sparks a voice warning asynchronously if cooldown has expired."""
        now = time.time()
        if now - self.last_speech_time > self.speech_cooldown:
            self.last_speech_time = now
            if HAS_TTS and self.tts_engine is not None:
                # Start speech on a separate thread to prevent blocking camera frame processing
                t = threading.Thread(target=self._speak_worker, args=(text,), daemon=True)
                t.start()
            else:
                print(f"\a[VOICE ALERT]: {text}")  # Triggers a system terminal beep and print

    def draw_hud(self, frame, seg_mask, depth_map, hazards, road_score, road_label, bike_speed):
        """
        Draws the visual HUD overlay on the camera frame.
        
        Args:
            frame (np.ndarray): Original BGR camera frame.
            seg_mask (np.ndarray): Segmentation mask (0-4).
            depth_map (np.ndarray): Calibrated depth map (Z in meters).
            hazards (list): List of detected hazards with bounding boxes and physical metrics.
            road_score (float): Road quality score (0-10).
            road_label (str): Road score category.
            bike_speed (float): Speed of the bike in km/h.
        Returns:
            hud_frame (np.ndarray): Combined visual output.
        """
        h, w = frame.shape[:2]
        hud_frame = frame.copy()
        
        # 1. Overlay Segmentation Mask (semi-transparent)
        color_mask = np.zeros_like(frame)
        for class_id, color in config.CLASS_COLORS.items():
            if class_id == 0:
                continue
            color_mask[seg_mask == class_id] = color
            
        hud_frame = cv2.addWeighted(hud_frame, 0.8, color_mask, 0.4, 0)
        
        # 2. Draw depth map inset or side-by-side
        # Normalize depth map values so we map near (small Z) to Red and far (large Z) to Blue.
        # Depth Anything V2 outputs Z distance. We want closer distance (e.g. < 5m) to be hot (Red),
        # and far distance (e.g. > 15m) to be cold (Blue).
        # We invert Z to get inverse distance: 1 / Z
        inv_z = 1.0 / (depth_map + 1e-5)
        # Normalize inv_z to [0, 255]
        min_val = inv_z.min()
        max_val = inv_z.max()
        norm_depth = (inv_z - min_val) / (max_val - min_val + 1e-5)
        norm_depth_u8 = (norm_depth * 255).astype(np.uint8)
        
        # Color colormap (JET color map: 255 is red, 0 is blue)
        depth_color = cv2.applyColorMap(norm_depth_u8, cv2.COLORMAP_JET)
        
        # Create a small inset for depth display (1/4 size in bottom-right corner)
        inset_h, inset_w = int(h * 0.28), int(w * 0.28)
        depth_inset = cv2.resize(depth_color, (inset_w, inset_h))
        
        # Draw border around inset
        cv2.rectangle(depth_inset, (0, 0), (inset_w-1, inset_h-1), (255, 255, 255), 2)
        cv2.putText(depth_inset, "DEPTH MAP (Near=Red)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Place inset on HUD frame
        hud_frame[h - inset_h - 10 : h - 10, w - inset_w - 10 : w - 10] = depth_inset
        
        # 3. Draw Bounding Boxes and Metrics for hazards
        highest_risk = 0.0
        critical_hazard_msg = None
        
        for idx, hazard in enumerate(hazards):
            xmin, ymin, xmax, ymax = hazard['bbox']
            label = hazard['label']
            class_id = hazard['class_id']
            color = config.CLASS_COLORS.get(class_id, (255, 255, 255))
            
            # Fetch physical properties calculated from the estimator
            dist = hazard.get('distance_m', 0.0)
            depth = hazard.get('depth_cm', 0.0)
            area = hazard.get('area_m2', 0.0)
            risk = hazard.get('risk_score', 0.0)
            risk_level = hazard.get('risk_level', 'Low')
            
            highest_risk = max(highest_risk, risk)
            
            # Choose border thickness based on risk level
            thickness = 2
            if risk_level == 'Critical':
                thickness = 4
                critical_hazard_msg = f"Critical {label} {int(dist)} meters ahead!"
            elif risk_level == 'High':
                thickness = 3
                if critical_hazard_msg is None:
                    critical_hazard_msg = f"Watch out! {label} ahead."
                    
            # Draw box
            cv2.rectangle(hud_frame, (xmin, ymin), (xmax, ymax), color, thickness)
            
            # Draw label banner
            text_lines = [
                f"{label.upper()} [Risk: {int(risk)}% ({risk_level})]",
                f"Dist: {dist}m | Area: {area}m2"
            ]
            if class_id in [1, 3]:  # Pothole
                text_lines.append(f"Depth: {depth}cm")
                
            y_offset = ymin - len(text_lines) * 15 - 5
            y_offset = max(15, y_offset)
            
            for line in text_lines:
                cv2.putText(hud_frame, line, (xmin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                y_offset += 15

        # 4. Trigger voice alert if critical/high hazard is detected
        if critical_hazard_msg:
            self.trigger_voice_alert(critical_hazard_msg)
            
        # 5. Draw Telemetry Dashboard
        # Dashboard Background
        dash_w, dash_h = 320, 160
        overlay = hud_frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + dash_w, 10 + dash_h), (20, 20, 20), -1)
        # Blend overlay (opacity 70%)
        hud_frame = cv2.addWeighted(hud_frame, 0.3, overlay, 0.7, 0)
        
        # Telemetry Texts
        cv2.putText(hud_frame, "ROAD HAZARD INTELLIGENCE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(hud_frame, (20, 38), (10 + dash_w - 10, 38), (100, 100, 100), 1)
        
        # Speed display
        cv2.putText(hud_frame, f"Speed: {bike_speed:.1f} km/h", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Road condition score
        score_color = (0, 255, 0)  # Green
        if road_score <= 2.0:
            score_color = (0, 0, 255)  # Red
        elif road_score <= 5.0:
            score_color = (0, 165, 255)  # Orange
        elif road_score <= 8.0:
            score_color = (0, 255, 255)  # Yellow
            
        cv2.putText(hud_frame, f"Road Quality: {road_score}/10", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(hud_frame, road_label.upper(), (200, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_color, 2, cv2.LINE_AA)
        
        # Highest hazard risk status
        cv2.putText(hud_frame, f"Max Collision Risk: {int(highest_risk)}%", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Visual Alert Indicator
        alert_text = "SYSTEM OK"
        alert_color = (0, 255, 0)
        if highest_risk >= 75.0:
            alert_text = "CRITICAL WARNING"
            alert_color = (0, 0, 255)
        elif highest_risk >= 45.0:
            alert_text = "WARNING AHEAD"
            alert_color = (0, 165, 255)
            
        cv2.rectangle(hud_frame, (20, 125), (10 + dash_w - 20, 155), (30, 30, 30), -1)
        cv2.rectangle(hud_frame, (20, 125), (10 + dash_w - 20, 155), alert_color, 1)
        
        # Center the text in alert banner
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = 20 + ((dash_w - 30) - text_size[0]) // 2
        cv2.putText(hud_frame, alert_text, (text_x, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, alert_color, 2, cv2.LINE_AA)
        
        return hud_frame
