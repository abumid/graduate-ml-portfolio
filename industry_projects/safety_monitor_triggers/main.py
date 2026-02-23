"""
Safety Monitoring System - Ensemble Detection Version
Uses multiple YOLO models with Ultralytics for enhanced accuracy
- Custom YOLO model: phone, smoking detection
- YOLOv5s model: general object detection (phone ensemble)
Detects: Face presence, Sleeping (closed eyes), Yawning (fatigue), Phone usage, Smoking
"""

import cv2
import numpy as np
import time
from collections import deque
from pathlib import Path
from datetime import datetime, timedelta

from cvzone.FaceMeshModule import FaceMeshDetector
from ultralytics import YOLO


# Configuration
CONFIG = {
    'camera_id': 0,
    'frame_width': 1920,   # Full HD width for bigger display
    'frame_height': 1080,  # Full HD height (16:9 aspect ratio)
    'yolo_custom_model': 'weights/phone_smoking_foodbest.pt',  # Custom trained model
    'yolo_general_model': 'weights/yolov5s.pt',  # General YOLOv5s model
    'use_ensemble': True,  # Use both models for better detection
    'yolo_conf': 0.1,
    'yolo_iou': 0.25,
    'eye_closed_threshold': 35,  # Lower = more sensitive (typical: 20-30)
    'yawn_threshold': 65,
    'buffer_size': 5,  # Smoothing buffer
    'logo_path': 'assests/logo.png',  # Logo image path
    'right_panel_width': 350,  # Width of right info panel
    
    # Per-class warning thresholds (consecutive frames before warning)
    'thresholds': {
        'eyes_closed': 1,   # Sleeping - quick alert
        'yawning': 5,       # Fatigue - slightly slower
        'phone': 3,         # Phone usage - quick alert
        'smoking': 2,       # Smoking - very quick alert
    },
    
    # Per-class confidence thresholds for YOLO detection
    'class_confidence': {
        'phone': 0.3,       # Phone detection confidence
        'smoking': 0.1,    # Smoking detection confidence (higher = more strict)
    },
}

# Colors (BGR format)
COLORS = {
    'OK': (0, 255, 0),
    'WARNING': (0, 165, 255),
    'DANGER': (0, 0, 255),
    'TEXT': (255, 255, 255),
    'BG': (0, 0, 0),
    'PANEL_BG': (40, 40, 40),
    'PANEL_BORDER': (80, 80, 80),
    'HEADER_BG': (20, 20, 20),
    'TIMER_ACTIVE': (0, 140, 255),
    'TIMER_TOTAL': (180, 180, 180),
}


class OptimizedSafetyMonitor:
    """Ensemble safety monitoring system using multiple YOLO models with Ultralytics"""
    
    def __init__(self):
        # Initialize camera with optimized settings
        self.cap = cv2.VideoCapture(CONFIG['camera_id'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['frame_width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['frame_height'])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        # Face detector
        self.face_detector = FaceMeshDetector(maxFaces=1)
        
        # Load custom YOLO model for phone, smoking, and food detection
        custom_model_path = Path(CONFIG['yolo_custom_model'])
        if not custom_model_path.exists():
            raise FileNotFoundError(f"Custom model not found: {custom_model_path}")
        
        print(f"Loading custom model: {custom_model_path}")
        self.yolo_custom = YOLO(str(custom_model_path))
        
        # Load general YOLOv5s model for ensemble detection
        self.yolo_general = None
        if CONFIG['use_ensemble']:
            general_model_path = Path(CONFIG['yolo_general_model'])
            if general_model_path.exists():
                print(f"Loading ensemble model: {general_model_path}")
                self.yolo_general = YOLO(str(general_model_path))
            else:
                print(f"Warning: Ensemble model not found: {general_model_path}")
                print("Continuing with single model only")
        
        # Force CPU usage to avoid cuDNN errors
        try:
            import torch
            if torch.cuda.is_available():
                print("CUDA available but using CPU for stability")
            self.yolo_custom.to('cpu')
            if self.yolo_general:
                self.yolo_general.to('cpu')
        except Exception as e:
            print(f"Device setup: {e}")
        
        # Set confidence and IOU thresholds
        self.yolo_custom.conf = CONFIG['yolo_conf']
        self.yolo_custom.iou = CONFIG['yolo_iou']
        if self.yolo_general:
            self.yolo_general.conf = CONFIG['yolo_conf']
            self.yolo_general.iou = CONFIG['yolo_iou']
        
        # Buffers for smoothing
        self.eye_buffer = deque(maxlen=CONFIG['buffer_size'])
        self.mouth_buffer = deque(maxlen=CONFIG['buffer_size'])
        
        # Warning counters
        self.counters = {
            'eyes_closed': 0,
            'yawning': 0,
            'phone': 0,
            'smoking': 0,
        }
        
        # Timers for tracking duration of each action
        self.timers = {
            'eyes_closed': None,
            'yawning': None,
            'phone': None,
            'smoking': None,
        }
        
        # Total duration tracking (in seconds)
        self.total_duration = {
            'eyes_closed': 0.0,
            'yawning': 0.0,
            'phone': 0.0,
            'smoking': 0.0,
        }
        
        # FPS counter
        self.fps = 0
        self.fps_time = time.time()
        
        # Load logo
        self.logo = None
        try:
            logo_path = Path(CONFIG['logo_path'])
            if logo_path.exists():
                self.logo = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
                # Resize logo to fit in header (max 60px height)
                if self.logo is not None:
                    h_logo = self.logo.shape[0]
                    w_logo = self.logo.shape[1]
                    target_height = 50
                    scale = target_height / h_logo
                    new_width = int(w_logo * scale)
                    self.logo = cv2.resize(self.logo, (new_width, target_height))
                    print(f"Logo loaded: {logo_path}")
        except Exception as e:
            print(f"Could not load logo: {e}")
        
        # Start time for session duration
        self.session_start = time.time()
    
    def calculate_ratio(self, p1, p2, p3, p4):
        """Calculate aspect ratio from 4 points (top, bottom, left, right)"""
        vertical = np.linalg.norm(np.array(p1) - np.array(p2))
        horizontal = np.linalg.norm(np.array(p3) - np.array(p4))
        return (vertical / horizontal * 100) if horizontal > 0 else 0
    
    def detect_yolo_objects(self, frame):
        """Detect phone and smoking using ensemble YOLO models"""
        phone = False
        smoking = False
        boxes = []
        
        # Run inference with custom model
        results_custom = self.yolo_custom(frame, verbose=False, conf=CONFIG['yolo_conf'], 
                                          iou=CONFIG['yolo_iou'], device='cpu')
        
        # Parse custom model results
        if len(results_custom) > 0:
            result = results_custom[0]
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Custom model classes: 0=phone, 1=smoking
                    # Apply per-class confidence thresholds
                    if cls_id == 0 and conf >= CONFIG['class_confidence']['phone']:  # Phone
                        phone = True
                        boxes.append(('phone', xyxy.tolist(), conf, 'custom'))
                    elif cls_id == 1 and conf >= CONFIG['class_confidence']['smoking']:  # Smoking
                        smoking = True
                        boxes.append(('smoking', xyxy.tolist(), conf, 'custom'))
        
        # Run inference with general YOLOv5s model for ensemble
        if self.yolo_general and CONFIG['use_ensemble']:
            results_general = self.yolo_general(frame, verbose=False, conf=CONFIG['yolo_conf'], 
                                                iou=CONFIG['yolo_iou'], device='cpu')
            
            # Parse general model results
            if len(results_general) > 0:
                result = results_general[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # YOLOv5s COCO classes: 67=cell phone, 0=person (check for phone near person)
                        # Apply phone confidence threshold
                        if cls_id == 67 and conf >= CONFIG['class_confidence']['phone']:  # Cell phone in COCO dataset
                            phone = True
                            boxes.append(('phone', xyxy.tolist(), conf, 'yolov5s'))
                        # Note: YOLOv5s doesn't have smoking class in COCO
        
        return phone, smoking, boxes
    
    def process(self):
        """Process one frame"""
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None
        
        frame = cv2.resize(frame, (CONFIG['frame_width'], CONFIG['frame_height']))
        
        # Initialize status
        status = {
            'face': False,
            'eyes_closed': False,
            'yawning': False,
            'phone': False,
            'smoking': False,
        }
        
        # Store current ratios for display
        debug_info = {'eye_ratio': 0, 'mouth_ratio': 0}
        
        # Detect face
        _, faces = self.face_detector.findFaceMesh(frame, draw=False)
        status['face'] = len(faces) > 0
        
        if status['face']:
            face = faces[0]
            
            # Eye detection (landmarks: 159-top, 23-bottom, 130-left, 243-right)
            eye_ratio = self.calculate_ratio(face[159], face[23], face[130], face[243])
            self.eye_buffer.append(eye_ratio)
            avg_eye = np.mean(self.eye_buffer)
            status['eyes_closed'] = avg_eye < CONFIG['eye_closed_threshold']
            debug_info['eye_ratio'] = avg_eye
            
            # Mouth detection (landmarks: 0-top, 17-bottom, 61-left, 291-right)
            mouth_ratio = self.calculate_ratio(face[0], face[17], face[61], face[291])
            self.mouth_buffer.append(mouth_ratio)
            avg_mouth = np.mean(self.mouth_buffer)
            status['yawning'] = avg_mouth > CONFIG['yawn_threshold']
            debug_info['mouth_ratio'] = avg_mouth
        
        # Detect phone and smoking
        phone, smoking, boxes = self.detect_yolo_objects(frame)
        status['phone'] = phone
        status['smoking'] = smoking
        
        # Update counters with per-class thresholds
        for key in ['eyes_closed', 'yawning', 'phone', 'smoking']:
            if status[key]:
                max_threshold = CONFIG['thresholds'][key] + 10
                self.counters[key] = min(self.counters[key] + 1, max_threshold)
                
                # Start timer if not already started
                if self.timers[key] is None:
                    self.timers[key] = time.time()
                    
            else:
                # If action stopped, add elapsed time to total duration
                if self.timers[key] is not None:
                    elapsed = time.time() - self.timers[key]
                    self.total_duration[key] += elapsed
                    self.timers[key] = None
                    
                self.counters[key] = max(self.counters[key] - 1, 0)
        
        # Calculate FPS
        current = time.time()
        self.fps = 1.0 / (current - self.fps_time)
        self.fps_time = current
        
        return frame, status, boxes, debug_info
    
    def format_duration(self, seconds):
        """Format duration in seconds to MM:SS or HH:MM:SS"""
        if seconds < 1:
            return "0s"
        elif seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {minutes}m {secs}s"
    
    def get_current_duration(self, key):
        """Get current duration including active timer"""
        total = self.total_duration[key]
        if self.timers[key] is not None:
            total += time.time() - self.timers[key]
        return total
    
    def overlay_logo(self, frame):
        """Overlay logo with transparency support"""
        if self.logo is None:
            return frame
        
        try:
            h, w = frame.shape[:2]
            h_logo, w_logo = self.logo.shape[:2]
            panel_w = CONFIG['right_panel_width']
            
            # Position logo in top-right corner of the panel
            x_offset = w - panel_w + 15
            y_offset = 15
            
            # Handle PNG with alpha channel
            if self.logo.shape[2] == 4:  # RGBA
                alpha_logo = self.logo[:, :, 3] / 255.0
                alpha_frame = 1.0 - alpha_logo
                
                for c in range(3):
                    frame[y_offset:y_offset+h_logo, x_offset:x_offset+w_logo, c] = \
                        (alpha_logo * self.logo[:, :, c] + 
                         alpha_frame * frame[y_offset:y_offset+h_logo, x_offset:x_offset+w_logo, c])
            else:  # No alpha channel
                frame[y_offset:y_offset+h_logo, x_offset:x_offset+w_logo] = self.logo
        except Exception as e:
            pass  # Silently fail if logo overlay doesn't work
        
        return frame
    
    def draw_right_panel(self, frame):
        """Draw right information panel"""
        h, w = frame.shape[:2]
        panel_w = CONFIG['right_panel_width']
        panel_x = w - panel_w
        
        # Draw panel background with border
        cv2.rectangle(frame, (panel_x, 0), (w, h), COLORS['PANEL_BG'], -1)
        cv2.line(frame, (panel_x, 0), (panel_x, h), COLORS['PANEL_BORDER'], 2)
        
        # Overlay logo at the top
        frame = self.overlay_logo(frame)
        
        y_pos = 90
        line_height = 28
        text_x = panel_x + 15
        
        # Session info
        cv2.putText(frame, "SESSION INFO", (text_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['TEXT'], 2)
        y_pos += line_height + 5
        
        session_duration = time.time() - self.session_start
        cv2.putText(frame, f"Duration: {self.format_duration(session_duration)}", 
                   (text_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['TIMER_TOTAL'], 1)
        y_pos += line_height
        
        cv2.putText(frame, f"FPS: {self.fps:.1f}", 
                   (text_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['TIMER_TOTAL'], 1)
        y_pos += line_height + 10
        
        # Separator line
        cv2.line(frame, (text_x, y_pos), (w - 15, y_pos), COLORS['PANEL_BORDER'], 1)
        y_pos += 20
        
        # Violation Statistics
        cv2.putText(frame, "VIOLATIONS", (text_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['TEXT'], 2)
        y_pos += line_height + 5
        
        # Define violation display info
        violations = [
            ('eyes_closed', 'Sleeping', COLORS['DANGER']),
            ('yawning', 'Fatigue', COLORS['WARNING']),
            ('phone', 'Phone Use', COLORS['DANGER']),
            ('smoking', 'Smoking', (0, 100, 255)),
        ]
        
        for key, label, color in violations:
            is_active = self.counters[key] >= CONFIG['thresholds'][key]
            current_duration = self.get_current_duration(key)
            
            # Draw label with status indicator
            status_color = color if is_active else COLORS['OK']
            indicator = "[!]" if is_active else "[ ]"
            
            cv2.putText(frame, f"{indicator} {label}", (text_x, y_pos),
                       cv2.FONT_HERSHEY_COMPLEX, 0.7, status_color, 1)
            y_pos += line_height - 3
            
            # Show current active time if happening now
            if self.timers[key] is not None:
                active_time = time.time() - self.timers[key]
                cv2.putText(frame, f"   Now: {self.format_duration(active_time)}", 
                           (text_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['TIMER_ACTIVE'], 1)
                y_pos += line_height - 5
            
            # Show total accumulated time
            if current_duration > 0:
                cv2.putText(frame, f"   Total: {self.format_duration(current_duration)}", 
                           (text_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['TIMER_TOTAL'], 1)
                y_pos += line_height - 3
            
            y_pos += 5  # Extra spacing between violations
        
        # Separator line
        y_pos += 10
        cv2.line(frame, (text_x, y_pos), (w - 15, y_pos), COLORS['PANEL_BORDER'], 1)
        y_pos += 25
        
        # Model info
        cv2.putText(frame, "DETECTION", (text_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['TEXT'], 2)
        y_pos += line_height + 5
        
        mode_text = "Ensemble" if CONFIG['use_ensemble'] and self.yolo_general else "Single"
        cv2.putText(frame, f"Mode: {mode_text}", 
                   (text_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['TIMER_TOTAL'], 1)
        y_pos += line_height - 5
        
        cv2.putText(frame, f"Conf: {CONFIG['yolo_conf']:.2f}", 
                   (text_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['TIMER_TOTAL'], 1)
        y_pos += line_height - 5
        
        cv2.putText(frame, f"IOU: {CONFIG['yolo_iou']:.2f}", 
                   (text_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['TIMER_TOTAL'], 1)
        
        return frame
    
    def draw_ui(self, frame, status, boxes, debug_info):
        """Draw optimized user interface with right panel"""
        h, w = frame.shape[:2]
        panel_w = CONFIG['right_panel_width']
        
        # Draw right info panel first
        frame = self.draw_right_panel(frame)
        
        # Header area (top left of screen, before right panel)
        header_h = 70
        cv2.rectangle(frame, (0, 0), (w - panel_w, header_h), COLORS['HEADER_BG'], -1)
        cv2.line(frame, (0, header_h), (w - panel_w, header_h), COLORS['PANEL_BORDER'], 2)
        
        title = "SAFETY MONITOR - ENSEMBLE MODE" if CONFIG['use_ensemble'] else "SAFETY MONITOR"
        cv2.putText(frame, title, (20, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS['TEXT'], 2)
        
        # Current status area (below header, to the left of panel)
        status_y = header_h + 30
        status_x = 20
        line_h = 40
        
        # Current status area (below header, to the right of panel)
        status_y = header_h + 30
        status_x = panel_w + 20
        line_h = 40
        
        # def draw_compact_status(text, is_ok, y):
        #     """Draw compact status line with icon"""
        #     color = COLORS['OK'] if is_ok else COLORS['DANGER']
        #     icon = "[OK]" if is_ok else "[!!]"
        #     cv2.putText(frame, f"{icon} {text}", (status_x, y),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Face detection status
        # draw_compact_status("Face Detected" if status['face'] else "No Face", 
        #                   status['face'], status_y)
        
        # if status['face']:
        #     # Eyes
        #     eyes_ok = self.counters['eyes_closed'] < CONFIG['thresholds']['eyes_closed']
        #     ear_text = f"Eyes (EAR: {debug_info['eye_ratio']:.1f})"
        #     draw_compact_status(ear_text, eyes_ok, status_y + line_h)
            
        #     # Mouth
        #     yawn_ok = self.counters['yawning'] < CONFIG['thresholds']['yawning']
        #     mar_text = f"Mouth (MAR: {debug_info['mouth_ratio']:.1f})"
        #     draw_compact_status(mar_text, yawn_ok, status_y + line_h * 2)
        
        # Draw bounding boxes - increased size for larger display
        for detection in boxes:
            obj_type, box, conf = detection[0], detection[1], detection[2]
            model_source = detection[3] if len(detection) > 3 else 'unknown'
            
            # Use different colors for different models
            if model_source == 'yolov5s':
                color = (255, 165, 0)  # Orange for YOLOv5s
            else:
                color = COLORS['DANGER']  # Red for custom model
            
            label = f"{obj_type.upper()}: {conf:.2f}"
            if CONFIG['use_ensemble']:
                label = f"{obj_type.upper()}: {conf:.2f} [{model_source}]"
            
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 3)
            
            # Draw label background for better readability
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (box[0], box[1] - label_size[1] - 10), 
                         (box[0] + label_size[0] + 5, box[1]), color, -1)
            cv2.putText(frame, label, (box[0] + 3, box[1] - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['TEXT'], 2)
        
        # Warning overlay if any danger - optimized placement
        warnings = []
        if self.counters['eyes_closed'] >= CONFIG['thresholds']['eyes_closed']:
            warnings.append("WARNING: SLEEPING DETECTED!")
        if self.counters['yawning'] >= CONFIG['thresholds']['yawning']:
            warnings.append("WARNING: FATIGUE DETECTED!")
        if self.counters['phone'] >= CONFIG['thresholds']['phone']:
            warnings.append("WARNING: PHONE IN USE!")
        if self.counters['smoking'] >= CONFIG['thresholds']['smoking']:
            warnings.append("WARNING: SMOKING!")
        
        if warnings:
            # Flashing warning banner at bottom
            if int(time.time() * 2) % 2 == 0:
                warning_height = 120
                # Semi-transparent red overlay (only on main area, not panel)
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h - warning_height), (w - panel_w, h), (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                # Draw warnings
                y_pos = h - warning_height + 35
                for warning in warnings:
                    text_size = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
                    x_center = (w - panel_w) // 2 - text_size[0] // 2
                    cv2.putText(frame, warning, (x_center, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS['TEXT'], 3)
                    y_pos += 40
        
        return frame
    
    def run(self):
        """Main loop"""
        print("=" * 70)
        print("SAFETY MONITORING SYSTEM - ENSEMBLE DETECTION")
        print("=" * 70)
        print("Models:")
        print(f"  • Custom YOLO: {CONFIG['yolo_custom_model']}")
        if CONFIG['use_ensemble'] and self.yolo_general:
            print(f"  • Ensemble YOLO: {CONFIG['yolo_general_model']}")
            print("  • Mode: Ensemble (Both models for better accuracy)")
        else:
            print("  • Mode: Single model")
        print(f"\nDisplay Resolution: {CONFIG['frame_width']}x{CONFIG['frame_height']}")
        print(f"Right Panel Width: {CONFIG['right_panel_width']}px")
        print("\nMonitoring:")
        print("  ✓ Face presence")
        print("  ✓ Sleeping (closed eyes) - with active & total duration")
        print("  ✓ Yawning (fatigue) - with active & total duration")
        print("  ✓ Phone usage (ensemble detection) - with active & total duration")
        print("  ✓ Smoking (custom model) - with active & total duration")
        print("\nFeatures:")
        print("  • Left info panel with session stats")
        print("  • Real-time violation timers")
        print("  • Total duration tracking")
        print("  • Logo display")
        print("\nPress 'q' to quit")
        print("=" * 70)
        
        # Create window and set it to a larger size
        window_title = "Safety Monitor - Ensemble Detection" if CONFIG['use_ensemble'] else "Safety Monitor"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, CONFIG['frame_width'], CONFIG['frame_height'])
        
        try:
            while True:
                result = self.process()
                if result[0] is None:
                    break
                
                frame, status, boxes, debug_info = result
                frame = self.draw_ui(frame, status, boxes, debug_info)
                
                cv2.imshow(window_title, frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nStopped by user")
        
        finally:
            # Print summary of total durations
            print("\n" + "=" * 70)
            print("SESSION SUMMARY - Total Duration for Each Action:")
            print("=" * 70)
            for key in ['eyes_closed', 'yawning', 'phone', 'smoking']:
                duration = self.get_current_duration(key)
                if duration > 0:
                    action_name = key.replace('_', ' ').title()
                    print(f"  • {action_name}: {self.format_duration(duration)}")
            print("=" * 70)
            
            self.cap.release()
            cv2.destroyAllWindows()
            print("System stopped.")


if __name__ == "__main__":
    monitor = OptimizedSafetyMonitor()
    monitor.run()
