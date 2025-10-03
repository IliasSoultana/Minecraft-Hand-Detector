"""
Visualization Module
Handles debug visualization and drawing functions
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
import time


class Visualizer:
    """Handles visualization and debug drawing for gesture recognition."""
    
    def __init__(self):
        """Initialize the visualizer."""
        self.logger = logging.getLogger(__name__)
        
        # Colors (BGR format for OpenCV)
        self.colors = {
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'gray': (128, 128, 128),
            'orange': (0, 165, 255),
            'purple': (128, 0, 128),
            'pink': (203, 192, 255)
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        
        # Drawing settings
        self.landmark_radius = 4
        self.connection_thickness = 2
        self.bbox_thickness = 2
        
        # Performance tracking
        self.fps_history = []
        self.frame_times = []
        
    def draw_debug_info(self, frame: np.ndarray, hand_results: Dict, 
                       gestures: List[Dict], yolo_results: Optional[Dict] = None) -> np.ndarray:
        """
        Draw comprehensive debug information on the frame.
        
        Args:
            frame: Input frame
            hand_results: Hand detection results
            gestures: Recognized gestures
            yolo_results: Optional YOLO detection results
            
        Returns:
            Frame with debug information drawn
        """
        debug_frame = frame.copy()
        
        # Draw hand bounding boxes and info
        if hand_results.get('hands'):
            debug_frame = self.draw_hand_info(debug_frame, hand_results['hands'])
        
        # Draw YOLO detections
        if yolo_results and yolo_results.get('detections'):
            debug_frame = self.draw_yolo_detections(debug_frame, yolo_results['detections'])
        
        # Draw gesture information
        debug_frame = self.draw_gesture_info(debug_frame, gestures)
        
        # Draw performance info
        debug_frame = self.draw_performance_info(debug_frame)
        
        # Draw status panel
        debug_frame = self.draw_status_panel(debug_frame, hand_results, gestures)
        
        return debug_frame
    
    def draw_landmarks(self, frame: np.ndarray, hand_results: Dict) -> np.ndarray:
        """
        Draw hand landmarks and connections.
        
        Args:
            frame: Input frame
            hand_results: Hand detection results
            
        Returns:
            Frame with landmarks drawn
        """
        if not hand_results.get('hands'):
            return frame
        
        landmark_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # MediaPipe hand connections
        hand_connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm
            (5, 9), (9, 13), (13, 17)
        ]
        
        for hand_idx, hand_data in enumerate(hand_results['hands']):
            landmarks = hand_data.get('landmarks', [])
            if len(landmarks) < 21:
                continue
            
            # Choose color based on hand
            hand_label = hand_data.get('label', 'Unknown')
            color = self.colors['green'] if hand_label == 'Right' else self.colors['blue']
            
            # Convert landmarks to pixel coordinates
            landmark_points = []
            for landmark in landmarks:
                x = int(landmark['x'] * width)
                y = int(landmark['y'] * height)
                landmark_points.append((x, y))
            
            # Draw connections
            for connection in hand_connections:
                start_idx, end_idx = connection
                if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                    start_point = landmark_points[start_idx]
                    end_point = landmark_points[end_idx]
                    cv2.line(landmark_frame, start_point, end_point, color, self.connection_thickness)
            
            # Draw landmark points
            for i, point in enumerate(landmark_points):
                # Different colors for fingertips
                if i in [4, 8, 12, 16, 20]:  # Fingertips
                    point_color = self.colors['red']
                    radius = self.landmark_radius + 1
                else:
                    point_color = color
                    radius = self.landmark_radius
                
                cv2.circle(landmark_frame, point, radius, point_color, -1)
                
                # Draw landmark index
                cv2.putText(landmark_frame, str(i), 
                           (point[0] + 5, point[1] - 5),
                           self.font, 0.3, self.colors['white'], 1)
        
        return landmark_frame
    
    def draw_hand_info(self, frame: np.ndarray, hands: List[Dict]) -> np.ndarray:
        """Draw hand detection information."""
        info_frame = frame.copy()
        
        for hand_data in hands:
            # Draw bounding box
            bbox = hand_data.get('bounding_box', {})
            if bbox:
                x1, y1 = bbox.get('x_min', 0), bbox.get('y_min', 0)
                x2, y2 = bbox.get('x_max', 0), bbox.get('y_max', 0)
                
                hand_label = hand_data.get('label', 'Unknown')
                color = self.colors['green'] if hand_label == 'Right' else self.colors['blue']
                
                cv2.rectangle(info_frame, (x1, y1), (x2, y2), color, self.bbox_thickness)
                
                # Draw hand info
                info_text = f"{hand_label} ({hand_data.get('score', 0):.2f})"
                cv2.putText(info_frame, info_text, (x1, y1 - 10),
                           self.font, self.font_scale, color, self.font_thickness)
            
            # Draw hand pose info
            pose = hand_data.get('pose', {})
            if pose and bbox:
                center = bbox.get('center', (0, 0))
                
                # Draw finger state
                fingers_up = pose.get('fingers_up', [])
                if fingers_up:
                    finger_text = ''.join(['1' if f else '0' for f in fingers_up])
                    cv2.putText(info_frame, f"Fingers: {finger_text}",
                               (center[0] - 50, center[1] + 30),
                               self.font, 0.5, self.colors['yellow'], 1)
                
                # Draw openness
                openness = pose.get('openness', 0)
                cv2.putText(info_frame, f"Open: {openness:.2f}",
                           (center[0] - 50, center[1] + 50),
                           self.font, 0.5, self.colors['cyan'], 1)
                
                # Draw orientation
                angle = pose.get('orientation_angle', 0)
                cv2.putText(info_frame, f"Angle: {angle:.1f}°",
                           (center[0] - 50, center[1] + 70),
                           self.font, 0.5, self.colors['magenta'], 1)
        
        return info_frame
    
    def draw_yolo_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw YOLO detection results."""
        yolo_frame = frame.copy()
        
        for detection in detections:
            bbox = detection.get('bbox', {})
            class_name = detection.get('class_name', 'Unknown')
            confidence = detection.get('confidence', 0)
            
            x1, y1 = bbox.get('x1', 0), bbox.get('y1', 0)
            x2, y2 = bbox.get('x2', 0), bbox.get('y2', 0)
            
            # Draw bounding box
            cv2.rectangle(yolo_frame, (x1, y1), (x2, y2), self.colors['orange'], 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, self.font, 0.5, 1)[0]
            
            # Draw label background
            cv2.rectangle(yolo_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), self.colors['orange'], -1)
            
            # Draw label text
            cv2.putText(yolo_frame, label, (x1, y1 - 5),
                       self.font, 0.5, self.colors['white'], 1)
        
        return yolo_frame
    
    def draw_gesture_info(self, frame: np.ndarray, gestures: List[Dict]) -> np.ndarray:
        """Draw recognized gesture information."""
        gesture_frame = frame.copy()
        
        if not gestures:
            return gesture_frame
        
        # Draw gesture panel
        panel_height = min(200, len(gestures) * 40 + 40)
        panel_width = 300
        
        # Create semi-transparent overlay
        overlay = gesture_frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), 
                     self.colors['black'], -1)
        gesture_frame = cv2.addWeighted(gesture_frame, 0.7, overlay, 0.3, 0)
        
        # Draw title
        cv2.putText(gesture_frame, "Recognized Gestures:", (20, 35),
                   self.font, 0.7, self.colors['white'], 2)
        
        # Draw gestures
        for i, gesture in enumerate(gestures):
            y_pos = 65 + i * 30
            gesture_name = gesture.get('display_name', gesture.get('name', 'Unknown'))
            confidence = gesture.get('confidence', 0)
            action = gesture.get('action', 'none')
            
            # Color based on confidence
            if confidence > 0.8:
                color = self.colors['green']
            elif confidence > 0.6:
                color = self.colors['yellow']
            else:
                color = self.colors['red']
            
            text = f"{gesture_name} -> {action} ({confidence:.2f})"
            cv2.putText(gesture_frame, text, (20, y_pos),
                       self.font, 0.5, color, 1)
        
        return gesture_frame
    
    def draw_performance_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw performance information."""
        perf_frame = frame.copy()
        
        # Update frame time
        current_time = time.time()
        if hasattr(self, 'last_frame_time'):
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
            
            # Keep only recent frame times
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
        
        self.last_frame_time = current_time
        
        # Calculate FPS
        if len(self.frame_times) > 1:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        else:
            fps = 0
        
        # Draw FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(perf_frame, fps_text, (frame.shape[1] - 150, 30),
                   self.font, 0.7, self.colors['green'], 2)
        
        # Draw frame size
        size_text = f"Size: {frame.shape[1]}x{frame.shape[0]}"
        cv2.putText(perf_frame, size_text, (frame.shape[1] - 150, 60),
                   self.font, 0.5, self.colors['white'], 1)
        
        return perf_frame
    
    def draw_status_panel(self, frame: np.ndarray, hand_results: Dict, 
                         gestures: List[Dict]) -> np.ndarray:
        """Draw system status panel."""
        status_frame = frame.copy()
        
        # Panel dimensions
        panel_x = frame.shape[1] - 250
        panel_y = 100
        panel_width = 240
        panel_height = 150
        
        # Create semi-transparent overlay
        overlay = status_frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     self.colors['black'], -1)
        status_frame = cv2.addWeighted(status_frame, 0.8, overlay, 0.2, 0)
        
        # Draw title
        cv2.putText(status_frame, "System Status", (panel_x + 10, panel_y + 25),
                   self.font, 0.6, self.colors['white'], 2)
        
        # Draw hand detection status
        hands_count = len(hand_results.get('hands', []))
        hands_text = f"Hands Detected: {hands_count}"
        hands_color = self.colors['green'] if hands_count > 0 else self.colors['red']
        cv2.putText(status_frame, hands_text, (panel_x + 10, panel_y + 50),
                   self.font, 0.5, hands_color, 1)
        
        # Draw gesture status
        gestures_count = len(gestures)
        gestures_text = f"Active Gestures: {gestures_count}"
        gestures_color = self.colors['green'] if gestures_count > 0 else self.colors['gray']
        cv2.putText(status_frame, gestures_text, (panel_x + 10, panel_y + 75),
                   self.font, 0.5, gestures_color, 1)
        
        # Draw detection quality
        if hands_count > 0:
            avg_confidence = np.mean([h.get('pose', {}).get('gesture_confidence', 0) 
                                    for h in hand_results.get('hands', [])])
            quality_text = f"Quality: {avg_confidence:.2f}"
            quality_color = self.colors['green'] if avg_confidence > 0.7 else self.colors['yellow']
            cv2.putText(status_frame, quality_text, (panel_x + 10, panel_y + 100),
                       self.font, 0.5, quality_color, 1)
        
        # Draw controls hint
        cv2.putText(status_frame, "Press 'q' to quit", (panel_x + 10, panel_y + 125),
                   self.font, 0.4, self.colors['gray'], 1)
        
        return status_frame
    
    def draw_crosshair(self, frame: np.ndarray, center: Tuple[int, int], 
                      size: int = 20, color: Tuple[int, int, int] = None) -> np.ndarray:
        """Draw a crosshair at the specified position."""
        if color is None:
            color = self.colors['red']
        
        crosshair_frame = frame.copy()
        x, y = center
        
        # Draw horizontal line
        cv2.line(crosshair_frame, (x - size, y), (x + size, y), color, 2)
        # Draw vertical line
        cv2.line(crosshair_frame, (x, y - size), (x, y + size), color, 2)
        # Draw center point
        cv2.circle(crosshair_frame, (x, y), 3, color, -1)
        
        return crosshair_frame
    
    def draw_grid(self, frame: np.ndarray, grid_size: int = 50) -> np.ndarray:
        """Draw a grid overlay on the frame."""
        grid_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw vertical lines
        for x in range(0, width, grid_size):
            cv2.line(grid_frame, (x, 0), (x, height), self.colors['gray'], 1)
        
        # Draw horizontal lines
        for y in range(0, height, grid_size):
            cv2.line(grid_frame, (0, y), (width, y), self.colors['gray'], 1)
        
        return grid_frame
    
    def create_info_panel(self, width: int, height: int, title: str, 
                         info_dict: Dict) -> np.ndarray:
        """Create a standalone information panel."""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw title
        cv2.putText(panel, title, (10, 30), self.font, 0.8, self.colors['white'], 2)
        
        # Draw information
        y_pos = 60
        for key, value in info_dict.items():
            text = f"{key}: {value}"
            cv2.putText(panel, text, (10, y_pos), self.font, 0.5, self.colors['green'], 1)
            y_pos += 25
        
        return panel
    
    def save_debug_frame(self, frame: np.ndarray, filename: str):
        """Save debug frame to file."""
        try:
            cv2.imwrite(filename, frame)
            self.logger.info(f"Saved debug frame to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save debug frame: {e}")
    
    def reset_performance_tracking(self):
        """Reset performance tracking metrics."""
        self.fps_history.clear()
        self.frame_times.clear()
        if hasattr(self, 'last_frame_time'):
            delattr(self, 'last_frame_time')