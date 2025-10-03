"""
OpenCV-Only Hand Detection (Python 3.13 Compatible)
Alternative hand detector using OpenCV DNN for Python 3.13 compatibility
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple


class OpenCVHandDetector:
    """OpenCV-based hand detection for Python 3.13 compatibility."""
    
    def __init__(self, confidence: float = 0.7, max_hands: int = 2):
        """
        Initialize OpenCV hand detector.
        
        Args:
            confidence: Minimum detection confidence
            max_hands: Maximum number of hands to detect
        """
        self.confidence = confidence
        self.max_hands = max_hands
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenCV-based hand detector
        self.detection_method = "contour"  # Use contour-based detection
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        
        # Skin color detection ranges (HSV) - balanced for better detection
        self.skin_lower = np.array([0, 25, 70], dtype=np.uint8)
        self.skin_upper = np.array([25, 255, 255], dtype=np.uint8)
        
        self.logger.info(f"OpenCV HandDetector initialized using {self.detection_method} method")
    
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Detect hands using OpenCV contour and skin color detection.
        
        Args:
            frame: Input image frame
            
        Returns:
            Detection results dictionary
        """
        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create skin mask
        skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply Gaussian blur
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
        
        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and process contours
        hands_data = []
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for i, contour in enumerate(contours[:self.max_hands]):
            area = cv2.contourArea(contour)
            
            # Filter by minimum area (balanced threshold)
            if area < 2000:  # Reduced for better hand detection
                continue
                
            # Filter by maximum area to avoid large false detections
            if area > 50000:  # Maximum hand area
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (hands are roughly rectangular, more restrictive)
            aspect_ratio = w / h
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:  # More realistic hand proportions
                continue
                
            # Filter by minimum dimensions
            if w < 50 or h < 50:  # Minimum hand size in pixels
                continue
            
            # Create hand data
            landmarks = self._create_mock_landmarks(x, y, w, h, frame.shape)
            hand_pose = self._calculate_simple_pose(landmarks, frame[y:y+h, x:x+w], contour)
            
            hands_data.append({
                'label': 'Right' if x > frame.shape[1] // 2 else 'Left',
                'score': min(0.9, max(0.6, area / 15000)),  # More realistic confidence scoring
                'landmarks': landmarks,
                'bounding_box': {
                    'x_min': x, 'y_min': y,
                    'x_max': x + w, 'y_max': y + h,
                    'width': w, 'height': h,
                    'center': (x + w//2, y + h//2)
                },
                'pose': hand_pose,
                'raw_landmarks': None
            })
        
        return {
            'hands': hands_data,
            'frame_shape': frame.shape,
            'detection_time': cv2.getTickCount()
        }
    
    def _create_mock_landmarks(self, x: int, y: int, w: int, h: int, frame_shape: Tuple) -> List[Dict]:
        """Create simplified hand landmarks."""
        height, width = frame_shape[:2]
        landmarks = []
        
        # Create 21 mock landmarks in hand region
        for i in range(21):
            # Distribute points across hand region
            lx = (x + (i % 5) * w // 4) / width
            ly = (y + (i // 5) * h // 4) / height
            
            landmarks.append({
                'x': lx,
                'y': ly,
                'z': 0.0
            })
        
        return landmarks
    
    def _calculate_simple_pose(self, landmarks: List[Dict], hand_region: np.ndarray, contour: np.ndarray = None) -> Dict:
        """Calculate simplified hand pose with optional contour analysis."""
        # Simple gesture detection based on hand region analysis
        if hand_region.size == 0:
            return self._default_pose()
        
        # Analyze hand region brightness/texture
        gray_region = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray_region)
        std_intensity = np.std(gray_region)
        
        # Enhanced analysis with contour if available
        finger_count = 0
        openness = min(1.0, std_intensity / 50.0)  # Higher variation = more open
        
        if contour is not None:
            try:
                # Calculate contour properties for better gesture detection
                area = cv2.contourArea(contour)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                
                # Calculate solidity (convexity)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Try to count fingers using convexity defects
                hull_indices = cv2.convexHull(contour, returnPoints=False)
                if len(hull_indices) > 3:
                    defects = cv2.convexityDefects(contour, hull_indices)
                    
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(contour[s][0])
                            end = tuple(contour[e][0])
                            far = tuple(contour[f][0])
                            
                            # Calculate angle between finger and palm
                            a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                            b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                            c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                            
                            if b > 0 and c > 0:
                                angle = np.arccos(max(-1, min(1, (b**2 + c**2 - a**2) / (2*b*c))))
                                
                                # If angle is acute and defect is deep enough, count as finger
                                if angle <= np.pi/2 and d > 5000:
                                    finger_count += 1
                
                # Adjust openness based on solidity
                if solidity > 0.8:
                    openness = 0.2  # Very compact = closed fist
                elif solidity < 0.5:
                    openness = 0.9  # Very spread out = open hand
                    
            except Exception:
                # Fall back to texture analysis
                pass
        
        # Determine finger state
        if finger_count == 0:
            # Use texture analysis fallback
            fingers_up = [1, 1, 1, 1, 1] if openness > 0.6 else [0, 0, 0, 0, 0]
        else:
            # Use contour-based detection
            fingers_up = [1] * min(5, finger_count + 1) + [0] * max(0, 4 - finger_count)
        
        final_finger_count = sum(fingers_up)
        
        return {
            'fingers_up': fingers_up,
            'fingers_count': final_finger_count,
            'orientation_angle': 0.0,
            'openness': openness,
            'is_closed_fist': final_finger_count == 0,
            'is_open_palm': final_finger_count == 5,
            'gesture_confidence': 0.7 if contour is not None else 0.5  # More conservative confidence
        }
    
    def _default_pose(self) -> Dict:
        """Return default pose for empty detection."""
        return {
            'fingers_up': [0, 0, 0, 0, 0],
            'fingers_count': 0,
            'orientation_angle': 0.0,
            'openness': 0.0,
            'is_closed_fist': True,
            'is_open_palm': False,
            'gesture_confidence': 0.0
        }
    
    def release(self):
        """Release resources."""
        self.logger.info("OpenCV HandDetector resources released")


# Create compatibility alias
HandDetector = OpenCVHandDetector