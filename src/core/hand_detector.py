"""
Hand Detection Module using MediaPipe with OpenCV fallback
Handles real-time hand landmark detection and tracking
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

# Try MediaPipe first, fallback to OpenCV
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available. Using OpenCV fallback.")
    from .hand_detector_opencv import OpenCVHandDetector


class HandDetector:
    """Hand detection and landmark extraction using MediaPipe with OpenCV fallback."""
    
    def __init__(self, confidence: float = 0.7, max_hands: int = 2):
        """
        Initialize the hand detector.
        
        Args:
            confidence: Minimum detection confidence threshold
            max_hands: Maximum number of hands to detect
        """
        self.confidence = confidence
        self.max_hands = max_hands
        self.logger = logging.getLogger(__name__)
        
        if not MEDIAPIPE_AVAILABLE:
            # Use OpenCV fallback
            self.logger.info("Using OpenCV hand detection fallback")
            self._detector = OpenCVHandDetector(confidence, max_hands)
            self.hands = None
            return
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create hands detection model
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=confidence,
            min_tracking_confidence=0.5
        )
        
        self.logger.info(f"MediaPipe HandDetector initialized with confidence={confidence}, max_hands={max_hands}")
        
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Detect hands in the given frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary containing detection results
        """
        if not MEDIAPIPE_AVAILABLE:
            # Use OpenCV fallback
            return self._detector.detect(frame)
        
        # Use MediaPipe detection (original code)
        return self._detect_mediapipe(frame)
    
    def _detect_mediapipe(self, frame: np.ndarray) -> Dict:
        """
        Detect hands in the given frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary containing detection results
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Extract hand information
        hands_data = []
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand label (Left/Right)
                hand_label = results.multi_handedness[idx].classification[0].label
                hand_score = results.multi_handedness[idx].classification[0].score
                
                # Extract landmark coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                
                # Calculate bounding box
                bbox = self._calculate_bounding_box(landmarks, frame.shape)
                
                # Calculate hand orientation and pose
                hand_pose = self._calculate_hand_pose(landmarks)
                
                hands_data.append({
                    'label': hand_label,
                    'score': hand_score,
                    'landmarks': landmarks,
                    'bounding_box': bbox,
                    'pose': hand_pose,
                    'raw_landmarks': hand_landmarks
                })
        
        return {
            'hands': hands_data,
            'frame_shape': frame.shape,
            'detection_time': cv2.getTickCount()
        }
    
    def _calculate_bounding_box(self, landmarks: List[Dict], frame_shape: Tuple) -> Dict:
        """Calculate bounding box for hand landmarks."""
        height, width = frame_shape[:2]
        
        x_coords = [lm['x'] * width for lm in landmarks]
        y_coords = [lm['y'] * height for lm in landmarks]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)
        
        return {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'width': x_max - x_min,
            'height': y_max - y_min,
            'center': ((x_min + x_max) // 2, (y_min + y_max) // 2)
        }
    
    def _calculate_hand_pose(self, landmarks: List[Dict]) -> Dict:
        """Calculate hand pose information."""
        # Key landmark indices (MediaPipe hand model)
        WRIST = 0
        THUMB_TIP = 4
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        RING_TIP = 16
        PINKY_TIP = 20
        
        INDEX_PIP = 6
        MIDDLE_PIP = 10
        RING_PIP = 14
        PINKY_PIP = 18
        
        # Calculate finger states
        fingers_up = []
        
        # Thumb (special case - compare x coordinates)
        if landmarks[THUMB_TIP]['x'] > landmarks[THUMB_TIP - 1]['x']:
            fingers_up.append(1)  # Right hand thumb up
        else:
            fingers_up.append(0)
            
        # Other fingers (compare y coordinates)
        for tip, pip in [(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP), 
                        (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)]:
            if landmarks[tip]['y'] < landmarks[pip]['y']:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        # Calculate hand orientation
        wrist = np.array([landmarks[WRIST]['x'], landmarks[WRIST]['y']])
        index_mcp = np.array([landmarks[5]['x'], landmarks[5]['y']])  # Index MCP
        
        # Vector from wrist to index MCP
        orientation_vector = index_mcp - wrist
        angle = np.arctan2(orientation_vector[1], orientation_vector[0])
        
        # Calculate hand openness (distance between fingertips)
        openness = self._calculate_hand_openness(landmarks)
        
        return {
            'fingers_up': fingers_up,
            'fingers_count': sum(fingers_up),
            'orientation_angle': np.degrees(angle),
            'openness': openness,
            'is_closed_fist': sum(fingers_up) == 0,
            'is_open_palm': sum(fingers_up) == 5,
            'gesture_confidence': self._calculate_gesture_confidence(landmarks)
        }
    
    def _calculate_hand_openness(self, landmarks: List[Dict]) -> float:
        """Calculate how open the hand is (0=closed, 1=fully open)."""
        # Calculate distances between fingertips
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        
        distances = []
        for i in range(len(fingertips)):
            for j in range(i + 1, len(fingertips)):
                p1 = landmarks[fingertips[i]]
                p2 = landmarks[fingertips[j]]
                dist = np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
                distances.append(dist)
        
        # Normalize by average distance (empirically determined)
        avg_distance = np.mean(distances)
        normalized_openness = min(1.0, avg_distance / 0.3)  # 0.3 is empirical threshold
        
        return normalized_openness
    
    def _calculate_gesture_confidence(self, landmarks: List[Dict]) -> float:
        """Calculate confidence score for gesture detection."""
        # Simple confidence based on landmark stability and visibility
        # This is a placeholder - can be enhanced with more sophisticated methods
        
        # Check if all landmarks are within frame
        all_visible = all(0 <= lm['x'] <= 1 and 0 <= lm['y'] <= 1 for lm in landmarks)
        
        # Calculate landmark spread (higher spread = more confident detection)
        x_coords = [lm['x'] for lm in landmarks]
        y_coords = [lm['y'] for lm in landmarks]
        spread = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
        
        # Normalize spread score
        spread_score = min(1.0, spread / 0.1)  # 0.1 is empirical threshold
        
        base_confidence = 0.8 if all_visible else 0.5
        return base_confidence * spread_score
    
    def get_landmark_positions(self, hand_data: Dict, frame_shape: Tuple) -> np.ndarray:
        """
        Convert normalized landmarks to pixel coordinates.
        
        Args:
            hand_data: Hand detection result
            frame_shape: Shape of the original frame
            
        Returns:
            Array of landmark positions in pixels
        """
        height, width = frame_shape[:2]
        landmarks = hand_data['landmarks']
        
        positions = []
        for landmark in landmarks:
            x = int(landmark['x'] * width)
            y = int(landmark['y'] * height)
            positions.append([x, y])
            
        return np.array(positions)
    
    def release(self):
        """Release MediaPipe resources."""
        if not MEDIAPIPE_AVAILABLE:
            if hasattr(self, '_detector'):
                self._detector.release()
            return
            
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
        self.logger.info("HandDetector resources released")