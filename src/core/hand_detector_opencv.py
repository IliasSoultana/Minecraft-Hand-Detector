import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
class OpenCVHandDetector:
    def __init__(self, confidence: float = 0.7, max_hands: int = 2):
        self.confidence = confidence
        self.max_hands = max_hands
        self.logger = logging.getLogger(__name__)
        self.detection_method = "contour"
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.skin_lower = np.array([0, 25, 70], dtype=np.uint8)
        self.skin_upper = np.array([25, 255, 255], dtype=np.uint8)
        self.logger.info(f"OpenCV HandDetector initialized using {self.detection_method} method")
    def detect(self, frame: np.ndarray) -> Dict:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hands_data = []
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for i, contour in enumerate(contours[:self.max_hands]):
            area = cv2.contourArea(contour)
            if area < 2000:
                continue
            if area > 50000:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            if w < 50 or h < 50:
                continue
            landmarks = self._create_mock_landmarks(x, y, w, h, frame.shape)
            hand_pose = self._calculate_simple_pose(landmarks, frame[y:y+h, x:x+w], contour)
            hands_data.append({
                'label': 'Right' if x > frame.shape[1] // 2 else 'Left',
                'score': min(0.9, max(0.6, area / 15000)),
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
        height, width = frame_shape[:2]
        landmarks = []
        for i in range(21):
            lx = (x + (i % 5) * w // 4) / width
            ly = (y + (i // 5) * h // 4) / height
            landmarks.append({
                'x': lx,
                'y': ly,
                'z': 0.0
            })
        return landmarks
    def _calculate_simple_pose(self, landmarks: List[Dict], hand_region: np.ndarray, contour: np.ndarray = None) -> Dict:
        if hand_region.size == 0:
            return self._default_pose()
        gray_region = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray_region)
        std_intensity = np.std(gray_region)
        finger_count = 0
        openness = min(1.0, std_intensity / 50.0)
        if contour is not None:
            try:
                area = cv2.contourArea(contour)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                hull_indices = cv2.convexHull(contour, returnPoints=False)
                if len(hull_indices) > 3:
                    defects = cv2.convexityDefects(contour, hull_indices)
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(contour[s][0])
                            end = tuple(contour[e][0])
                            far = tuple(contour[f][0])
                            a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                            b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                            c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                            if b > 0 and c > 0:
                                angle = np.arccos(max(-1, min(1, (b**2 + c**2 - a**2) / (2*b*c))))
                                if angle <= np.pi/2 and d > 5000:
                                    finger_count += 1
                if solidity > 0.8:
                    openness = 0.2
                elif solidity < 0.5:
                    openness = 0.9
            except Exception:
                pass
        if finger_count == 0:
            fingers_up = [1, 1, 1, 1, 1] if openness > 0.6 else [0, 0, 0, 0, 0]
        else:
            fingers_up = [1] * min(5, finger_count + 1) + [0] * max(0, 4 - finger_count)
        final_finger_count = sum(fingers_up)
        return {
            'fingers_up': fingers_up,
            'fingers_count': final_finger_count,
            'orientation_angle': 0.0,
            'openness': openness,
            'is_closed_fist': final_finger_count == 0,
            'is_open_palm': final_finger_count == 5,
            'gesture_confidence': 0.7 if contour is not None else 0.5
        }
    def _default_pose(self) -> Dict:
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
        self.logger.info("OpenCV HandDetector resources released")
HandDetector = OpenCVHandDetector