import numpy as np
import yaml
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pickle
import cv2
class GestureRecognizer:
    def __init__(self, config_path: str = "config/gestures.yaml"):
        self.config_path = config_path
        self.gestures_config = self._load_config()
        self.gesture_history = []
        self.history_length = 5
        self.logger = logging.getLogger(__name__)
        self.current_gesture = None
        self.gesture_start_time = None
        self.gesture_hold_threshold = 0.5
        self.min_confidence = 0.6
        self.stable_frames_required = 3
        self.logger.info(f"GestureRecognizer initialized with {len(self.gestures_config)} gestures")
    def _load_config(self) -> Dict:
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config.get('gestures', {})
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {self.config_path}. Using default gestures.")
            return self._get_default_gestures()
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing config file: {e}")
            return self._get_default_gestures()
    def _get_default_gestures(self) -> Dict:
        return {
            'closed_fist': {
                'name': 'Closed Fist',
                'action': 'move_forward',
                'conditions': {
                    'fingers_up': [0, 0, 0, 0, 0],
                    'min_confidence': 0.7,
                    'openness_threshold': 0.3
                }
            },
            'open_palm': {
                'name': 'Open Palm',
                'action': 'stop_all',
                'conditions': {
                    'fingers_up': [1, 1, 1, 1, 1],
                    'min_confidence': 0.8,
                    'openness_threshold': 0.7
                }
            },
            'thumbs_up': {
                'name': 'Thumbs Up',
                'action': 'jump',
                'conditions': {
                    'fingers_up': [1, 0, 0, 0, 0],
                    'min_confidence': 0.8
                }
            },
            'peace_sign': {
                'name': 'Peace Sign',
                'action': 'crouch',
                'conditions': {
                    'fingers_up': [0, 1, 1, 0, 0],
                    'min_confidence': 0.8
                }
            },
            'point_right': {
                'name': 'Point Right',
                'action': 'turn_right',
                'conditions': {
                    'fingers_up': [0, 1, 0, 0, 0],
                    'min_confidence': 0.7,
                    'orientation_range': [0, 45]
                }
            },
            'point_left': {
                'name': 'Point Left',
                'action': 'turn_left',
                'conditions': {
                    'fingers_up': [0, 1, 0, 0, 0],
                    'min_confidence': 0.7,
                    'orientation_range': [135, 225]
                }
            },
            'l_shape': {
                'name': 'L Shape',
                'action': 'inventory',
                'conditions': {
                    'fingers_up': [1, 1, 0, 0, 0],
                    'min_confidence': 0.8
                }
            },
            'pinch': {
                'name': 'Pinch',
                'action': 'use_item',
                'conditions': {
                    'custom_detector': 'detect_pinch',
                    'min_confidence': 0.7
                }
            }
        }
    def recognize(self, hand_results: Dict) -> List[Dict]:
        recognized_gestures = []
        if not hand_results.get('hands'):
            return recognized_gestures
        for hand_data in hand_results['hands']:
            gesture = self._recognize_single_hand(hand_data)
            if gesture:
                recognized_gestures.append(gesture)
        self._update_gesture_history(recognized_gestures)
        filtered_gestures = self._apply_temporal_filtering(recognized_gestures)
        return filtered_gestures
    def _recognize_single_hand(self, hand_data: Dict) -> Optional[Dict]:
        pose = hand_data.get('pose', {})
        confidence = pose.get('gesture_confidence', 0.0)
        if confidence < self.min_confidence:
            return None
        for gesture_name, gesture_config in self.gestures_config.items():
            if self._matches_gesture(pose, gesture_config, hand_data):
                return {
                    'name': gesture_name,
                    'display_name': gesture_config.get('name', gesture_name),
                    'action': gesture_config.get('action'),
                    'confidence': confidence,
                    'hand_label': hand_data.get('label'),
                    'timestamp': cv2.getTickCount()
                }
        return None
    def _matches_gesture(self, pose: Dict, gesture_config: Dict, hand_data: Dict) -> bool:
        conditions = gesture_config.get('conditions', {})
        min_confidence = conditions.get('min_confidence', 0.5)
        if pose.get('gesture_confidence', 0) < min_confidence:
            return False
        if 'fingers_count' in conditions:
            expected_count = conditions['fingers_count']
            actual_count = pose.get('fingers_count', 0)
            if actual_count != expected_count:
                return False
        if 'is_closed_fist' in conditions:
            expected_fist = conditions['is_closed_fist']
            actual_fist = pose.get('is_closed_fist', False)
            if actual_fist != expected_fist:
                return False
        if 'is_open_palm' in conditions:
            expected_palm = conditions['is_open_palm']
            actual_palm = pose.get('is_open_palm', False)
            if actual_palm != expected_palm:
                return False
        return True
    def detect_pinch(self, pose: Dict, hand_data: Dict, conditions: Dict) -> bool:
        landmarks = hand_data.get('landmarks', [])
        if len(landmarks) < 21:
            return False
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = np.sqrt(
            (thumb_tip['x'] - index_tip['x'])**2 + 
            (thumb_tip['y'] - index_tip['y'])**2
        )
        pinch_threshold = conditions.get('pinch_threshold', 0.05)
        return distance < pinch_threshold
    def _update_gesture_history(self, gestures: List[Dict]):
        self.gesture_history.append(gestures)
        if len(self.gesture_history) > self.history_length:
            self.gesture_history.pop(0)
    def _apply_temporal_filtering(self, current_gestures: List[Dict]) -> List[Dict]:
        if len(self.gesture_history) < 3:
            return current_gestures
        filtered_gestures = []
        for gesture in current_gestures:
            gesture_name = gesture['name']
            recent_count = 0
            for hist_gestures in self.gesture_history[-3:]:
                if any(g['name'] == gesture_name for g in hist_gestures):
                    recent_count += 1
            if recent_count >= 2:
                filtered_gestures.append(gesture)
        return filtered_gestures
    def add_custom_gesture(self, name: str, config: Dict):
        self.gestures_config[name] = config
        self.logger.info(f"Added custom gesture: {name}")
    def calibrate_gesture(self, gesture_name: str, samples: List[Dict]) -> Dict:
        if not samples:
            return {}
        finger_patterns = []
        openness_values = []
        orientation_angles = []
        for sample in samples:
            pose = sample.get('pose', {})
            finger_patterns.append(pose.get('fingers_up', []))
            openness_values.append(pose.get('openness', 0))
            orientation_angles.append(pose.get('orientation_angle', 0))
        if finger_patterns:
            finger_tuples = [tuple(fp) for fp in finger_patterns if len(fp) == 5]
            if finger_tuples:
                from collections import Counter
                most_common_fingers = Counter(finger_tuples).most_common(1)[0][0]
            else:
                most_common_fingers = [0, 0, 0, 0, 0]
        else:
            most_common_fingers = [0, 0, 0, 0, 0]
        avg_openness = np.mean(openness_values) if openness_values else 0.5
        std_openness = np.std(openness_values) if openness_values else 0.1
        calibrated_config = {
            'name': gesture_name.replace('_', ' ').title(),
            'action': gesture_name,
            'conditions': {
                'fingers_up': list(most_common_fingers),
                'min_confidence': 0.7,
                'openness_threshold': max(0.1, avg_openness - std_openness),
                'finger_tolerance': 1
            }
        }
        self.logger.info(f"Calibrated gesture '{gesture_name}' from {len(samples)} samples")
        return calibrated_config
    def save_gestures_config(self, output_path: str = None):
        if output_path is None:
            output_path = self.config_path
        config_data = {'gestures': self.gestures_config}
        with open(output_path, 'w') as file:
            yaml.dump(config_data, file, default_flow_style=False, indent=2)
        self.logger.info(f"Saved gesture configuration to {output_path}")
    def get_gesture_info(self, gesture_name: str) -> Dict:
        return self.gestures_config.get(gesture_name, {})