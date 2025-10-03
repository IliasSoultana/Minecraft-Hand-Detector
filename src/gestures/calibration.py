"""
Gesture Calibration Module
Handles gesture training and calibration
"""

import numpy as np
import cv2
import pickle
import yaml
import logging
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from ..core.hand_detector import HandDetector
from ..utils.camera_utils import CameraManager
from ..utils.visualization import Visualizer


class GestureCalibrator:
    """Handles gesture calibration and training."""
    
    def __init__(self, output_dir: str = "models/gestures"):
        """
        Initialize gesture calibrator.
        
        Args:
            output_dir: Directory to save calibration data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.hand_detector = HandDetector()
        self.camera_manager = None
        self.visualizer = Visualizer()
        
        # Calibration data
        self.gesture_samples = defaultdict(list)
        self.current_gesture = None
        self.recording = False
        
        # Calibration settings
        self.samples_per_gesture = 50
        self.recording_duration = 0.1  # seconds per sample
        self.min_confidence = 0.7
        
    def start_calibration_session(self, camera_id: int = 0):
        """Start interactive calibration session."""
        self.logger.info("Starting gesture calibration session")
        
        # Initialize camera
        self.camera_manager = CameraManager(camera_id=camera_id)
        
        try:
            self._run_calibration_loop()
        finally:
            if self.camera_manager:
                self.camera_manager.release()
    
    def _run_calibration_loop(self):
        """Main calibration loop."""
        print("\n=== Gesture Calibration Tool ===")
        print("Instructions:")
        print("- Press number keys (1-9) to start recording a gesture")
        print("- Hold the gesture steady while recording")
        print("- Press 's' to save current session")
        print("- Press 'q' to quit")
        print("- Press 'c' to clear current gesture data")
        print("- Press 'r' to reset all data")
        print()
        
        gesture_map = {
            ord('1'): 'closed_fist',
            ord('2'): 'open_palm',
            ord('3'): 'thumbs_up',
            ord('4'): 'peace_sign',
            ord('5'): 'point_right',
            ord('6'): 'point_left',
            ord('7'): 'l_shape',
            ord('8'): 'pinch',
            ord('9'): 'rock_on'
        }
        
        while True:
            frame = self.camera_manager.get_frame()
            if frame is None:
                continue
            
            # Detect hands
            hand_results = self.hand_detector.detect(frame)
            
            # Draw visualization
            display_frame = self._draw_calibration_interface(frame, hand_results)
            
            cv2.imshow('Gesture Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key in gesture_map:
                gesture_name = gesture_map[key]
                self._start_recording_gesture(gesture_name, hand_results)
            elif key == ord('s'):
                self._save_calibration_data()
            elif key == ord('c'):
                self._clear_current_gesture()
            elif key == ord('r'):
                self._reset_all_data()
        
        cv2.destroyAllWindows()
    
    def _draw_calibration_interface(self, frame: np.ndarray, 
                                  hand_results: Dict) -> np.ndarray:
        """Draw calibration interface."""
        display_frame = frame.copy()
        
        # Draw hand landmarks if detected
        if hand_results.get('hands'):
            display_frame = self.visualizer.draw_landmarks(display_frame, hand_results)
            display_frame = self.visualizer.draw_hand_info(display_frame, hand_results['hands'])
        
        # Draw instruction panel
        panel_height = 200
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, panel_height), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)
        
        # Instructions
        instructions = [
            "Gesture Calibration",
            "1: Closed Fist  2: Open Palm  3: Thumbs Up",
            "4: Peace Sign   5: Point Right 6: Point Left",
            "7: L Shape      8: Pinch       9: Rock On",
            "",
            f"Current: {self.current_gesture or 'None'}",
            f"Samples: {len(self.gesture_samples.get(self.current_gesture or '', []))}",
            "",
            "s: Save  c: Clear  r: Reset  q: Quit"
        ]
        
        for i, text in enumerate(instructions):
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            if i == 5 and self.current_gesture:
                color = (0, 255, 255)  # Highlight current gesture
            
            cv2.putText(display_frame, text, (20, 35 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Recording indicator
        if self.recording:
            cv2.circle(display_frame, (frame.shape[1] - 50, 50), 20, (0, 0, 255), -1)
            cv2.putText(display_frame, "REC", (frame.shape[1] - 70, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Progress bars for each gesture
        y_start = panel_height + 30
        for i, gesture_name in enumerate(['closed_fist', 'open_palm', 'thumbs_up', 
                                        'peace_sign', 'point_right', 'point_left',
                                        'l_shape', 'pinch', 'rock_on']):
            y_pos = y_start + i * 25
            samples_count = len(self.gesture_samples[gesture_name])
            progress = min(1.0, samples_count / self.samples_per_gesture)
            
            # Progress bar background
            cv2.rectangle(display_frame, (450, y_pos), (650, y_pos + 15), (50, 50, 50), -1)
            
            # Progress bar fill
            if progress > 0:
                fill_width = int(200 * progress)
                color = (0, 255, 0) if progress >= 1.0 else (0, 255, 255)
                cv2.rectangle(display_frame, (450, y_pos), (450 + fill_width, y_pos + 15), color, -1)
            
            # Label
            cv2.putText(display_frame, f"{gesture_name}: {samples_count}/{self.samples_per_gesture}",
                       (460, y_pos + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return display_frame
    
    def _start_recording_gesture(self, gesture_name: str, hand_results: Dict):
        """Start recording samples for a gesture."""
        if not hand_results.get('hands'):
            self.logger.warning("No hands detected. Cannot record gesture.")
            return
        
        # Check hand quality
        for hand_data in hand_results['hands']:
            confidence = hand_data.get('pose', {}).get('gesture_confidence', 0)
            if confidence < self.min_confidence:
                self.logger.warning(f"Hand confidence too low: {confidence:.2f}")
                return
        
        self.current_gesture = gesture_name
        self.recording = True
        
        # Record sample
        sample_data = {
            'timestamp': time.time(),
            'hands': hand_results['hands'],
            'gesture_name': gesture_name
        }
        
        self.gesture_samples[gesture_name].append(sample_data)
        self.recording = False
        
        self.logger.info(f"Recorded sample for {gesture_name}. "
                        f"Total: {len(self.gesture_samples[gesture_name])}")
        
        # Auto-save when reaching target samples
        if len(self.gesture_samples[gesture_name]) >= self.samples_per_gesture:
            self.logger.info(f"Reached target samples for {gesture_name}")
    
    def _save_calibration_data(self):
        """Save calibration data to files."""
        if not self.gesture_samples:
            self.logger.warning("No calibration data to save")
            return
        
        timestamp = int(time.time())
        
        # Save raw sample data
        raw_data_file = self.output_dir / f"calibration_raw_{timestamp}.pkl"
        with open(raw_data_file, 'wb') as f:
            pickle.dump(dict(self.gesture_samples), f)
        
        # Generate gesture configurations
        gesture_configs = {}
        for gesture_name, samples in self.gesture_samples.items():
            if len(samples) >= 5:  # Minimum samples for analysis
                config = self._analyze_gesture_samples(gesture_name, samples)
                gesture_configs[gesture_name] = config
        
        # Save gesture configuration
        config_file = self.output_dir / f"gestures_calibrated_{timestamp}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({'gestures': gesture_configs}, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Saved calibration data:")
        self.logger.info(f"  Raw data: {raw_data_file}")
        self.logger.info(f"  Config: {config_file}")
        
        print(f"\nCalibration data saved!")
        print(f"Raw data: {raw_data_file}")
        print(f"Config: {config_file}")
    
    def _analyze_gesture_samples(self, gesture_name: str, samples: List[Dict]) -> Dict:
        """Analyze gesture samples to create configuration."""
        if not samples:
            return {}
        
        # Extract pose data
        finger_patterns = []
        openness_values = []
        confidence_values = []
        
        for sample in samples:
            for hand_data in sample['hands']:
                pose = hand_data.get('pose', {})
                
                fingers_up = pose.get('fingers_up', [])
                if len(fingers_up) == 5:
                    finger_patterns.append(tuple(fingers_up))
                
                openness_values.append(pose.get('openness', 0))
                confidence_values.append(pose.get('gesture_confidence', 0))
        
        # Analyze finger patterns
        if finger_patterns:
            from collections import Counter
            most_common_pattern = Counter(finger_patterns).most_common(1)[0][0]
        else:
            most_common_pattern = (0, 0, 0, 0, 0)
        
        # Calculate statistics
        avg_openness = np.mean(openness_values) if openness_values else 0.5
        std_openness = np.std(openness_values) if openness_values else 0.1
        avg_confidence = np.mean(confidence_values) if confidence_values else 0.7
        
        # Create configuration
        config = {
            'name': gesture_name.replace('_', ' ').title(),
            'action': gesture_name,
            'description': f"Calibrated {gesture_name} gesture",
            'conditions': {
                'fingers_up': list(most_common_pattern),
                'min_confidence': max(0.5, avg_confidence - 0.1),
                'openness_threshold': max(0.1, avg_openness - std_openness),
                'finger_tolerance': 1
            },
            'sensitivity': 0.8,
            'hold_time': 0.1 if gesture_name in ['move_forward', 'move_left', 'move_right'] else 0.0,
            'calibration_stats': {
                'samples_count': len(samples),
                'avg_openness': float(avg_openness),
                'std_openness': float(std_openness),
                'avg_confidence': float(avg_confidence),
                'pattern_frequency': float(Counter(finger_patterns)[most_common_pattern] / len(finger_patterns)) if finger_patterns else 0.0
            }
        }
        
        return config
    
    def _clear_current_gesture(self):
        """Clear samples for current gesture."""
        if self.current_gesture and self.current_gesture in self.gesture_samples:
            del self.gesture_samples[self.current_gesture]
            self.logger.info(f"Cleared samples for {self.current_gesture}")
            print(f"Cleared samples for {self.current_gesture}")
    
    def _reset_all_data(self):
        """Reset all calibration data."""
        self.gesture_samples.clear()
        self.current_gesture = None
        self.logger.info("Reset all calibration data")
        print("Reset all calibration data")
    
    def load_calibration_data(self, data_file: str):
        """Load previously saved calibration data."""
        try:
            with open(data_file, 'rb') as f:
                self.gesture_samples = defaultdict(list, pickle.load(f))
            self.logger.info(f"Loaded calibration data from {data_file}")
        except Exception as e:
            self.logger.error(f"Failed to load calibration data: {e}")
    
    def export_gesture_config(self, output_file: str, gestures: List[str] = None):
        """Export gesture configuration for specific gestures."""
        if gestures is None:
            gestures = list(self.gesture_samples.keys())
        
        gesture_configs = {}
        for gesture_name in gestures:
            if gesture_name in self.gesture_samples:
                samples = self.gesture_samples[gesture_name]
                if len(samples) >= 5:
                    config = self._analyze_gesture_samples(gesture_name, samples)
                    gesture_configs[gesture_name] = config
        
        with open(output_file, 'w') as f:
            yaml.dump({'gestures': gesture_configs}, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Exported gesture config to {output_file}")
    
    def get_calibration_stats(self) -> Dict:
        """Get calibration statistics."""
        stats = {
            'total_gestures': len(self.gesture_samples),
            'total_samples': sum(len(samples) for samples in self.gesture_samples.values()),
            'gestures': {}
        }
        
        for gesture_name, samples in self.gesture_samples.items():
            stats['gestures'][gesture_name] = {
                'samples_count': len(samples),
                'completion': min(1.0, len(samples) / self.samples_per_gesture)
            }
        
        return stats


def quick_calibrate_gesture(gesture_name: str, samples_count: int = 20, 
                          camera_id: int = 0) -> Dict:
    """
    Quick calibration for a single gesture.
    
    Args:
        gesture_name: Name of gesture to calibrate
        samples_count: Number of samples to collect
        camera_id: Camera device ID
        
    Returns:
        Calibrated gesture configuration
    """
    calibrator = GestureCalibrator()
    calibrator.samples_per_gesture = samples_count
    
    # Initialize components
    camera_manager = CameraManager(camera_id=camera_id)
    hand_detector = HandDetector()
    
    samples = []
    
    print(f"\nQuick calibration for '{gesture_name}'")
    print(f"Hold the gesture steady. Collecting {samples_count} samples...")
    print("Press SPACE to record sample, 'q' to quit")
    
    try:
        sample_count = 0
        while sample_count < samples_count:
            frame = camera_manager.get_frame()
            if frame is None:
                continue
            
            # Detect hands
            hand_results = hand_detector.detect(frame)
            
            # Draw visualization
            display_frame = frame.copy()
            if hand_results.get('hands'):
                display_frame = calibrator.visualizer.draw_landmarks(display_frame, hand_results)
            
            # Progress indicator
            progress = sample_count / samples_count
            cv2.rectangle(display_frame, (10, 10), (310, 50), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (15, 15), (15 + int(290 * progress), 45), (0, 255, 0), -1)
            cv2.putText(display_frame, f"{gesture_name}: {sample_count}/{samples_count}",
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Quick Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and hand_results.get('hands'):
                # Record sample
                sample_data = {
                    'timestamp': time.time(),
                    'hands': hand_results['hands'],
                    'gesture_name': gesture_name
                }
                samples.append(sample_data)
                sample_count += 1
                print(f"Sample {sample_count}/{samples_count} recorded")
    
    finally:
        camera_manager.release()
        cv2.destroyAllWindows()
    
    # Analyze samples
    if len(samples) >= 5:
        config = calibrator._analyze_gesture_samples(gesture_name, samples)
        print(f"\nCalibration completed! {len(samples)} samples collected.")
        return config
    else:
        print(f"\nCalibration incomplete. Only {len(samples)} samples collected.")
        return {}


if __name__ == "__main__":
    # Run calibration session
    calibrator = GestureCalibrator()
    calibrator.start_calibration_session()