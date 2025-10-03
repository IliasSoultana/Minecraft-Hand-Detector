"""
Test the complete Minecraft Gesture Controller system
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from core.hand_detector import HandDetector
from core.gesture_recognizer import GestureRecognizer
from core.input_controller import InputController
from utils.camera_utils import CameraManager
import cv2
import time

def test_system():
    """Test the complete gesture control system."""
    print("Testing Minecraft Gesture Controller System")
    print("=" * 50)
    
    # Test 1: Camera Connection
    print("1. Testing IP Camera Connection...")
    try:
        camera_manager = CameraManager(ip_camera_url="http://192.168.178.156:8080/video")
        frame = camera_manager.get_frame()
        if frame is not None:
            print("✓ IP Camera connected successfully")
            print(f"✓ Frame resolution: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print("✗ Failed to get frame from camera")
            return False
    except Exception as e:
        print(f"✗ Camera connection failed: {e}")
        return False
    
    # Test 2: Hand Detection
    print("\n2. Testing Hand Detection...")
    try:
        hand_detector = HandDetector()
        result = hand_detector.detect(frame)
        hands_count = len(result.get('hands', []))
        print(f"✓ Hand detector initialized (detected {hands_count} hands)")
    except Exception as e:
        print(f"✗ Hand detection failed: {e}")
        return False
    
    # Test 3: Gesture Recognition
    print("\n3. Testing Gesture Recognition...")
    try:
        gesture_recognizer = GestureRecognizer(config_path='config/gestures.yaml')
        gestures = gesture_recognizer.recognize(result)
        print(f"✓ Gesture recognizer initialized (found {len(gestures)} gestures)")
    except Exception as e:
        print(f"✗ Gesture recognition failed: {e}")
        return False
    
    # Test 4: Input Controller
    print("\n4. Testing Input Controller...")
    try:
        input_controller = InputController(config_path='config/controls.yaml')
        print("✓ Input controller initialized")
        
        # Test gesture processing (without actually sending keys)
        for gesture in gestures:
            print(f"  - Would process gesture: {gesture.get('name', 'Unknown')}")
            
    except Exception as e:
        print(f"✗ Input controller failed: {e}")
        return False
    
    # Test 5: Full Pipeline
    print("\n5. Testing Full Pipeline...")
    try:
        frame_count = 0
        for i in range(5):  # Test 5 frames
            frame = camera_manager.get_frame()
            if frame is not None:
                hand_results = hand_detector.detect(frame)
                gestures = gesture_recognizer.recognize(hand_results)
                frame_count += 1
                
                if frame_count % 1 == 0:
                    hands_detected = len(hand_results.get('hands', []))
                    gestures_found = len(gestures)
                    print(f"  Frame {frame_count}: {hands_detected} hands, {gestures_found} gestures")
            
            time.sleep(0.1)  # Small delay
            
        print(f"✓ Full pipeline processed {frame_count} frames successfully")
        
    except Exception as e:
        print(f"✗ Full pipeline test failed: {e}")
        return False
    
    # Cleanup
    try:
        camera_manager.release()
        hand_detector.release()
        input_controller.release_all_keys()
        print("\n✓ Cleanup completed successfully")
    except Exception as e:
        print(f"✗ Cleanup failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! Minecraft Gesture Controller is ready!")
    print("\nTo use the system:")
    print("1. Start Minecraft")
    print("2. Run: python src/main.py --ip-camera http://192.168.178.156:8080/video")
    print("3. Use hand gestures to control the game")
    print("\nSupported gestures:")
    print("- Fist: Attack/Mine")
    print("- Open Palm: Move Forward")
    print("- Point: Use Item/Place Block")
    print("- Peace Sign: Jump")
    print("- Thumbs Up: Sprint")
    print("- Thumbs Down: Sneak")
    
    return True

if __name__ == "__main__":
    test_system()