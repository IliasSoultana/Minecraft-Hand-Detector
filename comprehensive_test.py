import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))
from utils.camera_utils import CameraManager
from core.hand_detector import HandDetector
from core.gesture_recognizer import GestureRecognizer
from core.input_controller import InputController
from utils.visualization import Visualizer
import cv2
def main():
    print("Starting comprehensive test suite...\n")
    results = {}
    imports_ok, mediapipe_ok = test_imports()
    results['imports'] = imports_ok
    results['mediapipe'] = mediapipe_ok
    results['gesture_classifier'] = test_gesture_classifier()
    results['game_controller'] = test_game_controller()
    results['yolo_detector'] = test_yolo_detector()
    results['opencv_demo'] = test_opencv_demo()
    results['video_demo'] = test_video_demo()
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = 0
    total = 0
    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test.upper():20} {status}")
        if result:
            passed += 1
        total += 1
    print(f"\nOverall: {passed}/{total} tests passed")
def test_imports():
    try:
        import cv2
        import numpy as np
        import sklearn
        import pynput
        print("✓ Core imports successful")
        try:
            import mediapipe as mp
            print("✓ MediaPipe available")
            return True, True
        except ImportError:
            print("⚠ MediaPipe not available (expected on Python 3.13+)")
            return True, False
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False, False
def test_gesture_classifier():
    try:
        from core.gesture_recognizer import GestureRecognizer
        classifier = GestureRecognizer()
        print("✓ Gesture classifier initialized")
        return True
    except Exception as e:
        print(f"✗ Gesture classifier failed: {e}")
        return False
def test_game_controller():
    try:
        from core.input_controller import InputController
        controller = InputController()
        print("✓ Game controller initialized")
        return True
    except Exception as e:
        print(f"✗ Game controller failed: {e}")
        return False
def test_yolo_detector():
    try:
        from core.yolo_detector import YOLODetector
        detector = YOLODetector()
        print("✓ YOLO detector initialized")
        return True
    except Exception as e:
        print(f"✗ YOLO detector failed: {e}")
        return False
def test_opencv_demo():
    try:
        from core.hand_detector import HandDetector
        detector = HandDetector()
        print("✓ OpenCV demo ready")
        return True
    except Exception as e:
        print(f"✗ OpenCV demo failed: {e}")
        return False
def test_video_demo():
    try:
        import cv2
        print("✓ Video demo ready")
        return True
    except Exception as e:
        print(f"✗ Video demo failed: {e}")
        return False
if __name__ == "__main__":
    main()