import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from core.hand_detector import HandDetector
from core.gesture_recognizer import GestureRecognizer
from utils.camera_utils import CameraManager
from utils.visualization import Visualizer
def main():
    print("Basic Gesture Detection Example")
    print("Press 'q' to quit")
    camera_manager = CameraManager(camera_id=0, resolution=(640, 480))
    hand_detector = HandDetector(confidence=0.7, max_hands=2)
    gesture_recognizer = GestureRecognizer()
    visualizer = Visualizer()
    try:
        while True:
            frame = camera_manager.get_frame()
            if frame is None:
                print("Failed to get frame")
                continue
            hand_results = hand_detector.detect(frame)
            gestures = gesture_recognizer.recognize(hand_results)
            display_frame = visualizer.draw_debug_info(frame, hand_results, gestures)
            display_frame = visualizer.draw_landmarks(display_frame, hand_results)
            cv2.imshow('Basic Gesture Detection', display_frame)
            if gestures:
                for gesture in gestures:
                    print(f"Detected: {gesture['display_name']} -> {gesture['action']} "
                          f"(confidence: {gesture['confidence']:.2f})")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        camera_manager.release()
        cv2.destroyAllWindows()
        print("Example completed")
if __name__ == "__main__":
    main()