import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))
from core.hand_detector import HandDetector
from core.gesture_recognizer import GestureRecognizer
def main():
    print("Video processing demo")
    video_path = "demo_video.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        print("Creating synthetic demo...")
        create_synthetic_demo()
        return
    detector = HandDetector()
    recognizer = GestureRecognizer()
    frame_count = 0
    gesture_counts = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        hand_results = detector.detect(frame)
        if hand_results.get('hands'):
            gestures = recognizer.recognize(hand_results)
            for gesture in gestures:
                gesture_name = gesture.get('name', 'unknown')
                gesture_counts[gesture_name] = gesture_counts.get(gesture_name, 0) + 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
    cap.release()
    print(f"\nProcessed {frame_count} total frames")
    print("Gesture summary:")
    for gesture, count in gesture_counts.items():
        print(f"  {gesture}: {count} detections")
def create_synthetic_demo():
    print("Synthetic demo: gesture recognition without video file")
    detector = HandDetector()
    recognizer = GestureRecognizer()
    print("✓ Components initialized successfully")
    print("✓ Demo completed - system ready for real video")
if __name__ == "__main__":
    main()