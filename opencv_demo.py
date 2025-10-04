import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))
from core.hand_detector import HandDetector
def main():
    print("OpenCV-only hand detection demo")
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = detector.detect(frame)
        if results.get('hands'):
            for hand in results['hands']:
                landmarks = hand.get('landmarks', [])
                for landmark in landmarks:
                    x = int(landmark.get('x', 0) * frame.shape[1])
                    y = int(landmark.get('y', 0) * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow('OpenCV Hand Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()