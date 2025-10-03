"""
Quick test of the gesture controller with IP camera
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.camera_utils import CameraManager
from core.hand_detector import HandDetector
from utils.visualization import Visualizer
import cv2

def main():
    print("Testing Gesture Controller with IP Camera")
    
    # Test camera
    ip_url = "http://192.168.178.156:8080/video"
    print(f"Connecting to: {ip_url}")
    
    try:
        # Initialize camera with IP URL
        camera_manager = CameraManager(ip_camera_url=ip_url)
        print("✓ Camera connected!")
        
        # Initialize hand detector
        hand_detector = HandDetector()
        print("✓ Hand detector initialized!")
        
        # Initialize visualizer
        visualizer = Visualizer()
        print("✓ Visualizer ready!")
        
        print("\nStarting gesture detection...")
        print("Press 'q' to quit")
        
        frame_count = 0
        while True:
            frame = camera_manager.get_frame()
            if frame is None:
                print("No frame received")
                continue
            
            frame_count += 1
            
            # Detect hands
            hand_results = hand_detector.detect(frame)
            
            # Draw visualization
            display_frame = visualizer.draw_debug_info(frame, hand_results, [])
            
            # Show info
            if frame_count % 30 == 0:  # Every second at 30fps
                hands_count = len(hand_results.get('hands', []))
                print(f"Frame {frame_count}: {hands_count} hands detected")
            
            # Display
            cv2.imshow('Gesture Test', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        camera_manager.release()
        cv2.destroyAllWindows()
        print("✓ Test completed!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()