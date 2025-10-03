"""
IP Webcam Setup Example
Example for connecting to phone camera using IP Webcam app
"""

import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.camera_utils import CameraManager


def test_ip_camera(ip_url: str):
    """Test IP camera connection."""
    print(f"Testing IP camera: {ip_url}")
    
    try:
        # Initialize camera manager with IP camera
        camera_manager = CameraManager(ip_camera_url=ip_url)
        
        print("✓ Connected successfully!")
        
        # Get camera info
        info = camera_manager.get_camera_info()
        print(f"Camera type: {info['type']}")
        print(f"Source: {info['source']}")
        print(f"Resolution: {info.get('resolution', 'Unknown')}")
        
        print("\nPress 'q' to quit, 'i' for info")
        
        frame_count = 0
        while True:
            frame = camera_manager.get_frame()
            if frame is not None:
                frame_count += 1
                
                # Add overlay info
                cv2.putText(frame, f"IP Camera Test", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_count}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {camera_manager.fps_counter.get_fps():.1f}", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('IP Camera Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('i'):
                info = camera_manager.get_camera_info()
                print(f"\nCamera Info:")
                for k, v in info.items():
                    print(f"  {k}: {v}")
        
        camera_manager.release()
        cv2.destroyAllWindows()
        print("✓ Test completed successfully")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check if IP Webcam app is running on your phone")
        print("2. Verify the IP address and port")
        print("3. Make sure phone and PC are on same network")
        print("4. Try different video URLs (see examples below)")


def main():
    """Main function with IP camera examples."""
    print("IP Webcam Setup and Test")
    print("=" * 40)
    
    print("\n📱 IP Webcam App Setup:")
    print("1. Install 'IP Webcam' app on your Android phone")
    print("2. Open the app and scroll down")
    print("3. Tap 'Start server'")
    print("4. Note the IP address shown (e.g., 192.168.1.100:8080)")
    
    print("\n🔗 Common IP Camera URLs:")
    print("- Video stream: http://IP:PORT/video")
    print("- MJPEG stream: http://IP:PORT/videofeed")
    print("- With auth: http://username:password@IP:PORT/video")
    
    print("\n📋 Example URLs:")
    print("- http://192.168.1.100:8080/video")
    print("- http://192.168.0.105:8080/videofeed")
    print("- http://10.0.0.50:8080/video")
    
    # Get IP camera URL from user
    ip_url = input("\nEnter your IP camera URL (or press Enter for demo): ").strip()
    
    if not ip_url:
        # Demo with common URLs
        demo_urls = [
            "http://192.168.1.100:8080/video",
            "http://192.168.0.105:8080/video",
            "http://10.0.0.50:8080/video"
        ]
        
        print("\nTrying demo URLs...")
        for url in demo_urls:
            print(f"\nTrying: {url}")
            try:
                cap = cv2.VideoCapture(url)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    print(f"✓ {url} - Connection successful!")
                    test_ip_camera(url)
                    break
                else:
                    print(f"✗ {url} - No response")
            except:
                print(f"✗ {url} - Connection failed")
        else:
            print("\n⚠ No demo URLs worked. Please check your IP Webcam setup.")
    else:
        test_ip_camera(ip_url)


if __name__ == "__main__":
    main()