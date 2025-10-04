import cv2
import sys
from pathlib import Path
import time
sys.path.append(str(Path(__file__).parent.parent / "src"))
from utils.camera_utils import (
    CameraManager, list_cameras, get_camera_resolutions, 
    test_camera_resolution, auto_adjust_camera_settings
)
def main():
    print("Camera Testing Example")
    cameras = list_cameras()
    print(f"Available cameras: {cameras}")
    if not cameras:
        print("No cameras found!")
        return
    camera_id = cameras[0]
    print(f"\nTesting camera {camera_id}")
    print("Getting supported resolutions...")
    resolutions = get_camera_resolutions(camera_id)
    print(f"Supported resolutions: {resolutions}")
    camera_manager = CameraManager(camera_id=camera_id)
    try:
        info = camera_manager.get_camera_info()
        print(f"\nCamera info: {info}")
        print("\nAuto-adjusting camera settings...")
        settings = auto_adjust_camera_settings(camera_manager)
        print(f"Applied settings: {settings}")
        test_resolutions = [(320, 240), (640, 480), (1280, 720)]
        for width, height in test_resolutions:
            print(f"\nTesting resolution {width}x{height}")
            if test_camera_resolution(camera_id, width, height):
                print(f"  ✓ Supported")
                camera_manager.set_resolution(width, height)
                actual_res = camera_manager.get_actual_resolution()
                print(f"  Actual resolution: {actual_res}")
                print("  Capturing test frames (5 seconds)...")
                start_time = time.time()
                frame_count = 0
                while time.time() - start_time < 5:
                    frame = camera_manager.get_frame()
                    if frame is not None:
                        frame_count += 1
                        cv2.imshow(f'Camera Test - {width}x{height}', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                fps = frame_count / 5.0
                print(f"  Average FPS: {fps:.1f}")
            else:
                print(f"  ✗ Not supported")
        print("\nEntering interactive mode. Press 'q' to quit, 'i' for info")
        camera_manager.set_resolution(640, 480)
        while True:
            frame = camera_manager.get_frame()
            if frame is not None:
                info_text = f"Camera {camera_id} - {camera_manager.get_actual_resolution()}"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                fps_text = f"FPS: {camera_manager.fps_counter.get_fps():.1f}"
                cv2.putText(frame, fps_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Camera Test - Interactive', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('i'):
                info = camera_manager.get_camera_info()
                print(f"\nCurrent camera info:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
    finally:
        camera_manager.release()
        cv2.destroyAllWindows()
        print("Camera testing completed")
if __name__ == "__main__":
    main()