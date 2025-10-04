import cv2
def main():
    print("Camera detection test")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera {i}: Working ({frame.shape[1]}x{frame.shape[0]})")
            else:
                print(f"⚠ Camera {i}: Opened but no frame")
            cap.release()
        else:
            print(f"✗ Camera {i}: Not available")
    print("\nTesting IP camera connectivity...")
    ip_url = "http://192.168.178.156:8080/video"
    cap = cv2.VideoCapture(ip_url)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✓ IP Camera: Working ({frame.shape[1]}x{frame.shape[0]})")
        else:
            print("⚠ IP Camera: Connected but no frame")
        cap.release()
    else:
        print("✗ IP Camera: Not available")
if __name__ == "__main__":
    main()