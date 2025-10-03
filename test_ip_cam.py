import cv2

print("Testing IP camera: http://192.168.178.156:8080/video")

try:
    cap = cv2.VideoCapture('http://192.168.178.156:8080/video')
    ret, frame = cap.read()
    
    if ret:
        print("✓ Connection successful!")
        print(f"Frame shape: {frame.shape}")
        print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("✗ Connection failed - trying alternative URL")
        cap.release()
        
        # Try alternative URL
        cap = cv2.VideoCapture('http://192.168.178.156:8080/videofeed')
        ret, frame = cap.read()
        
        if ret:
            print("✓ Alternative URL successful!")
            print(f"Frame shape: {frame.shape}")
        else:
            print("✗ Both URLs failed")
    
    cap.release()
    
except Exception as e:
    print(f"Error: {e}")