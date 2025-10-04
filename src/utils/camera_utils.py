import cv2
import numpy as np
import logging
import threading
import time
from typing import Optional, Tuple, List
from queue import Queue, Empty
class CameraManager:
    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (640, 480), ip_camera_url: str = None):
        self.camera_id = camera_id
        self.resolution = resolution
        self.ip_camera_url = ip_camera_url
        self.camera = None
        self.logger = logging.getLogger(__name__)
        self.capture_thread = None
        self.frame_queue = Queue(maxsize=5)
        self.running = False
        self.lock = threading.Lock()
        self.frame_count = 0
        self.fps_counter = FPSCounter()
        self.brightness = 0
        self.contrast = 0
        self.saturation = 0
        self.exposure = -1
        self._initialize_camera()
    def _initialize_camera(self):
        try:
            if self.ip_camera_url:
                self.logger.info(f"Connecting to IP camera: {self.ip_camera_url}")
                self.camera = cv2.VideoCapture(self.ip_camera_url)
            else:
                self.logger.info(f"Connecting to local camera: {self.camera_id}")
                self.camera = cv2.VideoCapture(self.camera_id)
            if not self.camera.isOpened():
                if self.ip_camera_url:
                    raise RuntimeError(f"Could not connect to IP camera: {self.ip_camera_url}")
                else:
                    raise RuntimeError(f"Could not open camera {self.camera_id}")
            if not self.ip_camera_url:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.camera.set(cv2.CAP_PROP_FPS, 30)
                self._apply_camera_settings()
            ret, frame = self.camera.read()
            if not ret:
                raise RuntimeError("Failed to capture test frame")
            if self.ip_camera_url:
                self.logger.info(f"IP camera connected successfully")
                self.logger.info(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
            else:
                self.logger.info(f"Camera {self.camera_id} initialized successfully")
                self.logger.info(f"Resolution: {self.get_actual_resolution()}")
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            if self.camera:
                self.camera.release()
                self.camera = None
            raise
    def _apply_camera_settings(self):
        if not self.camera:
            return
        try:
            if self.brightness != 0:
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
            if self.contrast != 0:
                self.camera.set(cv2.CAP_PROP_CONTRAST, self.contrast)
            if self.saturation != 0:
                self.camera.set(cv2.CAP_PROP_SATURATION, self.saturation)
            if self.exposure != -1:
                self.camera.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            self.logger.warning(f"Could not set some camera properties: {e}")
    def start_capture_thread(self):
        if self.capture_thread and self.capture_thread.is_alive():
            return
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.logger.info("Started capture thread")
    def stop_capture_thread(self):
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        self.logger.info("Stopped capture thread")
    def _capture_loop(self):
        while self.running and self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.fps_counter.update()
                self.frame_count += 1
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                try:
                    self.frame_queue.put_nowait(frame)
                except:
                    pass
            else:
                self.logger.warning("Failed to capture frame")
                time.sleep(0.01)
    def get_frame(self) -> Optional[np.ndarray]:
        if not self.camera or not self.camera.isOpened():
            return None
        if self.running and not self.frame_queue.empty():
            try:
                return self.frame_queue.get_nowait()
            except Empty:
                pass
        with self.lock:
            ret, frame = self.camera.read()
            if ret:
                self.fps_counter.update()
                self.frame_count += 1
                return frame
            else:
                self.logger.warning("Failed to capture frame")
                return None
    def get_frame_sync(self) -> Optional[np.ndarray]:
        if not self.camera or not self.camera.isOpened():
            return None
        with self.lock:
            ret, frame = self.camera.read()
            if ret:
                self.fps_counter.update()
                self.frame_count += 1
                return frame
            else:
                return None
    def set_resolution(self, width: int, height: int):
        if not self.camera:
            return False
        old_resolution = self.resolution
        self.resolution = (width, height)
        try:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.logger.info(f"Resolution changed to {actual_width}x{actual_height}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to set resolution: {e}")
            self.resolution = old_resolution
            return False
    def get_actual_resolution(self) -> Tuple[int, int]:
        if not self.camera:
            return (0, 0)
        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    def set_fps(self, fps: int):
        if not self.camera:
            return False
        try:
            self.camera.set(cv2.CAP_PROP_FPS, fps)
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            self.logger.info(f"FPS set to {actual_fps}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to set FPS: {e}")
            return False
    def adjust_brightness(self, value: float):
        self.brightness = max(-1.0, min(1.0, value))
        if self.camera:
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
    def adjust_contrast(self, value: float):
        self.contrast = max(-1.0, min(1.0, value))
        if self.camera:
            self.camera.set(cv2.CAP_PROP_CONTRAST, self.contrast)
    def adjust_exposure(self, value: float):
        self.exposure = value
        if self.camera:
            if value == -1:
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            else:
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
                self.camera.set(cv2.CAP_PROP_EXPOSURE, value)
    def get_camera_info(self) -> dict:
        if not self.camera:
            return {'available': False}
        info = {
            'available': True,
            'type': 'IP Camera' if self.ip_camera_url else 'Local Camera',
            'source': self.ip_camera_url or self.camera_id,
            'frame_count': self.frame_count,
            'current_fps': self.fps_counter.get_fps()
        }
        if not self.ip_camera_url:
            info.update({
                'camera_id': self.camera_id,
                'resolution': self.get_actual_resolution(),
                'fps': self.camera.get(cv2.CAP_PROP_FPS),
                'brightness': self.camera.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': self.camera.get(cv2.CAP_PROP_CONTRAST),
                'saturation': self.camera.get(cv2.CAP_PROP_SATURATION),
                'exposure': self.camera.get(cv2.CAP_PROP_EXPOSURE),
                'auto_exposure': self.camera.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            })
        else:
            ret, frame = self.camera.read()
            if ret:
                info['resolution'] = (frame.shape[1], frame.shape[0])
        return info
    def reinitialize(self):
        self.stop_capture_thread()
        self.release()
        time.sleep(0.5)
        self._initialize_camera()
    def release(self):
        self.stop_capture_thread()
        if self.camera:
            self.camera.release()
            self.camera = None
        self.logger.info("Camera resources released")
    def __del__(self):
        self.release()
class FPSCounter:
    def __init__(self, buffer_size: int = 30):
        self.buffer_size = buffer_size
        self.frame_times = []
        self.last_time = time.time()
    def update(self):
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
        if len(self.frame_times) > self.buffer_size:
            self.frame_times.pop(0)
    def get_fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    def reset(self):
        self.frame_times.clear()
        self.last_time = time.time()
def list_cameras() -> List[int]:
    available_cameras = []
    for camera_id in range(10):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(camera_id)
        cap.release()
    return available_cameras
def test_camera_resolution(camera_id: int, width: int, height: int) -> bool:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return False
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return actual_width == width and actual_height == height
    finally:
        cap.release()
def get_camera_resolutions(camera_id: int) -> List[Tuple[int, int]]:
    common_resolutions = [
        (160, 120), (320, 240), (640, 480), (800, 600),
        (1024, 768), (1280, 720), (1280, 960), (1600, 1200),
        (1920, 1080), (2560, 1440), (3840, 2160)
    ]
    supported_resolutions = []
    for width, height in common_resolutions:
        if test_camera_resolution(camera_id, width, height):
            supported_resolutions.append((width, height))
    return supported_resolutions
def auto_adjust_camera_settings(camera_manager: CameraManager) -> dict:
    if not camera_manager.camera:
        return {}
    settings = {}
    test_frames = []
    for _ in range(10):
        frame = camera_manager.get_frame_sync()
        if frame is not None:
            test_frames.append(frame)
        time.sleep(0.1)
    if not test_frames:
        return settings
    avg_brightness = np.mean([np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) for frame in test_frames])
    if avg_brightness < 80:
        camera_manager.adjust_brightness(0.3)
        settings['brightness'] = 0.3
    elif avg_brightness > 180:
        camera_manager.adjust_brightness(-0.3)
        settings['brightness'] = -0.3
    avg_contrast = np.mean([np.std(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) for frame in test_frames])
    if avg_contrast < 30:
        camera_manager.adjust_contrast(0.2)
        settings['contrast'] = 0.2
    camera_manager.adjust_exposure(-1)
    settings['exposure'] = 'auto'
    return settings