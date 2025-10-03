"""
Minecraft Hand Gesture Controller - Main application entry point
"""

import cv2
import argparse
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from core.hand_detector import HandDetector
from core.gesture_recognizer import GestureRecognizer
from core.input_controller import InputController
from core.yolo_detector import YOLODetector
from utils.config_loader import ConfigLoader
from utils.camera_utils import CameraManager
from utils.visualization import Visualizer


class MinecraftGestureController:
    """Main controller class for the Minecraft gesture control system."""
    
    def __init__(self, config_path: str = 'config/settings.yaml', ip_camera_url: str = None):
        """Initialize the Minecraft Gesture Controller.
        
        Args:
            config_path: Path to configuration file
            ip_camera_url: Optional IP camera URL override
        """
        self.config = ConfigLoader(config_path)
        self.setup_logging()
        
        # Initialize components
        camera_config = {
            'camera_id': self.config.get('camera.id', 0),
            'resolution': self.config.get('camera.resolution', (640, 480))
        }
        
        # Add IP camera URL if specified (command line override takes precedence)
        if ip_camera_url:
            camera_config['ip_camera_url'] = ip_camera_url
        else:
            config_ip_url = self.config.get('camera.ip_url', None)
            if config_ip_url:
                camera_config['ip_camera_url'] = config_ip_url

        self.camera_manager = CameraManager(**camera_config)
        
        self.hand_detector = HandDetector(
            confidence=self.config.get('hand_detection.confidence', 0.7),
            max_hands=self.config.get('hand_detection.max_hands', 2)
        )
        
        self.gesture_recognizer = GestureRecognizer(
            config_path=self.config.get('gestures.config_path', 'config/gestures.yaml')
        )
        
        self.input_controller = InputController(
            config_path=self.config.get('controls.config_path', 'config/controls.yaml')
        )
        
        # Initialize optional YOLO detector
        if self.config.get('yolo.enabled', False):
            self.yolo_detector = YOLODetector(
                model_path=self.config.get('yolo.model_path', 'models/yolo11n.pt'),
                confidence=self.config.get('yolo.confidence', 0.5)
            )
        else:
            self.yolo_detector = None
        
        self.visualizer = Visualizer()
        
        # Runtime flags
        self.debug_mode = self.config.get('debug.enabled', False)
        self.show_landmarks = self.config.get('debug.show_landmarks', True)
        self.running = False
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("MinecraftGestureController initialized")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config.get('logging.level', 'INFO')
        log_format = self.config.get('logging.format', 
                                   '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.config.get('logging.file', 'logs/app.log'))
            ]
        )
    
    def run(self):
        """Main application loop."""
        self.logger.info("Starting Minecraft Gesture Controller")
        self.running = True
        
        try:
            while self.running:
                # Capture frame
                frame = self.camera_manager.get_frame()
                if frame is None:
                    self.logger.warning("Failed to capture frame")
                    continue
                
                # Detect hands
                hand_results = self.hand_detector.detect(frame)
                
                # Optional: YOLO object detection
                objects = []
                if self.yolo_detector:
                    objects = self.yolo_detector.detect(frame)
                
                # Recognize gestures
                gestures = self.gesture_recognizer.recognize(hand_results)
                
                # Process input commands
                for gesture in gestures:
                    self.input_controller.process_gesture(gesture)
                
                # Visualization in debug mode
                if self.debug_mode:
                    vis_frame = self.visualizer.draw_hands(
                        frame, hand_results, 
                        show_landmarks=self.show_landmarks
                    )
                    
                    if objects:
                        vis_frame = self.visualizer.draw_objects(vis_frame, objects)
                    
                    vis_frame = self.visualizer.draw_gestures(vis_frame, gestures)
                    
                    cv2.imshow('Minecraft Gesture Controller - Debug', vis_frame)
                    
                    # Exit on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Update FPS counter
                self.camera_manager.fps_counter.update()
                
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Runtime error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up...")
        self.running = False
        
        if hasattr(self, 'camera_manager'):
            self.camera_manager.release()
        if hasattr(self, 'hand_detector'):
            self.hand_detector.release()
        if hasattr(self, 'yolo_detector') and self.yolo_detector:
            self.yolo_detector.release()
        
        # Release all keys
        self.input_controller.release_all_keys()
        cv2.destroyAllWindows()
        self.logger.info("Cleanup completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Minecraft Hand Gesture Controller')
    parser.add_argument('--config', default='config/settings.yaml',
                       help='Path to configuration file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with visualization')
    parser.add_argument('--show-landmarks', action='store_true',
                       help='Show hand landmarks in debug mode')
    parser.add_argument('--camera-id', type=int, default=None,
                       help='Camera device ID')
    parser.add_argument('--ip-camera', type=str, default=None,
                       help='IP camera URL (e.g., http://192.168.1.100:8080/video)')
    
    args = parser.parse_args()
    
    # Create controller with optional IP camera URL
    if args.ip_camera:
        controller = MinecraftGestureController(args.config, ip_camera_url=args.ip_camera)
    else:
        controller = MinecraftGestureController(args.config)
    
    # Apply command line overrides
    if args.debug:
        controller.debug_mode = True
    if args.show_landmarks:
        controller.show_landmarks = True
    if args.camera_id is not None:
        controller.config.set('camera.id', args.camera_id)
        # Reinitialize camera with new camera ID
        camera_config = {
            'camera_id': args.camera_id,
            'resolution': controller.config.get('camera.resolution', (640, 480))
        }
        controller.camera_manager = CameraManager(**camera_config)
    
    # Run the controller
    controller.run()


if __name__ == "__main__":
    main()