"""
YOLO Detection Module
Integrates YOLO object detection for enhanced hand and object recognition
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("Ultralytics YOLO not available. Install with: pip install ultralytics")


class YOLODetector:
    """YOLO-based object detection for enhanced gesture recognition."""
    
    def __init__(self, model_path: str = "models/yolov8n.pt"):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model file
        """
        self.model_path = model_path
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        if not YOLO_AVAILABLE:
            self.logger.error("YOLO not available. Please install ultralytics.")
            return
        
        self._load_model()
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        self.max_detections = 100
        
        # Classes of interest for gesture control
        self.target_classes = {
            'person': 0,
            'cell phone': 67,
            'laptop': 63,
            'mouse': 64,
            'remote': 65,
            'keyboard': 66,
            'book': 73,
            'bottle': 39,
            'cup': 41,
            'fork': 42,
            'knife': 43,
            'spoon': 44
        }
        
    def _load_model(self):
        """Load YOLO model."""
        try:
            # Check if model file exists
            if not Path(self.model_path).exists():
                self.logger.info(f"Model file not found at {self.model_path}. Downloading...")
                # YOLO will automatically download if path doesn't exist
            
            self.model = YOLO(self.model_path)
            self.logger.info(f"YOLO model loaded: {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
    
    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect objects in the frame using YOLO.
        
        Args:
            frame: Input image frame
            
        Returns:
            Detection results dictionary
        """
        if self.model is None:
            return None
        
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )
            
            # Process results
            detections = self._process_results(results[0], frame.shape)
            
            return {
                'detections': detections,
                'frame_shape': frame.shape,
                'detection_time': cv2.getTickCount()
            }
            
        except Exception as e:
            self.logger.error(f"YOLO detection failed: {e}")
            return None
    
    def _process_results(self, result, frame_shape: Tuple) -> List[Dict]:
        """Process YOLO detection results."""
        detections = []
        
        if result.boxes is None:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        height, width = frame_shape[:2]
        
        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box
            
            # Get class name
            class_name = self.model.names.get(class_id, f"class_{class_id}")
            
            # Calculate center and dimensions
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            bbox_width = int(x2 - x1)
            bbox_height = int(y2 - y1)
            
            # Calculate relative coordinates
            rel_x1 = x1 / width
            rel_y1 = y1 / height
            rel_x2 = x2 / width
            rel_y2 = y2 / height
            
            detection = {
                'class_id': int(class_id),
                'class_name': class_name,
                'confidence': float(conf),
                'bbox': {
                    'x1': int(x1), 'y1': int(y1),
                    'x2': int(x2), 'y2': int(y2),
                    'width': bbox_width,
                    'height': bbox_height,
                    'center': (center_x, center_y)
                },
                'bbox_normalized': {
                    'x1': rel_x1, 'y1': rel_y1,
                    'x2': rel_x2, 'y2': rel_y2
                },
                'area': bbox_width * bbox_height,
                'aspect_ratio': bbox_width / bbox_height if bbox_height > 0 else 0
            }
            
            detections.append(detection)
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def filter_detections_by_class(self, detections: List[Dict], 
                                 target_classes: List[str]) -> List[Dict]:
        """Filter detections by class names."""
        if not detections:
            return []
        
        filtered = []
        for detection in detections:
            if detection['class_name'] in target_classes:
                filtered.append(detection)
        
        return filtered
    
    def get_person_detections(self, detections: List[Dict]) -> List[Dict]:
        """Get all person detections."""
        return self.filter_detections_by_class(detections, ['person'])
    
    def find_largest_detection(self, detections: List[Dict], 
                             class_filter: Optional[List[str]] = None) -> Optional[Dict]:
        """Find the largest detection, optionally filtered by class."""
        if class_filter:
            detections = self.filter_detections_by_class(detections, class_filter)
        
        if not detections:
            return None
        
        return max(detections, key=lambda x: x['area'])
    
    def find_closest_detection(self, detections: List[Dict], 
                             point: Tuple[int, int],
                             class_filter: Optional[List[str]] = None) -> Optional[Dict]:
        """Find detection closest to a given point."""
        if class_filter:
            detections = self.filter_detections_by_class(detections, class_filter)
        
        if not detections:
            return None
        
        px, py = point
        min_distance = float('inf')
        closest_detection = None
        
        for detection in detections:
            cx, cy = detection['bbox']['center']
            distance = np.sqrt((px - cx)**2 + (py - cy)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_detection = detection
        
        return closest_detection
    
    def detect_hands_in_person_bbox(self, frame: np.ndarray, 
                                  person_bbox: Dict) -> Optional[np.ndarray]:
        """
        Extract person region for more focused hand detection.
        
        Args:
            frame: Input frame
            person_bbox: Person bounding box from YOLO
            
        Returns:
            Cropped frame containing person region
        """
        bbox = person_bbox['bbox']
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        
        # Add padding
        padding = 20
        height, width = frame.shape[:2]
        
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        
        # Extract region
        person_region = frame[y1:y2, x1:x2]
        
        return person_region if person_region.size > 0 else None
    
    def enhance_hand_detection(self, hand_results: Dict, 
                             yolo_results: Optional[Dict]) -> Dict:
        """
        Enhance hand detection results using YOLO context.
        
        Args:
            hand_results: Results from MediaPipe hand detection
            yolo_results: Results from YOLO detection
            
        Returns:
            Enhanced hand detection results
        """
        if not yolo_results or not yolo_results.get('detections'):
            return hand_results
        
        enhanced_results = hand_results.copy()
        
        # Find person detections
        person_detections = self.get_person_detections(yolo_results['detections'])
        
        if not person_detections or not hand_results.get('hands'):
            return enhanced_results
        
        # Enhance each hand with context
        for hand_data in enhanced_results['hands']:
            hand_bbox = hand_data.get('bounding_box', {})
            if not hand_bbox:
                continue
            
            hand_center = hand_bbox.get('center', (0, 0))
            
            # Find closest person
            closest_person = self.find_closest_detection(
                person_detections, hand_center
            )
            
            if closest_person:
                # Add context information
                hand_data['context'] = {
                    'person_bbox': closest_person['bbox'],
                    'person_confidence': closest_person['confidence'],
                    'relative_position': self._calculate_relative_position(
                        hand_bbox, closest_person['bbox']
                    )
                }
        
        return enhanced_results
    
    def _calculate_relative_position(self, hand_bbox: Dict, 
                                   person_bbox: Dict) -> Dict:
        """Calculate hand position relative to person."""
        hand_center = hand_bbox['center']
        person_center = person_bbox['center']
        
        # Relative position
        rel_x = (hand_center[0] - person_center[0]) / person_bbox['width']
        rel_y = (hand_center[1] - person_center[1]) / person_bbox['height']
        
        # Classify position
        position_x = 'center'
        if rel_x < -0.2:
            position_x = 'left'
        elif rel_x > 0.2:
            position_x = 'right'
        
        position_y = 'center'
        if rel_y < -0.3:
            position_y = 'upper'
        elif rel_y > 0.3:
            position_y = 'lower'
        
        return {
            'relative_x': rel_x,
            'relative_y': rel_y,
            'position_x': position_x,
            'position_y': position_y,
            'position': f"{position_y}_{position_x}" if position_y != 'center' or position_x != 'center' else 'center'
        }
    
    def set_confidence_threshold(self, threshold: float):
        """Set detection confidence threshold."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        self.logger.info(f"Set confidence threshold to {self.confidence_threshold}")
    
    def set_target_classes(self, classes: List[str]):
        """Set target classes for detection."""
        self.target_classes = {name: idx for idx, name in enumerate(classes)}
        self.logger.info(f"Set target classes: {classes}")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {'available': False}
        
        return {
            'available': True,
            'model_path': self.model_path,
            'model_type': str(type(self.model)),
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'target_classes': list(self.target_classes.keys())
        }
    
    def is_available(self) -> bool:
        """Check if YOLO is available and loaded."""
        return YOLO_AVAILABLE and self.model is not None