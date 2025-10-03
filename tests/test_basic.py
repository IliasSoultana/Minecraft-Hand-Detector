"""
Unit tests for Minecraft Hand Gesture Controller
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.hand_detector import HandDetector
from core.gesture_recognizer import GestureRecognizer
from core.input_controller import InputController
from utils.config_loader import ConfigLoader


class TestHandDetector(unittest.TestCase):
    """Test hand detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hand_detector = HandDetector(confidence=0.7, max_hands=2)
    
    def test_initialization(self):
        """Test hand detector initialization."""
        self.assertIsNotNone(self.hand_detector)
        self.assertEqual(self.hand_detector.confidence, 0.7)
        self.assertEqual(self.hand_detector.max_hands, 2)
    
    def test_detect_with_empty_frame(self):
        """Test detection with empty frame."""
        # Create a black frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        results = self.hand_detector.detect(frame)
        
        self.assertIsInstance(results, dict)
        self.assertIn('hands', results)
        self.assertIn('frame_shape', results)
        self.assertEqual(len(results['hands']), 0)
    
    def tearDown(self):
        """Clean up after tests."""
        self.hand_detector.release()


class TestGestureRecognizer(unittest.TestCase):
    """Test gesture recognition functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.gesture_recognizer = GestureRecognizer()
    
    def test_initialization(self):
        """Test gesture recognizer initialization."""
        self.assertIsNotNone(self.gesture_recognizer)
        self.assertIsInstance(self.gesture_recognizer.gestures_config, dict)
    
    def test_recognize_empty_results(self):
        """Test recognition with empty hand results."""
        hand_results = {'hands': []}
        
        gestures = self.gesture_recognizer.recognize(hand_results)
        
        self.assertIsInstance(gestures, list)
        self.assertEqual(len(gestures), 0)
    
    def test_gesture_matching(self):
        """Test gesture matching logic."""
        # Create mock hand pose
        pose = {
            'fingers_up': [1, 1, 1, 1, 1],
            'gesture_confidence': 0.8,
            'openness': 0.9
        }
        
        # Test open palm gesture
        gesture_config = {
            'conditions': {
                'fingers_up': [1, 1, 1, 1, 1],
                'min_confidence': 0.7,
                'openness_threshold': 0.7
            }
        }
        
        matches = self.gesture_recognizer._matches_gesture(pose, gesture_config, {})
        self.assertTrue(matches)


class TestInputController(unittest.TestCase):
    """Test input controller functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use a mock config to avoid file dependencies
        self.input_controller = InputController()
    
    def test_initialization(self):
        """Test input controller initialization."""
        self.assertIsNotNone(self.input_controller)
        self.assertIsInstance(self.input_controller.controls_config, dict)
    
    def test_key_object_conversion(self):
        """Test key name to object conversion."""
        # Test special keys
        space_key = self.input_controller._get_key_object('space')
        self.assertIsNotNone(space_key)
        
        # Test regular character
        w_key = self.input_controller._get_key_object('w')
        self.assertEqual(w_key, 'w')
        
        # Test mouse buttons
        left_click = self.input_controller._get_key_object('left_click')
        self.assertEqual(left_click, 'left_click')


class TestConfigLoader(unittest.TestCase):
    """Test configuration loading functionality."""
    
    def test_default_config(self):
        """Test default configuration loading."""
        # This will use default config since file doesn't exist
        config_loader = ConfigLoader("nonexistent.yaml")
        
        self.assertIsInstance(config_loader.config, dict)
        self.assertIn('camera', config_loader.config)
        self.assertIn('hand_detection', config_loader.config)
    
    def test_config_access(self):
        """Test configuration access methods."""
        config_loader = ConfigLoader("nonexistent.yaml")
        
        # Test dot notation access
        camera_id = config_loader.get('camera.id', 0)
        self.assertEqual(camera_id, 0)
        
        # Test setting values
        config_loader.set('test.value', 42)
        self.assertEqual(config_loader.get('test.value'), 42)
    
    def test_config_validation(self):
        """Test configuration validation."""
        config_loader = ConfigLoader("nonexistent.yaml")
        
        # Simple validation schema
        schema = {
            'camera': {
                'id': int,
                'resolution': list
            }
        }
        
        is_valid = config_loader.validate(schema)
        self.assertTrue(is_valid)


class TestUtilities(unittest.TestCase):
    """Test utility functions."""
    
    def test_camera_utils(self):
        """Test camera utility functions."""
        from utils.camera_utils import list_cameras
        
        # This might fail in testing environment without cameras
        try:
            cameras = list_cameras()
            self.assertIsInstance(cameras, list)
        except:
            self.skipTest("No cameras available in test environment")
    
    def test_visualization(self):
        """Test visualization utilities."""
        from utils.visualization import Visualizer
        
        visualizer = Visualizer()
        self.assertIsNotNone(visualizer)
        self.assertIsInstance(visualizer.colors, dict)


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)