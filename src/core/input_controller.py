"""
Input Controller Module
Handles keyboard and mouse input simulation using pynput
"""

import time
import yaml
import logging
from typing import Dict, List, Set, Optional
from pynput import keyboard, mouse
from pynput.keyboard import Key
from threading import Timer, Lock
import cv2


class InputController:
    """Controls keyboard and mouse inputs for Minecraft."""
    
    def __init__(self, config_path: str = "config/minecraft_controls.yaml"):
        """
        Initialize the input controller.
        
        Args:
            config_path: Path to Minecraft controls configuration
        """
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        self.config_path = config_path
        self.controls_config = self._load_config()
        
        # Initialize pynput controllers
        self.keyboard_controller = keyboard.Controller()
        self.mouse_controller = mouse.Controller()
        
        # Track pressed keys and active gestures
        self.pressed_keys: Set[str] = set()
        self.active_gestures: Dict[str, Dict] = {}
        self.gesture_timers: Dict[str, Timer] = {}
        
        # Thread safety
        self.lock = Lock()
        
        # Gesture state tracking
        self.last_gesture_time = {}
        self.gesture_cooldown = 0.5  # Balanced cooldown for responsiveness
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("InputController initialized")
    
    def _load_config(self) -> Dict:
        """Load Minecraft controls configuration."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config.get('minecraft_controls', {})
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {self.config_path}. Using default controls.")
            return self._get_default_controls()
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing config file: {e}")
            return self._get_default_controls()
    
    def _get_default_controls(self) -> Dict:
        """Return default Minecraft control mappings."""
        return {
            'move_forward': {'key': 'w', 'type': 'hold'},
            'move_backward': {'key': 's', 'type': 'hold'},
            'move_left': {'key': 'a', 'type': 'hold'},
            'move_right': {'key': 'd', 'type': 'hold'},
            'jump': {'key': 'space', 'type': 'tap'},
            'crouch': {'key': 'shift', 'type': 'hold'},
            'sprint': {'key': 'ctrl', 'type': 'hold'},
            'inventory': {'key': 'e', 'type': 'tap'},
            'use_item': {'key': 'right_click', 'type': 'tap'},
            'attack': {'key': 'left_click', 'type': 'tap'},
            'drop_item': {'key': 'q', 'type': 'tap'},
            'chat': {'key': 't', 'type': 'tap'},
            'sneak': {'key': 'shift', 'type': 'hold'},
            'stop_all': {'action': 'release_all'},
            'turn_left': {'action': 'mouse_move', 'direction': 'left', 'amount': 50},
            'turn_right': {'action': 'mouse_move', 'direction': 'right', 'amount': 50},
            'look_up': {'action': 'mouse_move', 'direction': 'up', 'amount': 30},
            'look_down': {'action': 'mouse_move', 'direction': 'down', 'amount': 30}
        }
    
    def execute_gesture(self, gesture: Dict):
        """
        Execute the action associated with a gesture.
        
        Args:
            gesture: Gesture dictionary from GestureRecognizer
        """
        if not gesture or 'action' not in gesture:
            return
        
        action_name = gesture['action']
        gesture_name = gesture['name']
        current_time = time.time()
        
        # Check cooldown
        if gesture_name in self.last_gesture_time:
            if current_time - self.last_gesture_time[gesture_name] < self.gesture_cooldown:
                return
        
        self.last_gesture_time[gesture_name] = current_time
        
        # Get action configuration
        action_config = self.controls_config.get(action_name)
        if not action_config:
            self.logger.warning(f"Unknown action: {action_name}")
            return
        
        with self.lock:
            self._execute_action(action_name, action_config, gesture)
    
    def _execute_action(self, action_name: str, action_config: Dict, gesture: Dict):
        """Execute a specific action."""
        try:
            if 'key' in action_config:
                self._handle_key_action(action_name, action_config, gesture)
            elif action_config.get('action') == 'release_all':
                self._release_all_keys()
            elif action_config.get('action') == 'mouse_move':
                self._handle_mouse_move(action_config)
            else:
                self.logger.warning(f"Unknown action type for {action_name}")
                
        except Exception as e:
            self.logger.error(f"Error executing action {action_name}: {e}")
    
    def _handle_key_action(self, action_name: str, action_config: Dict, gesture: Dict):
        """Handle keyboard actions."""
        key_name = action_config['key']
        action_type = action_config.get('type', 'tap')
        
        # Convert key name to pynput key
        key = self._get_key_object(key_name)
        if key is None:
            self.logger.warning(f"Unknown key: {key_name}")
            return
        
        if action_type == 'tap':
            self._tap_key(key)
        elif action_type == 'hold':
            self._hold_key(action_name, key)
        elif action_type == 'toggle':
            self._toggle_key(action_name, key)
    
    def _get_key_object(self, key_name: str):
        """Convert key name to pynput Key object."""
        # Special keys
        special_keys = {
            'space': Key.space,
            'enter': Key.enter,
            'shift': Key.shift,
            'ctrl': Key.ctrl,
            'alt': Key.alt,
            'tab': Key.tab,
            'esc': Key.esc,
            'backspace': Key.backspace,
            'delete': Key.delete,
            'up': Key.up,
            'down': Key.down,
            'left': Key.left,
            'right': Key.right,
            'f1': Key.f1, 'f2': Key.f2, 'f3': Key.f3, 'f4': Key.f4,
            'f5': Key.f5, 'f6': Key.f6, 'f7': Key.f7, 'f8': Key.f8,
            'f9': Key.f9, 'f10': Key.f10, 'f11': Key.f11, 'f12': Key.f12
        }
        
        # Mouse buttons
        if key_name == 'left_click':
            return 'left_click'
        elif key_name == 'right_click':
            return 'right_click'
        elif key_name == 'middle_click':
            return 'middle_click'
        
        # Check special keys first
        if key_name.lower() in special_keys:
            return special_keys[key_name.lower()]
        
        # Regular character keys
        if len(key_name) == 1:
            return key_name.lower()
        
        return None
    
    def _tap_key(self, key):
        """Perform a single key tap."""
        if isinstance(key, str) and key.endswith('_click'):
            self._click_mouse(key)
        else:
            self.keyboard_controller.press(key)
            time.sleep(0.05)  # Brief press duration
            self.keyboard_controller.release(key)
        
        self.logger.debug(f"Tapped key: {key}")
    
    def _hold_key(self, action_name: str, key):
        """Hold a key down (release previous if different)."""
        # Release other movement keys if this is a movement action
        if action_name in ['move_forward', 'move_backward', 'move_left', 'move_right']:
            movement_actions = ['move_forward', 'move_backward', 'move_left', 'move_right']
            for action in movement_actions:
                if action != action_name and action in self.active_gestures:
                    self._release_gesture(action)
        
        # Check if already holding this key
        if action_name in self.active_gestures:
            return
        
        # Press and hold the key
        if isinstance(key, str) and key.endswith('_click'):
            # Mouse buttons can't be held in the same way
            self._click_mouse(key)
        else:
            self.keyboard_controller.press(key)
            self.pressed_keys.add(str(key))
        
        # Track active gesture
        self.active_gestures[action_name] = {
            'key': key,
            'start_time': time.time()
        }
        
        # Set release timer (in case gesture stops being detected)
        timer = Timer(2.0, self._release_gesture, args=[action_name])
        timer.start()
        self.gesture_timers[action_name] = timer
        
        self.logger.debug(f"Holding key: {key} for action: {action_name}")
    
    def _toggle_key(self, action_name: str, key):
        """Toggle key state."""
        if action_name in self.active_gestures:
            self._release_gesture(action_name)
        else:
            self._hold_key(action_name, key)
    
    def _click_mouse(self, button_name: str):
        """Perform mouse click."""
        button_map = {
            'left_click': mouse.Button.left,
            'right_click': mouse.Button.right,
            'middle_click': mouse.Button.middle
        }
        
        button = button_map.get(button_name)
        if button:
            self.mouse_controller.click(button)
            self.logger.debug(f"Clicked mouse button: {button_name}")
    
    def _handle_mouse_move(self, action_config: Dict):
        """Handle mouse movement."""
        direction = action_config.get('direction', 'right')
        amount = action_config.get('amount', 50)
        
        dx, dy = 0, 0
        
        if direction == 'left':
            dx = -amount
        elif direction == 'right':
            dx = amount
        elif direction == 'up':
            dy = -amount
        elif direction == 'down':
            dy = amount
        
        self.mouse_controller.move(dx, dy)
        self.logger.debug(f"Moved mouse: direction={direction}, amount={amount}")
    
    def _release_gesture(self, action_name: str):
        """Release a held gesture/key."""
        if action_name not in self.active_gestures:
            return
        
        gesture_data = self.active_gestures[action_name]
        key = gesture_data['key']
        
        # Release the key
        if not (isinstance(key, str) and key.endswith('_click')):
            try:
                self.keyboard_controller.release(key)
                self.pressed_keys.discard(str(key))
            except Exception as e:
                self.logger.warning(f"Error releasing key {key}: {e}")
        
        # Clean up tracking
        del self.active_gestures[action_name]
        
        # Cancel timer
        if action_name in self.gesture_timers:
            self.gesture_timers[action_name].cancel()
            del self.gesture_timers[action_name]
        
        self.logger.debug(f"Released key: {key} for action: {action_name}")
    
    def _release_all_keys(self):
        """Release all currently pressed keys."""
        self.logger.info("Releasing all keys")
        
        # Release all tracked gestures
        for action_name in list(self.active_gestures.keys()):
            self._release_gesture(action_name)
        
        # Force release any remaining keys
        for key_str in list(self.pressed_keys):
            try:
                # Try to convert back to key object
                if len(key_str) == 1:
                    key = key_str
                else:
                    # This is a simplification - in practice you'd need better key mapping
                    continue
                
                self.keyboard_controller.release(key)
            except Exception as e:
                self.logger.warning(f"Error force-releasing key {key_str}: {e}")
        
        self.pressed_keys.clear()
    
    def update_gesture_timers(self, active_gesture_names: List[str]):
        """Update timers for active gestures to prevent auto-release."""
        current_time = time.time()
        
        for action_name in list(self.active_gestures.keys()):
            if action_name not in active_gesture_names:
                # Gesture no longer detected, release after short delay
                if action_name in self.gesture_timers:
                    self.gesture_timers[action_name].cancel()
                
                # Short delay to avoid flickering
                timer = Timer(0.2, self._release_gesture, args=[action_name])
                timer.start()
                self.gesture_timers[action_name] = timer
            else:
                # Gesture still active, reset timer
                if action_name in self.gesture_timers:
                    self.gesture_timers[action_name].cancel()
                
                timer = Timer(2.0, self._release_gesture, args=[action_name])
                timer.start()
                self.gesture_timers[action_name] = timer
    
    def get_active_gestures(self) -> Dict:
        """Get currently active gestures."""
        return self.active_gestures.copy()
    
    def release_all_keys(self):
        """Public method to release all keys (for cleanup)."""
        with self.lock:
            self._release_all_keys()
    
    def is_key_pressed(self, action_name: str) -> bool:
        """Check if a specific action/key is currently active."""
        return action_name in self.active_gestures
    
    def add_custom_control(self, action_name: str, control_config: Dict):
        """Add a custom control mapping."""
        self.controls_config[action_name] = control_config
        self.logger.info(f"Added custom control: {action_name}")
    
    def save_controls_config(self, output_path: str = None):
        """Save current controls configuration to file."""
        if output_path is None:
            output_path = self.config_path
        
        config_data = {'minecraft_controls': self.controls_config}
        
        with open(output_path, 'w') as file:
            yaml.dump(config_data, file, default_flow_style=False, indent=2)
        
        self.logger.info(f"Saved controls configuration to {output_path}")