import yaml
import json
import configparser
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = {}
        self.logger = logging.getLogger(__name__)
        self._load_config()
    def _load_config(self):
        if not self.config_path.exists():
            self.logger.warning(f"Config file not found: {self.config_path}")
            self.config = self._get_default_config()
            return
        try:
            if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                self._load_yaml()
            elif self.config_path.suffix.lower() == '.json':
                self._load_json()
            elif self.config_path.suffix.lower() == '.ini':
                self._load_ini()
            else:
                self.logger.error(f"Unsupported config file format: {self.config_path.suffix}")
                self.config = self._get_default_config()
        except Exception as e:
            self.logger.error(f"Error loading config file {self.config_path}: {e}")
            self.config = self._get_default_config()
    def _load_yaml(self):
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file) or {}
        self.logger.info(f"Loaded YAML config from {self.config_path}")
    def _load_json(self):
        with open(self.config_path, 'r') as file:
            self.config = json.load(file)
        self.logger.info(f"Loaded JSON config from {self.config_path}")
    def _load_ini(self):
        parser = configparser.ConfigParser()
        parser.read(self.config_path)
        self.config = {}
        for section_name in parser.sections():
            self.config[section_name] = dict(parser[section_name])
        self.logger.info(f"Loaded INI config from {self.config_path}")
    def _get_default_config(self) -> Dict:
        return {
            'camera': {
                'id': 0,
                'resolution': [640, 480],
                'fps': 30
            },
            'hand_detection': {
                'confidence': 0.7,
                'max_hands': 2,
                'tracking_confidence': 0.5
            },
            'gestures': {
                'config_path': 'config/gestures.yaml',
                'temporal_filtering': True,
                'cooldown': 0.1
            },
            'minecraft': {
                'config_path': 'config/minecraft_controls.yaml'
            },
            'yolo': {
                'enabled': False,
                'model_path': 'models/yolov8n.pt',
                'confidence': 0.5,
                'iou_threshold': 0.45
            },
            'debug': {
                'enabled': False,
                'show_landmarks': False,
                'show_fps': True,
                'save_frames': False
            },
            'performance': {
                'frame_skip': 1,
                'max_fps': 30,
                'resize_factor': 1.0
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': None
            }
        }
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    def set(self, key: str, value: Any):
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self.logger.debug(f"Set config {key} = {value}")
    def update(self, updates: Dict):
        self._deep_update(self.config, updates)
        self.logger.debug(f"Updated config with {len(updates)} changes")
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    def save(self, output_path: Optional[str] = None):
        if output_path is None:
            output_path = self.config_path
        else:
            output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if output_path.suffix.lower() in ['.yaml', '.yml']:
                self._save_yaml(output_path)
            elif output_path.suffix.lower() == '.json':
                self._save_json(output_path)
            elif output_path.suffix.lower() == '.ini':
                self._save_ini(output_path)
            else:
                self._save_yaml(output_path)
        except Exception as e:
            self.logger.error(f"Error saving config to {output_path}: {e}")
    def _save_yaml(self, output_path: Path):
        with open(output_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
        self.logger.info(f"Saved YAML config to {output_path}")
    def _save_json(self, output_path: Path):
        with open(output_path, 'w') as file:
            json.dump(self.config, file, indent=2)
        self.logger.info(f"Saved JSON config to {output_path}")
    def _save_ini(self, output_path: Path):
        parser = configparser.ConfigParser()
        for section_name, section_data in self.config.items():
            if isinstance(section_data, dict):
                parser[section_name] = {}
                for key, value in section_data.items():
                    parser[section_name][key] = str(value)
        with open(output_path, 'w') as file:
            parser.write(file)
        self.logger.info(f"Saved INI config to {output_path}")
    def validate(self, schema: Dict) -> bool:
        try:
            self._validate_recursive(self.config, schema)
            return True
        except Exception as e:
            self.logger.error(f"Config validation failed: {e}")
            return False
    def _validate_recursive(self, config: Dict, schema: Dict):
        for key, expected_type in schema.items():
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
            value = config[key]
            if isinstance(expected_type, dict):
                if not isinstance(value, dict):
                    raise TypeError(f"Expected dict for {key}, got {type(value)}")
                self._validate_recursive(value, expected_type)
            elif isinstance(expected_type, type):
                if not isinstance(value, expected_type):
                    raise TypeError(f"Expected {expected_type} for {key}, got {type(value)}")
            elif isinstance(expected_type, list):
                if not any(isinstance(value, t) for t in expected_type):
                    raise TypeError(f"Expected one of {expected_type} for {key}, got {type(value)}")
    def get_section(self, section: str) -> Dict:
        return self.config.get(section, {})
    def list_keys(self, section: str = None) -> list:
        if section:
            return list(self.config.get(section, {}).keys())
        else:
            return self._get_all_keys(self.config)
    def _get_all_keys(self, config_dict: Dict, prefix: str = '') -> list:
        keys = []
        for key, value in config_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.append(full_key)
            if isinstance(value, dict):
                keys.extend(self._get_all_keys(value, full_key))
        return keys
    def reload(self):
        self._load_config()
        self.logger.info("Configuration reloaded")
    def merge_config_file(self, other_config_path: str):
        other_loader = ConfigLoader(other_config_path)
        self.update(other_loader.config)
        self.logger.info(f"Merged config from {other_config_path}")
    def export_defaults(self, output_path: str):
        default_config = self._get_default_config()
        with open(output_path, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False, indent=2)
        self.logger.info(f"Exported default config to {output_path}")
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    def __setitem__(self, key: str, value: Any):
        self.set(key, value)
    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None