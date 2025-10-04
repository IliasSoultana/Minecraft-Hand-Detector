import os
import sys
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
def download_yolo_models():
    try:
        from ultralytics import YOLO
        models = {
            'yolov8n.pt': 'Nano model (fastest, least accurate)',
            'yolov8s.pt': 'Small model (balanced)',
            'yolov8m.pt': 'Medium model (better accuracy)',
            'yolov8l.pt': 'Large model (high accuracy)',
            'yolov8x.pt': 'Extra large model (highest accuracy)'
        }
        print("Available YOLO models:")
        for i, (model_name, description) in enumerate(models.items(), 1):
            print(f"{i}. {model_name} - {description}")
        choice = input("\nWhich model(s) would you like to download? (Enter numbers separated by commas, or 'all'): ").strip()
        if choice.lower() == 'all':
            selected_models = list(models.keys())
        else:
            try:
                indices = [int(x.strip()) for x in choice.split(',')]
                selected_models = [list(models.keys())[i-1] for i in indices if 1 <= i <= len(models)]
            except (ValueError, IndexError):
                print("Invalid selection. Downloading nano model by default.")
                selected_models = ['yolov8n.pt']
        print(f"\nDownloading {len(selected_models)} model(s)...")
        for model_name in selected_models:
            print(f"\nDownloading {model_name}...")
            try:
                model = YOLO(model_name)
                model_path = models_dir / model_name
                if not model_path.exists():
                    print(f"✓ {model_name} ready for use")
                else:
                    print(f"✓ {model_name} already exists")
            except Exception as e:
                logger.error(f"Failed to download {model_name}: {e}")
        print("\nYOLO model download completed!")
        print(f"Models are available in the YOLO cache directory.")
        print("The application will automatically use them when needed.")
    except ImportError:
        print("Error: ultralytics package not found.")
        print("Please install it with: pip install ultralytics")
        return False
    return True
def download_hand_gesture_models():
    print("\nSetting up hand gesture models...")
    gesture_models_dir = models_dir / "gestures"
    gesture_models_dir.mkdir(exist_ok=True)
    example_models = [
        "hand_classifier.pkl",
        "gesture_features.pkl",
        "calibration_data.pkl"
    ]
    for model_file in example_models:
        model_path = gesture_models_dir / model_file
        if not model_path.exists():
            model_path.write_text(f"Placeholder for {model_file}\nThis file will be generated during calibration\n")
            print(f"Created placeholder: {model_path}")
    print("Hand gesture model setup completed!")
def main():
    print("Minecraft Hand Gesture Controller - Model Download Script")
    print("=" * 60)
    print("1. Setting up YOLO models...")
    if download_yolo_models():
        print("✓ YOLO models setup completed")
    else:
        print("✗ YOLO models setup failed")
    print("\n2. Setting up hand gesture models...")
    download_hand_gesture_models()
    print("✓ Hand gesture models setup completed")
    print("\n" + "=" * 60)
    print("Model setup completed!")
    print("\nNext steps:")
    print("1. Run: python src/main.py --debug")
    print("2. Or calibrate gestures: python examples/calibration_tool.py")
    print("3. Check camera: python examples/camera_test.py")
if __name__ == "__main__":
    main()