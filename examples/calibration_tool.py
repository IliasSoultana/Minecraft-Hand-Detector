import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from gestures.calibration import GestureCalibrator, quick_calibrate_gesture
def main():
    print("Gesture Calibration Example")
    print("Choose calibration mode:")
    print("1. Full interactive calibration")
    print("2. Quick single gesture calibration")
    choice = input("Enter choice (1 or 2): ").strip()
    if choice == "1":
        calibrator = GestureCalibrator()
        calibrator.start_calibration_session(camera_id=0)
    elif choice == "2":
        gesture_name = input("Enter gesture name: ").strip()
        if not gesture_name:
            gesture_name = "custom_gesture"
        samples_count = input("Enter number of samples (default 20): ").strip()
        try:
            samples_count = int(samples_count) if samples_count else 20
        except ValueError:
            samples_count = 20
        config = quick_calibrate_gesture(gesture_name, samples_count)
        if config:
            print(f"\nCalibrated gesture '{gesture_name}':")
            print(f"Finger pattern: {config['conditions']['fingers_up']}")
            print(f"Confidence threshold: {config['conditions']['min_confidence']:.2f}")
            print(f"Openness threshold: {config['conditions']['openness_threshold']:.2f}")
            output_file = f"calibrated_{gesture_name}.yaml"
            import yaml
            with open(output_file, 'w') as f:
                yaml.dump({gesture_name: config}, f, default_flow_style=False, indent=2)
            print(f"Saved to {output_file}")
        else:
            print("Calibration failed or incomplete")
    else:
        print("Invalid choice")
if __name__ == "__main__":
    main()