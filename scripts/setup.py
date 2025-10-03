"""
Installation and Setup Script for Minecraft Hand Gesture Controller
"""

import subprocess
import sys
import os
from pathlib import Path
import platform

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def setup_directories():
    """Create necessary directories."""
    print("Setting up directories...")
    
    directories = [
        "models",
        "models/gestures",
        "debug_output",
        "logs",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {directory}")
    
    print("✓ Directories setup completed")

def check_camera():
    """Check if camera is available."""
    print("Checking camera availability...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✓ Camera is working")
                cap.release()
                return True
            else:
                print("✗ Camera found but cannot capture frames")
        else:
            print("✗ No camera found")
        cap.release()
    except ImportError:
        print("✗ OpenCV not installed")
    except Exception as e:
        print(f"✗ Camera check failed: {e}")
    
    return False

def check_system_requirements():
    """Check system requirements."""
    print("Checking system requirements...")
    
    system = platform.system()
    print(f"Operating System: {system}")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"Total RAM: {memory_gb:.1f} GB")
        
        if memory_gb < 4:
            print("⚠ Warning: Less than 4GB RAM detected. Performance may be limited.")
        else:
            print("✓ Sufficient memory available")
    except ImportError:
        print("Cannot check memory (psutil not installed)")
    
    # Check disk space
    try:
        disk_usage = psutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        print(f"Free disk space: {free_gb:.1f} GB")
        
        if free_gb < 2:
            print("⚠ Warning: Less than 2GB free space. Consider freeing up space.")
        else:
            print("✓ Sufficient disk space available")
    except:
        print("Cannot check disk space")

def create_shortcuts():
    """Create shortcuts for common operations."""
    print("Creating shortcuts...")
    
    # Create batch files for Windows
    if platform.system() == "Windows":
        shortcuts = {
            "start_controller.bat": "python src/main.py",
            "start_debug.bat": "python src/main.py --debug --show-landmarks",
            "calibrate_gestures.bat": "python examples/calibration_tool.py",
            "test_camera.bat": "python examples/camera_test.py"
        }
        
        for filename, command in shortcuts.items():
            with open(filename, 'w') as f:
                f.write(f"@echo off\n{command}\npause\n")
            print(f"  Created: {filename}")
    
    # Create shell scripts for Unix-like systems
    else:
        shortcuts = {
            "start_controller.sh": "python src/main.py",
            "start_debug.sh": "python src/main.py --debug --show-landmarks",
            "calibrate_gestures.sh": "python examples/calibration_tool.py",
            "test_camera.sh": "python examples/camera_test.py"
        }
        
        for filename, command in shortcuts.items():
            with open(filename, 'w') as f:
                f.write(f"#!/bin/bash\n{command}\n")
            os.chmod(filename, 0o755)
            print(f"  Created: {filename}")
    
    print("✓ Shortcuts created")

def main():
    """Main setup function."""
    print("Minecraft Hand Gesture Controller - Setup Script")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Install requirements
    if not install_requirements():
        print("Setup failed due to installation errors")
        sys.exit(1)
    
    # Check system requirements
    check_system_requirements()
    
    # Check camera
    camera_ok = check_camera()
    if not camera_ok:
        print("⚠ Warning: Camera issues detected. Please check your camera setup.")
    
    # Create shortcuts
    create_shortcuts()
    
    print("\n" + "=" * 60)
    print("Setup completed!")
    
    if camera_ok:
        print("\n✓ Everything looks good! You can now:")
        print("1. Run the controller: python src/main.py")
        print("2. Start with debug mode: python src/main.py --debug")
        print("3. Calibrate gestures: python examples/calibration_tool.py")
    else:
        print("\n⚠ Please fix camera issues before running the controller")
    
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()