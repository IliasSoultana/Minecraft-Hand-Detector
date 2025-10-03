# Installation Guide

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Operating System** | Windows 10, macOS 10.15, Ubuntu 18.04 | Windows 11, macOS 12+, Ubuntu 20.04+ |
| **Python Version** | 3.11.0 | 3.12+ |
| **CPU** | Intel i3 / AMD Ryzen 3 | Intel i5 / AMD Ryzen 5 |
| **RAM** | 4GB | 8GB+ |
| **Storage** | 2GB free space | 5GB+ free space |
| **Camera** | 480p webcam or IP camera | 1080p webcam/IP camera |
| **Internet** | Required for initial setup | Required for model downloads |

### Software Dependencies

- **Python 3.11+**: Primary runtime environment
- **pip**: Package manager for Python dependencies
- **Git**: Version control (optional, for development)

## Installation Methods

### Method 1: Quick Install (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/IliasSoultana/Minecraft-Hand-Detector.git
cd Minecraft-Hand-Detector

# 2. Run the automated setup script
python scripts/setup.py --auto

# 3. Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# 4. Test installation
python test_system.py
```

### Method 2: Manual Installation

#### Step 1: Environment Setup

```bash
# Create project directory
mkdir minecraft-gesture-controller
cd minecraft-gesture-controller

# Clone repository
git clone https://github.com/IliasSoultana/Minecraft-Hand-Detector.git .

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

#### Step 2: Install Core Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install core requirements
pip install -r requirements.txt

# Verify OpenCV installation
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

#### Step 3: Install Optional Dependencies

```bash
# For MediaPipe support (Python 3.11 only)
pip install mediapipe==0.10.0

# For YOLO object detection
pip install ultralytics

# For development tools
pip install -r requirements-dev.txt
```

#### Step 4: Download Models

```bash
# Download YOLO model (optional)
python scripts/download_models.py

# Or manually download
mkdir models
wget -O models/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

### Method 3: Docker Installation

```bash
# Build Docker image
docker build -t minecraft-gesture-controller .

# Run with camera access
docker run --privileged \
  --device /dev/video0:/dev/video0 \
  -p 8080:8080 \
  minecraft-gesture-controller

# For IP camera (no device access needed)
docker run -p 8080:8080 \
  -e CAMERA_URL="http://192.168.1.100:8080/video" \
  minecraft-gesture-controller
```

## Platform-Specific Installation

### Windows

#### Prerequisites
```powershell
# Install Python 3.11+ from Microsoft Store or python.org
# Install Visual C++ Build Tools (for some dependencies)
# Install Git for Windows (optional)
```

#### Installation
```powershell
# Clone repository
git clone https://github.com/IliasSoultana/Minecraft-Hand-Detector.git
cd Minecraft-Hand-Detector

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python test_system.py
```

#### Common Windows Issues
```powershell
# If pip install fails due to missing compiler
pip install --only-binary=all -r requirements.txt

# If camera access is denied
# Run as administrator or check Windows camera privacy settings

# For DirectShow camera issues
pip install opencv-python-headless
```

### macOS

#### Prerequisites
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Install system dependencies
brew install cmake pkg-config
```

#### Installation
```bash
# Clone repository
git clone https://github.com/IliasSoultana/Minecraft-Hand-Detector.git
cd Minecraft-Hand-Detector

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Grant camera permissions in System Preferences
```

#### macOS-Specific Issues
```bash
# For M1/M2 Macs, install ARM-compatible versions
pip install --upgrade pip
pip install opencv-python --no-cache-dir

# If MediaPipe installation fails
pip install mediapipe --no-deps
pip install protobuf==3.20.*
```

### Linux (Ubuntu/Debian)

#### Prerequisites
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and development tools
sudo apt install python3.11 python3.11-venv python3.11-dev
sudo apt install build-essential cmake pkg-config

# Install system libraries for OpenCV
sudo apt install libopencv-dev python3-opencv
sudo apt install libgtk-3-dev libcairo2-dev libpango1.0-dev
```

#### Installation
```bash
# Clone repository
git clone https://github.com/IliasSoultana/Minecraft-Hand-Detector.git
cd Minecraft-Hand-Detector

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add user to video group for camera access
sudo usermod -a -G video $USER
# Log out and back in for changes to take effect
```

#### Linux-Specific Issues
```bash
# If camera permission denied
sudo chmod 666 /dev/video0

# For headless servers (no display)
export DISPLAY=:0.0
# Or use virtual display
sudo apt install xvfb
xvfb-run -a python src/main.py

# If OpenCV import fails
sudo apt install python3-opencv
pip install opencv-contrib-python
```

## Configuration

### Initial Setup

```bash
# Copy example configurations
cp config/settings.example.yaml config/settings.yaml
cp config/gestures.example.yaml config/gestures.yaml
cp config/controls.example.yaml config/controls.yaml

# Edit configurations as needed
nano config/settings.yaml  # Linux/macOS
notepad config/settings.yaml  # Windows
```

### Camera Setup

#### Local Webcam
```yaml
# config/settings.yaml
camera:
  type: "local"
  id: 0  # Usually 0 for default camera
  resolution: [1920, 1080]
  fps: 30
```

#### IP Camera Setup

1. **Install IP Camera App**
   - Android: "IP Webcam" by Pavel Khlebovich
   - iOS: "iVCam" or similar

2. **Configure Network**
   ```bash
   # Find your phone's IP address
   # Android: Settings > About > Status > IP Address
   # iPhone: Settings > WiFi > (your network) > IP Address
   ```

3. **Update Configuration**
   ```yaml
   # config/settings.yaml
   camera:
     type: "ip"
     url: "http://192.168.1.100:8080/video"
     resolution: [1920, 1080]
     timeout: 5
   ```

### Gesture Calibration

```bash
# Run calibration tool
python scripts/calibrate_gestures.py

# Test individual gestures
python scripts/test_gesture.py --gesture fist
python scripts/test_gesture.py --gesture open_palm
```

## Verification

### System Test
```bash
# Full system test
python test_system.py

# Expected output:
# ✓ Camera connected
# ✓ Hand detector initialized
# ✓ Gesture recognizer ready
# ✓ Input controller active
# ✓ All tests passed!
```

### Component Tests
```bash
# Test camera connectivity
python test_ip_cam.py

# Test hand detection
python quick_test.py

# Test input simulation
python test_keyboard.py
```

### Performance Benchmarks
```bash
# Run performance tests
python scripts/benchmark.py

# Expected metrics:
# FPS: 25-30
# Latency: <50ms
# CPU: <30%
# Memory: <200MB
```

## Troubleshooting

### Common Installation Issues

#### Python Version Conflicts
```bash
# Check Python version
python --version
python3 --version

# Use specific Python version
python3.11 -m venv .venv
```

#### Permission Errors
```bash
# Windows: Run as administrator
# macOS/Linux: Use sudo for system packages only
sudo apt install python3-dev  # System packages
pip install opencv-python     # User packages (no sudo)
```

#### Package Installation Failures
```bash
# Clear pip cache
pip cache purge

# Use alternative index
pip install -i https://pypi.org/simple/ opencv-python

# Install from wheel
pip install --only-binary=all opencv-python
```

#### Camera Access Issues
```bash
# Windows: Check camera privacy settings
# macOS: Grant camera permission in System Preferences
# Linux: Add user to video group
sudo usermod -a -G video $USER
```

### Getting Help

1. **Check logs**: Look in `logs/` directory for error details
2. **GitHub Issues**: Search existing issues or create new one
3. **Documentation**: Check `docs/` for detailed guides
4. **Community**: Join discussions for community support

### Environment Variables

```bash
# Optional environment variables
export OPENCV_LOG_LEVEL=ERROR  # Reduce OpenCV logging
export CUDA_VISIBLE_DEVICES=0  # Specify GPU for YOLO
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Add src to path
```

## Next Steps

After successful installation:

1. **Configure Gestures**: Customize gesture mappings in `config/gestures.yaml`
2. **Test with Minecraft**: Launch Minecraft and test gesture controls
3. **Optimize Performance**: Adjust settings for your hardware
4. **Join Community**: Contribute to the project or get support

## Updating

### Update to Latest Version
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Run migration script if needed
python scripts/migrate_config.py
```

### Version Management
```bash
# Check current version
python -c "from src import __version__; print(__version__)"

# List available versions
git tag -l

# Switch to specific version
git checkout v1.2.0
pip install -r requirements.txt
```