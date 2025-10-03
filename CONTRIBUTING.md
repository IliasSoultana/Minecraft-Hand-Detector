# 🤝 Contributing to Minecraft Hand Gesture Controller

Thank you for your interest in contributing to the Minecraft Hand Gesture Controller! This guide will help you get started with contributing to this open-source project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Development Guidelines](#development-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

### Our Standards

**Positive behaviors include:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behaviors include:**
- The use of sexualized language or imagery
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Git
- Basic knowledge of computer vision, OpenCV, and Python
- Familiarity with hand gesture recognition concepts

### Areas for Contribution

We welcome contributions in these areas:

1. **🖐️ Hand Detection**
   - Improve detection accuracy
   - Add new detection algorithms
   - Optimize performance

2. **🎯 Gesture Recognition**
   - Add new gesture types
   - Improve classification accuracy
   - Implement machine learning models

3. **🎮 Game Integration**
   - Add support for other games
   - Improve Minecraft controls
   - Create new control schemes

4. **📱 Platform Support**
   - macOS optimizations
   - Linux compatibility
   - Mobile integration

5. **📚 Documentation**
   - Improve existing docs
   - Add tutorials
   - Create video guides

6. **🧪 Testing**
   - Add unit tests
   - Performance benchmarks
   - Integration tests

## 🛠️ Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/yourusername/Minecraft-Hand-Detector.git
cd Minecraft-Hand-Detector

# Add upstream remote
git remote add upstream https://github.com/IliasSoultana/Minecraft-Hand-Detector.git
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -r requirements.txt
```

### 3. Install Pre-commit Hooks

```bash
# Install pre-commit hooks for code quality
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files
```

### 4. Verify Setup

```bash
# Run tests to ensure everything works
python -m pytest tests/

# Run system test
python test_system.py

# Check code style
flake8 src/
black --check src/
```

## 💡 How to Contribute

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

**Bug Report Template:**
```markdown
**Bug Description**
A clear description of what the bug is.

**Steps to Reproduce**
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g. Windows 11]
- Python Version: [e.g. 3.11.0]
- OpenCV Version: [e.g. 4.12.0]
- Camera Type: [e.g. IP Camera, Webcam]

**Additional Context**
Add any other context about the problem here.
```

### 💡 Suggesting Features

Feature requests are welcome! Please include:

**Feature Request Template:**
```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Problem Statement**
What problem does this feature solve?

**Proposed Solution**
How do you envision this feature working?

**Alternatives Considered**
Other solutions you've considered.

**Additional Context**
Any other context or screenshots about the feature request.
```

### 🔧 Code Contributions

#### 1. Choose an Issue

- Look for issues labeled `good first issue` for beginners
- Check `help wanted` issues for areas needing assistance
- Comment on issues you'd like to work on

#### 2. Create a Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number
```

#### 3. Make Changes

Follow our development guidelines (see below) while making changes.

#### 4. Test Your Changes

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python test_system.py

# Test specific functionality
python quick_test.py

# Check code style
flake8 src/
black src/
```

#### 5. Commit Changes

```bash
# Stage changes
git add .

# Commit with meaningful message
git commit -m "feat: add new gesture recognition algorithm

- Implement neural network-based gesture classification
- Improve accuracy by 15% in testing
- Add configuration options for model selection
- Update documentation with new options

Fixes #123"
```

**Commit Message Format:**
```
type(scope): brief description

Detailed description of changes made.
Include motivation and implementation details.

Fixes #issue_number
```

**Commit Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

## 🏗️ Development Guidelines

### Code Style

#### Python Style Guide
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [flake8](https://flake8.pycqa.org/) for linting
- Maximum line length: 88 characters

#### Type Hints
```python
from typing import Dict, List, Optional, Tuple
import numpy as np

def detect_hands(frame: np.ndarray) -> Dict[str, List]:
    """Detect hands in the given frame.
    
    Args:
        frame: Input image frame
        
    Returns:
        Dictionary containing detection results
    """
    pass
```

#### Documentation Style
```python
class HandDetector:
    """Hand detection using MediaPipe and OpenCV.
    
    This class provides hand detection capabilities using MediaPipe
    as the primary method with OpenCV as a fallback.
    
    Attributes:
        confidence: Detection confidence threshold
        max_hands: Maximum number of hands to detect
        
    Example:
        >>> detector = HandDetector(confidence=0.7)
        >>> results = detector.detect(frame)
        >>> print(f"Found {len(results['hands'])} hands")
    """
    
    def __init__(self, confidence: float = 0.7, max_hands: int = 2):
        """Initialize the hand detector.
        
        Args:
            confidence: Minimum detection confidence (0.0-1.0)
            max_hands: Maximum number of hands to track
            
        Raises:
            ValueError: If confidence is not between 0 and 1
        """
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
            
        self.confidence = confidence
        self.max_hands = max_hands
```

### Project Structure

```
src/
├── core/                   # Core functionality
│   ├── __init__.py
│   ├── hand_detector.py    # Hand detection algorithms
│   ├── gesture_recognizer.py # Gesture classification
│   ├── input_controller.py # Input simulation
│   └── yolo_detector.py    # Object detection
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── camera_utils.py     # Camera management
│   ├── config_loader.py    # Configuration handling
│   └── visualization.py    # Debug visualization
└── main.py                # Application entry point

tests/                      # Test files
├── unit/                   # Unit tests
│   ├── test_hand_detector.py
│   ├── test_gesture_recognizer.py
│   └── test_input_controller.py
├── integration/            # Integration tests
│   ├── test_full_pipeline.py
│   └── test_camera_integration.py
└── fixtures/               # Test data
    ├── sample_frames/
    └── config_examples/

docs/                       # Documentation
├── INSTALLATION.md
├── TECHNICAL.md
├── API.md
└── CONTRIBUTING.md
```

### Configuration Management

```python
# Use configuration files for settings
config = ConfigLoader('config/settings.yaml')

# Provide sensible defaults
def get_detection_confidence(config):
    return config.get('hand_detection.confidence', 0.7)

# Document configuration options
# config/settings.yaml
hand_detection:
  confidence: 0.7          # Detection confidence threshold (0.0-1.0)
  max_hands: 2             # Maximum hands to detect (1-4)
  min_detection_confidence: 0.5  # Minimum confidence for detection
```

### Error Handling

```python
import logging

logger = logging.getLogger(__name__)

def detect_hands(frame):
    """Detect hands with proper error handling."""
    try:
        if frame is None:
            raise ValueError("Input frame cannot be None")
            
        # Detection logic here
        results = perform_detection(frame)
        
        if not results:
            logger.warning("No hands detected in frame")
            return {'hands': []}
            
        return results
        
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in hand detection: {e}")
        return {'hands': []}  # Graceful degradation
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/unit/test_hand_detector.py

# Run with coverage
python -m pytest --cov=src

# Run integration tests
python -m pytest tests/integration/

# Run performance tests
python -m pytest tests/performance/ --benchmark-only
```

### Writing Tests

#### Unit Tests
```python
import pytest
import numpy as np
from src.core.hand_detector import HandDetector

class TestHandDetector:
    """Test cases for HandDetector class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.detector = HandDetector(confidence=0.7)
        
    def test_initialization(self):
        """Test detector initialization."""
        assert self.detector.confidence == 0.7
        assert self.detector.max_hands == 2
        
    def test_detect_with_valid_frame(self):
        """Test detection with valid input frame."""
        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Perform detection
        results = self.detector.detect(frame)
        
        # Verify results structure
        assert isinstance(results, dict)
        assert 'hands' in results
        assert isinstance(results['hands'], list)
        
    def test_detect_with_none_frame(self):
        """Test detection with None input."""
        with pytest.raises(ValueError, match="Input frame cannot be None"):
            self.detector.detect(None)
```

#### Integration Tests
```python
def test_full_gesture_pipeline():
    """Test complete gesture recognition pipeline."""
    # Initialize components
    camera = CameraManager(camera_id=0)
    detector = HandDetector()
    recognizer = GestureRecognizer()
    
    # Capture frame
    frame = camera.get_frame()
    assert frame is not None
    
    # Detect hands
    hand_results = detector.detect(frame)
    
    # Recognize gestures
    gestures = recognizer.recognize(hand_results)
    
    # Verify pipeline completion
    assert isinstance(gestures, list)
```

### Performance Tests

```python
import time
import pytest

def test_detection_performance():
    """Test hand detection performance requirements."""
    detector = HandDetector()
    frame = create_test_frame()
    
    # Measure detection time
    start_time = time.time()
    for _ in range(100):
        detector.detect(frame)
    end_time = time.time()
    
    # Calculate average time per detection
    avg_time = (end_time - start_time) / 100
    
    # Assert performance requirement (< 50ms per detection)
    assert avg_time < 0.05, f"Detection too slow: {avg_time:.3f}s"
```

## 📚 Documentation

### Code Documentation

- Use docstrings for all public functions and classes
- Include parameter types and return value descriptions
- Provide usage examples where appropriate
- Document any side effects or exceptions

### User Documentation

- Update README.md for user-facing changes
- Add installation instructions for new dependencies
- Create tutorials for new features
- Update configuration documentation

### API Documentation

```python
def recognize_gesture(hand_landmarks: List[Dict], confidence_threshold: float = 0.7) -> Optional[Dict]:
    """Recognize gesture from hand landmarks.
    
    Analyzes hand landmarks to classify the gesture being performed.
    
    Args:
        hand_landmarks: List of normalized hand landmark coordinates.
            Each landmark is a dict with 'x', 'y', 'z' keys.
        confidence_threshold: Minimum confidence required for gesture
            recognition. Must be between 0.0 and 1.0.
    
    Returns:
        Dict containing gesture information:
            - name: Gesture name (str)
            - confidence: Recognition confidence (float)
            - action: Associated action (str)
        Returns None if no gesture recognized above threshold.
    
    Raises:
        ValueError: If confidence_threshold is not between 0 and 1.
        TypeError: If hand_landmarks is not a list.
    
    Example:
        >>> landmarks = [{'x': 0.5, 'y': 0.5, 'z': 0.0}, ...]
        >>> gesture = recognize_gesture(landmarks, confidence_threshold=0.8)
        >>> if gesture:
        ...     print(f"Recognized: {gesture['name']} ({gesture['confidence']:.2f})")
    """
```

## 🔄 Pull Request Process

### 1. Prepare Your Pull Request

- Ensure your branch is up to date with main
- All tests pass
- Code follows style guidelines
- Documentation is updated

### 2. Create Pull Request

**PR Title Format:**
```
type(scope): brief description

Examples:
feat(detection): add neural network gesture recognition
fix(camera): resolve IP camera connection issues
docs(readme): update installation instructions
```

**PR Description Template:**
```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Screenshots/Videos
If applicable, add screenshots or videos to help explain your changes.

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Code is commented, particularly in hard-to-understand areas
- [ ] Corresponding changes to documentation made
- [ ] No new warnings introduced

## Related Issues
Fixes #(issue_number)
```

### 3. Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and style checks
2. **Code Review**: Maintainers review code for quality and design
3. **Testing**: Manual testing of new features
4. **Approval**: At least one maintainer approval required
5. **Merge**: Squash and merge into main branch

### 4. After Merge

- Delete feature branch
- Update local main branch
- Close related issues
- Update project documentation if needed

## 👥 Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community support
- **Pull Requests**: Code review and collaboration

### Getting Help

1. **Search existing issues** before creating new ones
2. **Check documentation** in the `docs/` directory
3. **Ask questions** in GitHub Discussions
4. **Join community calls** (announced in discussions)

### Recognition

Contributors are recognized in:
- Contributors section of README.md
- Release notes for significant contributions
- Hall of Fame for major contributions

## 📈 Roadmap

See our [project roadmap](https://github.com/IliasSoultana/Minecraft-Hand-Detector/projects) for upcoming features and priorities.

### Current Priorities

1. **Performance Optimization**: Improve FPS and reduce latency
2. **Gesture Accuracy**: Enhance recognition algorithms
3. **Platform Support**: Better macOS and Linux compatibility
4. **Documentation**: Comprehensive user and developer guides

Thank you for contributing to the Minecraft Hand Gesture Controller! Your contributions help make hands-free gaming accessible to everyone. 🎮✋