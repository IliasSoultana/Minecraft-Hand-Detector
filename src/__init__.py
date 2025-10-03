"""
Minecraft Hand Gesture Controller

A revolutionary real-time hand gesture recognition system that enables 
hands-free control of Minecraft using computer vision and machine learning.
"""

__version__ = "1.0.0"
__author__ = "Ilias Soultana"
__email__ = "ilias.soultana@example.com"
__license__ = "MIT"
__copyright__ = "2025 Ilias Soultana"

# Project metadata
PROJECT_NAME = "Minecraft Hand Gesture Controller"
PROJECT_DESCRIPTION = "Real-time hand gesture recognition for hands-free Minecraft control"
PROJECT_URL = "https://github.com/IliasSoultana/Minecraft-Hand-Detector"
PROJECT_REPO = "https://github.com/IliasSoultana/Minecraft-Hand-Detector.git"

# Version info
VERSION_INFO = {
    "major": 1,
    "minor": 0,
    "patch": 0,
    "release": "stable"
}

def get_version():
    """Get the current version string."""
    return __version__

def get_version_info():
    """Get detailed version information."""
    return VERSION_INFO

def print_banner():
    """Print project banner."""
    banner = f"""
    🎮 {PROJECT_NAME} v{__version__}
    {'='*50}
    {PROJECT_DESCRIPTION}
    
    Author: {__author__}
    License: {__license__}
    Repository: {PROJECT_URL}
    {'='*50}
    """
    print(banner)

# Module exports
__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "__license__",
    "__copyright__",
    "PROJECT_NAME",
    "PROJECT_DESCRIPTION", 
    "PROJECT_URL",
    "PROJECT_REPO",
    "VERSION_INFO",
    "get_version",
    "get_version_info",
    "print_banner"
]