"""
Setup script for Minecraft Hand Gesture Controller
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="minecraft-hand-detector",
    version="1.0.0",
    author="Ilias Soultana",
    author_email="your.email@example.com",
    description="A real-time hand gesture recognition system for hands-free Minecraft control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IliasSoultana/Minecraft-Hand-Detector",
    project_urls={
        "Bug Tracker": "https://github.com/IliasSoultana/Minecraft-Hand-Detector/issues",
        "Documentation": "https://github.com/IliasSoultana/Minecraft-Hand-Detector/docs",
        "Source Code": "https://github.com/IliasSoultana/Minecraft-Hand-Detector",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Games/Entertainment",
        "Topic :: Multimedia :: Video :: Capture",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "gpu": [
            "torch>=1.12.0",
            "torchvision>=0.13.0",
        ],
        "advanced": [
            "scikit-learn>=1.0",
            "pandas>=1.5.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "minecraft-gesture-controller=main:main",
            "calibrate-gestures=gestures.calibration:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "models/*.pt", "docs/*"],
    },
    data_files=[
        ("config", ["config/settings.yaml", "config/gestures.yaml", "config/minecraft_controls.yaml"]),
    ],
    zip_safe=False,
    keywords=[
        "minecraft", "hand-tracking", "gesture-recognition", "computer-vision", 
        "mediapipe", "opencv", "yolo", "gaming", "accessibility", "hands-free"
    ],
    license="MIT",
    platforms=["Windows", "macOS", "Linux"],
)