# Data Directory

This directory is for storing training datasets if you want to collect custom gesture data.

## Structure
```
data/
├── training/           # Training gesture samples
│   ├── idle/          # Idle gesture samples
│   ├── jump/          # Jump gesture samples  
│   ├── left/          # Left gesture samples
│   └── right/         # Right gesture samples
├── validation/        # Validation datasets
└── test/             # Test datasets
```

## Data Collection
Currently the system uses synthetic training data, but you can extend it to collect real gesture samples for improved accuracy.

## File Formats
- Training data: NumPy arrays (.npy files)
- Features: 13-dimensional vectors per gesture sample
- Labels: String class names (idle, jump, left, right)