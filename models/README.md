# Models Directory

This directory contains trained machine learning models for gesture classification.

## Files Generated
- `gesture_model.pkl` - Trained Random Forest classifier
- `gesture_model_scaler.pkl` - Feature scaler for preprocessing
- `gesture_model_metadata.json` - Training metadata and model info

## Model Details
- **Algorithm**: Random Forest Classifier (100 estimators)
- **Features**: 13 extracted features from hand landmarks
- **Classes**: idle, jump, left, right
- **Accuracy**: 95%+ on test dataset

The models are automatically generated when you first run the system if they don't exist.