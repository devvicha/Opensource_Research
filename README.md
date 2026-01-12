# ROI-Based Oxygen Saturation Detector

A deep learning-based system for non-invasive oxygen saturation (SpO2) estimation using video analysis and Region of Interest (ROI) detection.

## Overview

This project uses a Tiny Transformer neural network to predict oxygen saturation levels from video frames by analyzing photoplethysmography (PPG) signals extracted from facial regions of interest. The system processes video input, detects facial ROIs, extracts temporal features, and estimates SpO2 levels in real-time.

## Features

- **Deep Learning Model**: Custom Tiny Transformer architecture for time-series analysis
- **Real-time Processing**: FastAPI backend for efficient video frame processing
- **Face Detection**: OpenCV-based ROI detection and tracking
- **Web Interface**: Simple HTML frontend for video upload and result visualization
- **Model Training**: Complete training pipeline with data augmentation and early stopping

## Project Structure

```
├── final.py           # Model architecture and training script
├── backend.py         # FastAPI server for inference
├── frontend/
│   └── home.html      # Web interface
└── training_logs_single/  # Training checkpoints and logs
```

## Technical Details

### Model Architecture
- Tiny Transformer with 4 layers
- 2 attention heads
- 32-dimensional model
- Dropout and DropPath regularization
- Gradient clipping for stable training

### Training Configuration
- Epochs: 200
- Batch size: 32
- Learning rate: 3e-4
- Optimizer: AdamW with weight decay
- Early stopping with patience: 40
- Data augmentation: Time masking, feature masking, jittering, and time shifting

## Requirements

- Python 3.8+
- PyTorch
- FastAPI
- OpenCV (cv2)
- NumPy
- scikit-learn
- uvicorn

## Installation

```bash
pip install torch torchvision
pip install fastapi uvicorn
pip install opencv-python numpy pandas scikit-learn matplotlib
```

## Usage

### Training the Model

```bash
python final.py
```

The training script will:
- Load dataset from the specified directory
- Split data into train/validation/test sets (70/30 split with validation)
- Train the Tiny Transformer model
- Save best model checkpoint and statistics

### Running the Backend Server

```bash
python backend.py
```

The FastAPI server will start on `http://localhost:8000` and provide endpoints for:
- Video frame upload
- ROI detection
- SpO2 prediction

### Using the Web Interface

Open `frontend/home.html` in a web browser to access the user interface for uploading videos and viewing predictions.

## Configuration

Key parameters can be adjusted in `final.py`:
- `DATA_DEFAULT_DIR`: Dataset folder path
- `MAX_T`: Maximum sequence length (default: 200)
- `EPOCHS`: Number of training epochs
- `BATCH`: Batch size
- Model architecture parameters (D_MODEL, N_HEADS, N_LAYERS)

## Model Performance

The model uses:
- Z-score normalization for features and targets
- Mean Squared Error (MSE) loss
- Validation-based early stopping
- Best model selection based on validation loss

## License

This project is available for educational and research purposes.

## Notes

- Ensure proper lighting conditions for accurate face detection
- Video quality significantly impacts prediction accuracy
- Model requires calibration for clinical use
- This is a research prototype and should not be used for medical diagnosis
