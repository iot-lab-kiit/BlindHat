# BlindHat

## Overview
This application provides an intelligent navigation system with real-time object detection and routing capabilities, designed to assist users in navigating through an environment while detecting potential obstacles.

## Features
- Real-time navigation routing
- Object detection using YOLOv8
- Dynamic route recalculation
- Obstacle identification
- Distance and direction guidance

## Prerequisites
- Python 3.8+
- Flask
- OpenCV
- PyTorch
- Ultralytics YOLO
- OSMnx
- CUDA (optional, but recommended for GPU acceleration)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/iot-lab-kiit/BlindHat.git
cd BlindHat-main
```

2. Create a virtual environment:
```bash
conda create -n BlindHatEnv
conda activate BlindHatEnv
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration
- Pre-configured with a default destination coordinate (KP 6)
- Supports CUDA for GPU-accelerated object detection
- Uses YOLOv8s model for object detection

## API Endpoints

### `/start_navigation`
- **Method**: POST
- **Payload**: 
  ```json
  {
    "device_id": "unique_device_identifier",
    "latitude": float,
    "longitude": float
  }
  ```
- **Description**: Initializes navigation route from current location to predefined destination

### `/detect`
- **Method**: POST
- **Payload**:
  ```json
  {
    "device_id": "unique_device_identifier",
    "latitude": float,
    "longitude": float,
    "image": "base64_encoded_image"
  }
  ```
- **Returns**: 
  - Navigation command
  - Detected obstacles
  - Current navigation step

## Running the Application
```bash
python app.py
```
The server will start on `0.0.0.0:5000`

## Key Technologies
- Flask for web server
- OSMnx for routing
- YOLOv8 for object detection
- PyTorch for machine learning computations

## Limitations
- Requires internet connection for route calculation
- Object detection accuracy depends on model and image quality
- Currently hardcoded to a specific destination

## Acknowledgments
- Ultralytics for YOLO
- OSMnx for routing capabilities
- Flask community
