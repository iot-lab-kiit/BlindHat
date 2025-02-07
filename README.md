# BlindHat

## Overview
An intelligent navigation system using real-time object detection and routing to assist users in navigating environments while detecting potential obstacles.

## Features
- Real-time navigation routing
- YOLOv8 object detection
- Dynamic route recalculation
- Obstacle identification
- Distance and direction guidance

## Prerequisites
- Python 3.8+
- Flask, OpenCV, PyTorch
- Ultralytics YOLO, OSMnx
- CUDA (optional)

## Installation
1. Clone repository:
```bash
git clone https://github.com/iot-lab-kiit/BlindHat.git
cd BlindHat-main
```

2. Create virtual environment:
```bash
conda create -n BlindHatEnv
conda activate BlindHatEnv
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

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

## Running the Application
```bash
python app.py
```
Server starts on `0.0.0.0:5000`

## Key Technologies
- Flask
- OSMnx
- YOLOv8
- PyTorch

## Limitations
- Requires internet connection
- Object detection accuracy varies
- Hardcoded destination

## Acknowledgments
- Ultralytics
- OSMnx
- Flask community

## License
MIT License

Copyright (c) 2024 IoT Lab KIIT

Permission is granted to use, modify, and distribute this software freely.
