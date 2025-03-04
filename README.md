# Blind Hat: Navigation Assistant for Visually Impaired

## Description

This project is a real-time navigation assistant that uses computer vision and GPS data to guide users to their destination while detecting and warning about obstacles. It's designed to run as a Flask server, potentially on a cloud platform with GPU support for efficient processing.

## Features

- Real-time video processing for obstacle detection
- GPS-based navigation with dynamic route recalculation
- Text-to-speech audio instructions
- Flask server for easy deployment and scalability

## Technologies Used

- Python 3.8+
- Flask
- OpenCV
- PyTorch
- YOLO (You Only Look Once) for object detection
- OSMnx for route planning
- pyttsx3 for text-to-speech conversion
- Geocoder for GPS location

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/iot-lab-kiit/BlindHat.git
   cd BlindHat-main
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the YOLO model:
   ```
   wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt
   ```

## Usage

1. Start the Flask server:
   ```
   python app.py
   ```

2. The server will start running on `http://0.0.0.0:5000`

3. Use the following endpoints:
   - `/video_feed`: GET request to receive the processed video stream
   - `/audio_instructions`: GET request to receive the latest audio instruction
   - `/update_gps`: POST request to update the current GPS location

## API Endpoints

### GET /video_feed

Returns a stream of JPEG images representing the processed video feed with obstacle detection.

### GET /audio_instructions

Returns the latest audio instruction as a JSON object.

Response format:
```
{
  "instruction": "Continue straight for 100 meters"
}
```

### POST /update_gps

Updates the current GPS location.

Request body:
```
{
  "latitude": 20.348865,
  "longitude": 85.816085
}
```

Response:
```
{
  "status": "success"
}
```

## Deployment

To deploy this on a cloud platform:

1. Choose a cloud provider (AWS, Google Cloud, Azure, etc.)
2. Set up a virtual machine with GPU support
3. Install all necessary dependencies
4. Copy the script to the cloud instance
5. Run the Flask server on the cloud instance

Ensure proper security measures (authentication, HTTPS) are implemented for cloud deployment.

## Client-side Implementation

For the client-side (e.g., mobile app), you need to:

1. Capture video frames and send them to the server
2. Periodically send GPS updates to the `/update_gps` endpoint
3. Fetch and play audio instructions from the `/audio_instructions` endpoint
4. Display the video feed from the `/video_feed` endpoint

## Configuration

The destination coordinates are currently hardcoded in the `process_frame` function. To change the destination, modify the following line:

```
destination = (20.348865, 85.816085)  # KP 6
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- YOLOv5 by Ultralytics
- OSMnx by Geoff Boeing
- Flask team for the excellent web framework
- OpenCV contributors
