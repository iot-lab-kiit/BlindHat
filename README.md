# Blind Hat: Navigation Assistant for the Visually Impaired

## 🧭 Description

**Blind Hat** is a real-time smart navigation assistant that helps visually impaired individuals by guiding them safely using computer vision and GPS. It uses advanced obstacle detection and voice-based instructions to help users move confidently and independently.

🌐 Powered by a Flask server, the system runs locally or on GPU-enabled cloud platforms for real-time performance.

---

## ✨ Features

- 🎯 **Real-time Obstacle Detection** with YOLOv8
- 📍 **GPS-based Navigation** with dynamic route updates
- 🗣️ **Text-to-Speech Voice Alerts** for instructions & warnings
- 🌐 **RESTful Flask API** for integration with mobile or wearable clients
- ⚙️ **GPU Acceleration Support** for high-performance processing

---

## 🛠️ Technologies Used

| Tech | Purpose |
|------|---------|
| 🐍 Python 3.8+ | Backend & core logic |
| 🔥 PyTorch | Deep learning model (YOLOv8) |
| 📷 OpenCV | Video frame processing |
| 🧠 YOLOv8 | Object detection |
| 🌐 Flask | Web server & APIs |
| 🗺️ OSMnx | Route planning & shortest path |
| 🗣️ pyttsx3 | Text-to-speech |
| 📡 Geocoder | Location via IP |

---

## 🚀 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/iot-lab-kiit/BlindHat.git
   cd BlindHat-main
   ```

2. **(Optional)** Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv8 Model**
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
   ```

---

## ▶️ Usage

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Access the server at:
   ```
   http://0.0.0.0:5000
   ```

---

## 📡 API Endpoints

### 🎥 `GET /video_feed`
- Returns a real-time video stream (JPEG-encoded) with obstacle detection.

---

### 🔊 `GET /audio_instructions`
- Returns the latest voice instruction.

Response:
```json
{
  "instruction": "Continue straight for 100 meters"
}
```

---

### 📍 `POST /update_gps`
- 📍 Updates the current location for route planning.

Request:
```json
{
  "latitude": 20.348865,
  "longitude": 85.816085
}
```

Response:
```json
{
  "status": "success"
}
```

---

## ☁️ Deployment

1. Choose a cloud provider (AWS, GCP, Azure)
2. Spin up a GPU-enabled VM (CUDA supported)
3. Install dependencies & upload the project
4. Run the Flask server:
   ```bash
   python app.py
   ```
5. Ensure secure access with HTTPS & authentication for production use

---

## 📱 Client-Side Integration

To use Blind Hat from a client device like a mobile app or wearable:

- Send captured video frames to the server
- Periodically update the GPS coordinates via `/update_gps`
- Fetch & play audio alerts from `/audio_instructions`
- Stream processed video from `/video_feed`

---

## ⚙️ Configuration

The destination coordinates are currently hardcoded in `process_frame()`:

```python
destination = (20.348865, 85.816085)  # KP 6
```

📍 Update this line to change the navigation destination.

---

## 📄 License

Licensed under the MIT License. See [LICENSE.md](LICENSE.md) for full details.

---

## 🙌 Acknowledgments

- YOLOv8 by [Ultralytics](https://github.com/ultralytics/yolov5)
- OSMnx by Geoff Boeing
- Flask team for their minimalistic web framework
- OpenCV community for making vision accessible
