from flask import Flask, Response, jsonify
import cv2
import threading
import queue
import torch
from ultralytics import YOLO
import osmnx as ox
import time
import geocoder
import pyttsx3
from math import radians, sin, cos, sqrt, atan2
import io
from PIL import Image
import base64

app = Flask(__name__)

# Global variables
frame_queue = queue.Queue(maxsize=10)
audio_queue = queue.Queue(maxsize=10)
gps_queue = queue.Queue(maxsize=1)
model = YOLO('yolov8s.pt')
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def say_text(text):
    print(text)
    audio_queue.put(text)

def get_gps_location():
    g = geocoder.ip('me')
    return g.latlng if g.latlng is not None else None

def calculate_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (sin(dlat/2) * sin(dlat/2) +
         cos(radians(lat1)) * cos(radians(lat2)) *
         sin(dlon/2) * sin(dlon/2))
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c * 1000

def find_walking_route(start_point, end_point):
    graph = ox.graph_from_point(start_point, dist=1000, network_type='walk')
    start_node = ox.nearest_nodes(graph, start_point[1], start_point[0])
    end_node = ox.nearest_nodes(graph, end_point[1], end_point[0])
    route = ox.shortest_path(graph, start_node, end_node, weight='length')
    return [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in route]

def detect_obstacles(frame):
    results = model(frame, device='cuda' if torch.cuda.is_available() else 'cpu')
    critical_objects = {'person','car', 'bicycle', 'motorcycle', 'bus', 'truck'}
    detected_objects = []
    
    for result in results:
        for box in result.boxes:
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            label = result.names[class_id]
            
            if confidence > 0.5:
                detected_objects.append(label)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if label in critical_objects:
                    say_text(f"Warning: {label} ahead")
    
    return frame, list(set(detected_objects))

def get_direction(prev_point, current_point, next_point):
    lat1, lon1 = prev_point
    lat2, lon2 = current_point
    lat3, lon3 = next_point
    
    bearing1 = atan2(sin(lon2-lon1)*cos(lat2), cos(lat1)*sin(lat2)-sin(lat1)*cos(lat2)*cos(lon2-lon1))
    bearing2 = atan2(sin(lon3-lon2)*cos(lat3), cos(lat2)*sin(lat3)-sin(lat2)*cos(lat3)*cos(lon3-lon2))
    
    angle = (bearing2 - bearing1 + 3*3.14159) % (2*3.14159) - 3.14159
    
    if angle > 0.5:
        return "Take a right"
    elif angle < -0.5:
        return "Take a left"
    else:
        return "Continue straight"

def process_frame():
    destination = (20.348865, 85.816085)  # KP 6
    route = None
    current_step = 0

    while True:
        if frame_queue.empty():
            time.sleep(0.1)
            continue

        frame = frame_queue.get()
        current_pos = gps_queue.get() if not gps_queue.empty() else None

        if current_pos is None:
            say_text("Waiting for GPS signal...")
            time.sleep(5)
            continue

        if route is None or calculate_distance(current_pos, route[current_step]) > 30:
            say_text("Re-aligning to route...")
            route = find_walking_route(current_pos, destination)
            current_step = 0
            say_text(f"Route adjusted with {len(route)} waypoints")

        frame, obstacles = detect_obstacles(frame)

        if current_step < len(route) - 1:
            prev_point = route[max(0, current_step - 1)]
            current_point = route[current_step]
            next_point = route[current_step + 1]
            
            distance = calculate_distance(current_pos, current_point)
            next_distance = calculate_distance(current_point, next_point)

            direction = get_direction(prev_point, current_point, next_point)
            
            if distance < 10:
                say_text(f"{direction} in {distance:.0f} meters, then continue for {next_distance:.0f} meters")
                current_step += 1
            else:
                say_text(f"Continue straight for {distance:.0f} meters")

        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        frame_queue.put(frame_bytes)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if frame_queue.empty():
                time.sleep(0.1)
                continue
            frame = frame_queue.get()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/audio_instructions')
def audio_instructions():
    if not audio_queue.empty():
        return jsonify({'instruction': audio_queue.get()})
    return jsonify({'instruction': ''})

@app.route('/update_gps', methods=['POST'])
def update_gps():
    gps_data = request.json
    if 'latitude' in gps_data and 'longitude' in gps_data:
        gps_queue.put((gps_data['latitude'], gps_data['longitude']))
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Invalid GPS data'})

if __name__ == '__main__':
    threading.Thread(target=process_frame, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
