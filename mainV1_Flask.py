from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import osmnx as ox
import torch
from ultralytics import YOLO
from math import radians, sin, cos, sqrt, atan2
import time

app = Flask(__name__)

# Load YOLO model with CUDA if available
model = YOLO('yolov8s.pt')
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Destination coordinates (KP 6)
DESTINATION = (20.348865, 85.816085)

# In-memory storage for device states
device_storage = {}

def calculate_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (sin(dlat/2) * sin(dlat/2) +
         cos(radians(lat1)) * cos(radians(lat2)) *
         sin(dlon/2) * sin(dlon/2))
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c * 1000  # Meters

def get_direction(prev_point, current_point, next_point):
    lat1, lon1 = prev_point
    lat2, lon2 = current_point
    lat3, lon3 = next_point
    
    bearing1 = atan2(sin(lon2-lon1)*cos(lat2), 
                    cos(lat1)*sin(lat2)-sin(lat1)*cos(lat2)*cos(lon2-lon1))
    bearing2 = atan2(sin(lon3-lon2)*cos(lat3), 
                    cos(lat2)*sin(lat3)-sin(lat2)*cos(lat3)*cos(lon3-lon2))
    
    angle = (bearing2 - bearing1 + 3*3.14159) % (2*3.14159) - 3.14159
    
    if angle > 0.5:
        return "right"
    elif angle < -0.5:
        return "left"
    else:
        return "straight"

@app.route('/start_navigation', methods=['POST'])
def start_navigation():
    data = request.json
    device_id = data['device_id']
    current_pos = (data['latitude'], data['longitude'])
    
    try:
        # Calculate initial route
        graph = ox.graph_from_point(current_pos, dist=1000, network_type='walk')
        start_node = ox.nearest_nodes(graph, current_pos[1], current_pos[0])
        end_node = ox.nearest_nodes(graph, DESTINATION[1], DESTINATION[0])
        route = ox.shortest_path(graph, start_node, end_node, weight='length')
        
        # Store route in device storage
        device_storage[device_id] = {
            'route': [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in route],
            'current_step': 0,
            'last_update': time.time()
        }
        
        return jsonify({
            'status': 'success',
            'message': f'Navigation started with {len(route)} waypoints'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    device_id = data['device_id']
    current_pos = (data['latitude'], data['longitude'])
    img_b64 = data['image']
    
    try:
        # Decode image
        img_bytes = base64.b64decode(img_b64)
        img_np = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        # Object detection
        results = model(frame, device='cuda' if torch.cuda.is_available() else 'cpu')
        obstacles = list(set([results[0].names[int(box.cls.item())] 
                            for box in results[0].boxes if box.conf.item() > 0.5]))
        
        # Navigation logic
        device_data = device_storage.get(device_id)
        if not device_data:
            return jsonify({'status': 'error', 'message': 'Start navigation first'}), 400
        
        route = device_data['route']
        current_step = device_data['current_step']
        
        # Recalculate route if needed
        if calculate_distance(current_pos, route[current_step]) > 20:
            graph = ox.graph_from_point(current_pos, dist=1000, network_type='walk')
            start_node = ox.nearest_nodes(graph, current_pos[1], current_pos[0])
            end_node = ox.nearest_nodes(graph, DESTINATION[1], DESTINATION[0])
            new_route = ox.shortest_path(graph, start_node, end_node, weight='length')
            route = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in new_route]
            current_step = 0
            device_data['route'] = route
            device_data['current_step'] = current_step
        
        # Generate navigation command
        if current_step < len(route) - 1:
            prev_point = route[max(0, current_step - 1)]
            current_point = route[current_step]
            next_point = route[current_step + 1]
            
            distance = calculate_distance(current_pos, current_point)
            direction = get_direction(prev_point, current_point, next_point)
            
            if distance < 10:
                device_data['current_step'] += 1
                next_distance = calculate_distance(current_point, next_point)
                command = f"Turn {direction} in {distance:.0f}m, then continue {next_distance:.0f}m"
            else:
                command = f"Continue straight for {distance:.0f}m"
        else:
            distance = calculate_distance(current_pos, DESTINATION)
            command = f"Destination {distance:.0f}m ahead" if distance > 10 else "You have arrived"
        
        # Update device storage
        device_data['last_update'] = time.time()
        device_storage[device_id] = device_data
        
        return jsonify({
            'status': 'success',
            'command': command,
            'obstacles': obstacles,
            'current_step': current_step,
            'total_steps': len(route)
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
