import torch
from ultralytics import YOLO
import osmnx as ox
import time
import geocoder
import cv2
import pyttsx3
from math import radians, sin, cos, sqrt, atan2

# Check CUDA availability
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. The script will run on CPU, which may be slower.")

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def say_text(text):
    print(text)
    engine.say(text)
    engine.runAndWait()

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

def detect_obstacles(model, frame):
    results = model(frame, device='cuda' if torch.cuda.is_available() else 'cpu')
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

def main():
    destination = (20.348865, 85.816085)  # KP 6
    say_text("Starting real-time navigation to KP 6")

    # Load YOLO model with CUDA support
    model = YOLO('yolov8s.pt')
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        say_text("Unable to access camera")
        return

    route = None
    current_step = 0

    try:
        while True:
            current_pos = get_gps_location()
            if current_pos is None:
                say_text("Waiting for GPS signal...")
                time.sleep(5)
                continue

            if route is None or calculate_distance(current_pos, route[current_step]) > 20:
                say_text("Calculating route...")
                route = find_walking_route(current_pos, destination)
                current_step = 0
                say_text(f"Route calculated with {len(route)} waypoints")

            ret, frame = cap.read()
            if not ret:
                say_text("Failed to capture frame")
                break

            frame, obstacles = detect_obstacles(model, frame)
            cv2.imshow("Obstacle Detection", frame)

            if obstacles:
                for obstacle in obstacles:
                    say_text(f"Caution: {obstacle} detected ahead")

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

            elif current_step == len(route) - 1:
                distance = calculate_distance(current_pos, destination)
                if distance < 10:
                    say_text("You have reached your destination")
                    break
                else:
                    say_text(f"Continue straight for {distance:.0f} meters to reach your destination")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(5)

    except KeyboardInterrupt:
        say_text("Navigation ended")

    except Exception as e:
        say_text(f"An error occurred: {str(e)}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.stop()

if __name__ == "__main__":
    main()
