import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import cvzone

# Load the YOLO model
model = YOLO('best1.pt')

# Mouse callback function for displaying RGB values
def display_rgb(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse Position - X: {x}, Y: {y}")

# Initialize OpenCV window
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', display_rgb)

# Video input and class labels
video_path = 'testvideo.mp4'
classes_file = 'coco1.txt'
cap = cv2.VideoCapture(video_path)

with open(classes_file, 'r') as file:
    class_list = file.read().splitlines()

# Get video FPS and calculate frame skip
video_fps = cap.get(cv2.CAP_PROP_FPS)
desired_fps = 160
frame_skip = max(int(video_fps / desired_fps), 1)

# Initialize variables
line_y = 411
line_offset = 6
unique_ids = set()
tracker = Tracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames for desired FPS
    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if frame_id % frame_skip != 0:
        continue

    # Resize frame for consistency
    frame = cv2.resize(frame, (1020, 500))

    # Perform object detection
    results = model.predict(frame, verbose=False)
    detections = results[0].boxes.data if results[0].boxes else []

    # Prepare bounding boxes for tracking
    bbox_list = []
    for detection in detections:
        x1, y1, x2, y2, _, class_id = map(int, detection[:6])
        bbox_list.append([x1, y1, x2, y2])

    # Update tracker and get IDs
    tracked_objects = tracker.update(bbox_list)
    for x1, y1, x2, y2, obj_id in tracked_objects:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw detections
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
        cvzone.putTextRect(frame, f'ID: {obj_id}', (x1, y1 - 10), 1, 1)

        # Check crossing line condition
        if line_y - line_offset <= cy <= line_y + line_offset and obj_id not in unique_ids:
            unique_ids.add(obj_id)

    # Display object count
    cv2.putText(frame, f'Count: {len(unique_ids)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.line(frame, (100, line_y), (1000, line_y), (255, 255, 255), 2)

    # Show video
    cv2.imshow("RGB", frame)

    # Exit on 'Esc' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
