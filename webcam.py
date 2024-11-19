# import cv2
# import pandas as pd
# from PIL.ImageChops import offset
# from ultralytics import YOLO
# from tracker import *
# import cvzone
#
# # Load the YOLO model
# model = YOLO('best1.pt')
#
# # Mouse callback function for RGB values
# def RGB(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE:
#         colorsBGR = [x, y]
#         print(colorsBGR)
#
# # Set up the OpenCV window and callback
# cv2.namedWindow('RGB')
# cv2.setMouseCallback('RGB', RGB)
#
# # Open the video capture (for webcam use cv2.VideoCapture(0))
# cap = cv2.VideoCapture(0)  # Use 0 for webcam or 'vid.mp4' for a video file
#
# # Read class names from the file
# my_file = open("coco1.txt", "r")
# data = my_file.read()
# class_list = data.split("\n")
#
# count = 0
# tracker = Tracker()
# cy1 = 311
# offset = 6
# pack = []
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     count += 1
#     if count % 3 != 0:
#         continue
#
#     # Resize the frame for better processing speed (optional)
#     frame = cv2.resize(frame, (1020, 500))
#
#     # Perform object detection with YOLO model
#     results = model.predict(frame)
#
#     # Check if any boxes were detected
#     if results[0].boxes is not None:
#         a = results[0].boxes.data
#         px = pd.DataFrame(a).astype("float")
#
#         list = []
#
#         # Process detected objects
#         for index, row in px.iterrows():
#             x1 = int(row[0])
#             y1 = int(row[1])
#             x2 = int(row[2])
#             y2 = int(row[3])
#             d = int(row[5])
#             c = class_list[d]
#
#             list.append([x1, y1, x2, y2])
#
#         # Update tracker with the list of bounding boxes
#         bbox_idx = tracker.update(list)
#
#         # Draw bounding boxes and IDs
#         for bbox in bbox_idx:
#             x3, y3, x4, y4, id = bbox
#             cx = int(x3 + x4) // 2
#             cy = int(y3 + y4) // 2
#             cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
#             cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
#             cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
#             if cy1 < (cy + offset) and cy1 > (cy - offset):
#                 cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
#                 cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
#                 cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
#                 if pack.count(id) == 0:
#                     pack.append(id)
#     else:
#         print("No objects detected")
#
#     # Print and display the count of detected unique objects
#     print(len(pack))
#     cv2.putText(frame, f'Count: {len(pack)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#     cv2.line(frame, (259, cy1), (547, cy1), (255, 255, 255), 2)
#
#     # Display the result
#     cv2.imshow("RGB", frame)
#
#     # Break loop if the 'Esc' key is pressed
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#
# # Release the video capture and close all windows
# cap.release()
# cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO
from tracker import Tracker
import cvzone

# Load YOLO model
model = YOLO('best1.pt')

# Initialize Tracker
tracker = Tracker()

# Get class names
with open("coco1.txt", "r") as f:
    class_list = f.read().splitlines()

# Line parameters for counting objects
line_y = 311
line_offset = 6
unique_ids = set()

# Open webcam or video feed (0 for default webcam)
cap = cv2.VideoCapture(0)

# Check if video source is available
if not cap.isOpened():
    print("Error: Cannot open webcam or video source.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (1020, 500))

    # Perform inference
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

        # Draw bounding boxes and IDs
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
        cvzone.putTextRect(frame, f'ID: {obj_id}', (x1, y1 - 10), 1, 1)

        # Check crossing line condition
        if line_y - line_offset <= cy <= line_y + line_offset and obj_id not in unique_ids:
            unique_ids.add(obj_id)

    # Display count
    cv2.putText(frame, f'Count: {len(unique_ids)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.line(frame, (100, line_y), (1000, line_y), (255, 255, 255), 2)

    # Show frame
    cv2.imshow("Live Inference", frame)

    # Exit on 'Esc' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
