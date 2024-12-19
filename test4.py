import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import time

# Load the model
model = YOLO('best1.pt')

# Load class names
with open("coco1.txt", "r") as my_file:
    class_list = my_file.read().strip().split("\n")
print(f"Loaded classes: {class_list}")
print(f"Number of classes: {len(class_list)}")

# Initialize video capture
cap = cv2.VideoCapture('Video-76.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1 / fps

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output1.mp4', fourcc, fps, (1020, 500))

# Define area of interest (invisible polygon)
area1 = [(4,2),(8,497),(1016,494),(1016,6)]

# Initialize variables for tracking
total_count = 0
tracked_objects = {}
tracking_threshold = 50  # Adjust this value based on your video's characteristics

# Set confidence and IOU thresholds
confidence_threshold = 0.5  # Adjust as needed
iou_threshold = 0.5  # Adjust as needed

# Speed up factor (e.g., 2 for double speed, 4 for quadruple speed)
speed_up_factor = 1

while True:    
    ret, frame = cap.read()
    if not ret:
        print("End of video reached")
        break  # End of video

    frame = cv2.resize(frame, (1020, 500))

    # Predict with the model
    results = model(frame, conf=confidence_threshold, iou=iou_threshold)[0]

    # Process detections
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf)
        class_id = int(box.cls)

        if 0 <= class_id < len(class_list):
            class_name = class_list[class_id]
        else:
            print(f"Warning: Class index {class_id} is out of range. Max index is {len(class_list)-1}")
            class_name = "Cow Ghee"

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False) >= 0:
            matched = False
            for obj_id, (last_pos, total_duration, last_seen) in tracked_objects.items():
                if np.sqrt((cx - last_pos[0])**2 + (cy - last_pos[1])**2) < tracking_threshold:
                    time_since_last_seen = frame_time * speed_up_factor
                    tracked_objects[obj_id] = ((cx, cy), total_duration + time_since_last_seen, 0)
                    matched = True
                    break
            if not matched:
                new_id = len(tracked_objects)
                tracked_objects[new_id] = ((cx, cy), frame_time * speed_up_factor, 0)
                total_count += 1

            obj_id = next(id for id, (pos, _, _) in tracked_objects.items() if pos == (cx, cy))
            total_duration = tracked_objects[obj_id][1]

            cvzone.cornerRect(frame, (x1, y1, x2-x1, y2-y1), 3, 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            cvzone.putTextRect(frame, f'{class_name} {total_duration:.2f}s', (x1, y1), 1, 1)

    # Update last_seen for objects not detected in this frame
    for obj_id in tracked_objects:
        if tracked_objects[obj_id][2] == 0:
            tracked_objects[obj_id] = (tracked_objects[obj_id][0], tracked_objects[obj_id][1], frame_time * speed_up_factor)
        else:
            tracked_objects[obj_id] = (tracked_objects[obj_id][0], tracked_objects[obj_id][1], tracked_objects[obj_id][2] + frame_time * speed_up_factor)

    # Display total count and total duration
    cvzone.putTextRect(frame, f'Total Count: {total_count}', (50, 60), 1, 1)
    if tracked_objects:
        total_duration = sum(duration for _, duration, _ in tracked_objects.values())
        cvzone.putTextRect(frame, f'Total Duration: {total_duration:.2f} seconds', (50, 100), 1, 1)

    # Write the frame to the output video
    out.write(frame)

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Final Total Count: {total_count}")
if tracked_objects:
    print(f"Total Object Duration: {sum(duration for _, duration, _ in tracked_objects.values()):.2f} seconds")
print("Video processing completed. Result saved as 'output1.mp4'")