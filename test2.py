import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import time

model = YOLO('best1.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('Video-76.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1 / fps
my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
print(f"Number of classes: {len(class_list)}")

count = 0
area1 = [(4,2),(8,497),(1016,494),(1016,6)]

# Initialize variables for tracking
total_count = 0
tracked_objects = {}  # Dictionary to store object durations
tracking_threshold = 50  # Adjust this value based on your video's characteristics

while True:    
    ret, frame = cap.read()
    if not ret:
        break  # End of video
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame,(1020,500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
   
    current_objects = []
    
    for index, row in px.iterrows():
        x1, y1, x2, y2 = map(int, row[:4])
        d = int(row[5])
        
        if d < len(class_list):
            c = class_list[d]
        else:
            print(f"Warning: Class index {d} is out of range. Max index is {len(class_list)-1}")
            c = "Unknown"
        
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        result = cv2.pointPolygonTest(np.array(area1,np.int32),((cx,cy)),False)
        if result >= 0:
            current_objects.append((cx, cy))
            matched = False
            for obj_id, (last_pos, duration) in tracked_objects.items():
                if np.sqrt((cx - last_pos[0])**2 + (cy - last_pos[1])**2) < tracking_threshold:
                    tracked_objects[obj_id] = ((cx, cy), duration + frame_time * 3)  # Multiply by 3 because we're processing every 3rd frame
                    matched = True
                    break
            if not matched:
                new_id = len(tracked_objects)
                tracked_objects[new_id] = ((cx, cy), frame_time * 3)  # Start with duration of one frame * 3
                total_count += 1
            
            obj_id = next(id for id, (pos, _) in tracked_objects.items() if pos == (cx, cy))
            duration = tracked_objects[obj_id][1]
            
            cvzone.cornerRect(frame,(x1,y1,x2-x1,y2-y1),3,2)
            cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
            cvzone.putTextRect(frame,f'cow ghee {duration:.2f}s',(x1,y1),1,1)

#    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),2)
    cvzone.putTextRect(frame,f'Total Count: {total_count}',(50,60),1,1)
    
    if tracked_objects:
        max_duration = max(duration for _, duration in tracked_objects.values())
        cvzone.putTextRect(frame,f'Duration: {max_duration:.2f} seconds',(50,100),1,1)
    
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Final Total Count: {total_count}")
if tracked_objects:
    print(f"Max Object Duration: {max(duration for _, duration in tracked_objects.values()):.2f} seconds")